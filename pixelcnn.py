
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pixelcnn_model as model
import numpy as np

def concat_elu(x):
    return F.elu(torch.cat([x, -x], 1))

def assert_nan(x):
    assert not np.isnan(x.data.cpu().numpy().sum())

class PixelCNN(nn.Module):

    def __init__(self, nr_filters=160, nr_resnet=5, nr_logistic_mix=10, disable_third=False, dropout_p=0.5, n_channel=3, image_wh=32):
        super(PixelCNN, self).__init__()
        self.nr_filters = nr_filters
        self.nr_resnet = nr_resnet
        self.nr_logistic_mix = nr_logistic_mix
        self.disable_third = disable_third
        self.dropout_p = dropout_p
        self.n_channel = n_channel

        self.forward(Variable(torch.Tensor(2, n_channel, image_wh, image_wh).random_().cuda()))

    def down_shifted_conv2d(self, x, num_filters, filter_size=(2, 3), stride=(1, 1), **kwargs):
        x = F.pad(x, (int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2), filter_size[0] - 1, 0))
        module = getattr(self, str(self.counter), None)
        if module is None:
            module = model.WN_Conv2d(x.size(1), num_filters, filter_size, stride, train_scale=True, **kwargs).cuda()
            self.add_module(str(self.counter), module)
        self.counter += 1
        return module(x)

    def down_shifted_deconv2d(self, x, num_filters, filter_size=(2, 3), stride=(1, 1), **kwargs):
        module = getattr(self, str(self.counter), None)
        if module is None:
            module = model.WN_ConvTranspose2d(x.size(1), num_filters, filter_size, stride, output_padding=1, train_scale=True, **kwargs).cuda()
            self.add_module(str(self.counter), module)
        self.counter += 1
        x = module(x)
        xs = x.size()
        return x[:, :, :(xs[2] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[3] - int((filter_size[1] - 1) / 2))]

    def down_right_shifted_conv2d(self, x, num_filters, filter_size=(2, 2), stride=(1, 1), **kwargs):
        x = F.pad(x, (filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        module = getattr(self, str(self.counter), None)
        if module is None:
            module = model.WN_Conv2d(x.size(1), num_filters, filter_size, stride, train_scale=True, **kwargs).cuda()
            self.add_module(str(self.counter), module)
        self.counter += 1
        return module(x)

    def down_right_shifted_deconv2d(self, x, num_filters, filter_size=(2, 2), stride=(1, 1), **kwargs):
        module = getattr(self, str(self.counter), None)
        if module is None:
            module = model.WN_ConvTranspose2d(x.size(1), num_filters, filter_size, stride, output_padding=1, train_scale=True, **kwargs).cuda()
            self.add_module(str(self.counter), module)
        self.counter += 1
        x = module(x)
        xs = x.size()
        return x[:, :, :(xs[2] - filter_size[0] + 1):, :(xs[3] - filter_size[1] + 1)]

    def down_shift(self, x):
        xs = x.size()
        return torch.cat([Variable(torch.zeros(xs[0], xs[1], 1, xs[3]).cuda()), x[:, :, :xs[2] - 1, :]], 2)

    def right_shift(self, x):
        xs = x.size()
        return torch.cat([Variable(torch.zeros(xs[0], xs[1], xs[2], 1).cuda()), x[:, :, :, :xs[3] - 1]], 3)

    def gated_resnet(self, x, a=None, nonlinearity=concat_elu, conv=None, dropout_p=None, **kwargs):
        dropout_p = self.dropout_p

        xs = x.size()
        num_filters = xs[1]

        c1 = conv(nonlinearity(x), num_filters)
        if a is not None:
            c1 += self.nin(nonlinearity(a), num_filters)
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = F.dropout(c1, dropout_p, training=self.training)
        c2 = conv(c1, num_filters * 2, init_stdv=0.1)
        a, b = torch.split(c2, num_filters, 1)
        c3 = a * F.sigmoid(b)
        return x + c3

    def nin(self, x, num_units, **kwargs):
        module = getattr(self, str(self.counter), None)
        if module is None:
            module = model.WN_Conv2d(x.size(1), num_units, 1, train_scale=True, **kwargs).cuda()
            self.add_module(str(self.counter), module)
        self.counter += 1
        xs = x.size()
        return module(x)

    def forward(self, input):
        # input size [B, H, W, Channel]

        self.counter = 0

        xs = input.size()
        x_pad = torch.cat([input, Variable(torch.ones(xs[0], 1, xs[2], xs[3]).cuda())], 1)
        u_list = [self.down_shift(self.down_shifted_conv2d(x_pad, self.nr_filters, filter_size=(2, 3)))]
        ul_list = [self.down_shift(self.down_shifted_conv2d(x_pad, self.nr_filters, filter_size=(1, 3))) + self.right_shift(self.down_right_shifted_conv2d(x_pad, self.nr_filters, filter_size=(2, 1)))]

        for rep in range(self.nr_resnet):
            u_list.append(self.gated_resnet(u_list[-1], conv=self.down_shifted_conv2d))
            ul_list.append(self.gated_resnet(ul_list[-1], u_list[-1], conv=self.down_right_shifted_conv2d))

        u_list.append(self.down_shifted_conv2d(u_list[-1], self.nr_filters, stride=(2, 2)))
        ul_list.append(self.down_right_shifted_conv2d(ul_list[-1], self.nr_filters, stride=(2, 2)))

        for rep in range(self.nr_resnet):
            u_list.append(self.gated_resnet(u_list[-1], conv=self.down_shifted_conv2d))
            ul_list.append(self.gated_resnet(ul_list[-1], u_list[-1], conv=self.down_right_shifted_conv2d))

        if not self.disable_third:
            u_list.append(self.down_shifted_conv2d(u_list[-1], self.nr_filters, stride=(2, 2)))
            ul_list.append(self.down_right_shifted_conv2d(ul_list[-1], self.nr_filters, stride=(2, 2)))

            for rep in range(self.nr_resnet):
                u_list.append(self.gated_resnet(u_list[-1], conv=self.down_shifted_conv2d))
                ul_list.append(self.gated_resnet(ul_list[-1], u_list[-1], conv=self.down_right_shifted_conv2d))

        # down pass
        u = u_list.pop()
        ul = ul_list.pop()

        for rep in range(self.nr_resnet):
            u = self.gated_resnet(u, u_list.pop(), conv=self.down_shifted_conv2d)
            ul = self.gated_resnet(ul, torch.cat([u, ul_list.pop()], 1), conv=self.down_right_shifted_conv2d)

        u = self.down_shifted_deconv2d(u, self.nr_filters, stride=(2, 2))
        ul = self.down_right_shifted_deconv2d(ul, self.nr_filters, stride=(2, 2))

        for rep in range(self.nr_resnet + 1):
            u = self.gated_resnet(u, u_list.pop(), conv=self.down_shifted_conv2d)
            ul = self.gated_resnet(ul, torch.cat([u, ul_list.pop()], 1), conv=self.down_right_shifted_conv2d)

        if not self.disable_third:
            u = self.down_shifted_deconv2d(u, self.nr_filters, stride=(2, 2))
            ul = self.down_right_shifted_deconv2d(ul, self.nr_filters, stride=(2, 2))

            for rep in range(self.nr_resnet + 1):
                u = self.gated_resnet(u, u_list.pop(), conv=self.down_shifted_conv2d)
                ul = self.gated_resnet(ul, torch.cat([u, ul_list.pop()], 1), conv=self.down_right_shifted_conv2d)

        if self.n_channel == 3:
            x_out = self.nin(F.elu(ul), 10 * self.nr_logistic_mix)
        else:
            x_out = self.nin(F.elu(ul), 3 * self.nr_logistic_mix)

        return x_out

if __name__ == '__main__':
    m = PixelCNN(nr_resnet=3, disable_third=True, n_channel=1, image_wh=28).cuda()
    x = Variable(torch.Tensor(10, 1, 28, 28).random_().cuda())
    t = m(x)
    assert_nan(t)
    print t.size()
