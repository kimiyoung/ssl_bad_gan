import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import torch.nn.init as nn_init

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class WN_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Linear, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.weight_scale = Parameter(torch.ones(self.out_features))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_features))

        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(0, std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # normalize weight matrix and linear projection
        norm_weight = self.weight * (weight_scale.unsqueeze(1) / torch.sqrt((self.weight ** 2).sum(1) + 1e-6)).expand_as(self.weight)
        activation = F.linear(input, norm_weight)

        if self.init_mode == True:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(0) + 1e-6).squeeze(0)
            activation = activation * inv_stdv.expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias.expand_as(activation)

        return activation

class WN_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))
        
        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [out x in x h x w]
        # for each output dimension, normalize through (in, h, w) = (1, 2, 3) dims
        norm_weight = self.weight * (weight_scale[:,None,None,None] / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(1) + 1e-6)).expand_as(self.weight)
        activation = F.conv2d(input, norm_weight, bias=None, 
                              stride=self.stride, padding=self.padding, 
                              dilation=self.dilation, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None].expand_as(activation)

        return activation

class WN_ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))
        
        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input, output_size=None):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [in x out x h x w]
        # for each output dimension, normalize through (in, h, w)  = (0, 2, 3) dims
        norm_weight = self.weight * (weight_scale[None,:,None,None] / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(0) + 1e-6)).expand_as(self.weight)
        output_padding = self._output_padding(input, output_size)
        activation = F.conv_transpose2d(input, norm_weight, bias=None, 
                                        stride=self.stride, padding=self.padding, 
                                        output_padding=output_padding, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None].expand_as(activation)

        return activation

class Discriminative(nn.Module):
    def __init__(self, config):
        super(Discriminative, self).__init__()

        print '===> Init small-conv for {}'.format(config.dataset)

        self.noise_size = config.noise_size
        self.num_label  = config.num_label

        if config.dataset == 'svhn':
            n_filter_1, n_filter_2 = 64, 128
        elif config.dataset == 'cifar':
            n_filter_1, n_filter_2 = 96, 192
        else:
            raise ValueError('dataset not found: {}'.format(config.dataset))

        # Assume X is of size [batch x 3 x 32 x 32]
        self.core_net = nn.Sequential(

            nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)) if config.dataset == 'svhn' \
                else nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.2)),

            WN_Conv2d(         3, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5) if config.dataset == 'svhn' else nn.Dropout(0.5),

            WN_Conv2d(n_filter_1, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(0.5) if config.dataset == 'svhn' else nn.Dropout(0.5),

            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),

            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )

        self.out_net = WN_Linear(n_filter_2, self.num_label, train_scale=True, init_stdv=0.1)

    def forward(self, X, feat=False):
        if X.dim() == 2:
            X = X.view(X.size(0), 3, 32, 32)
        
        if feat:
            return self.core_net(X)
        else:
            return self.out_net(self.core_net(X))

class Generator(nn.Module):
    def __init__(self, image_size, noise_size=100, large=False):
        super(Generator, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size

        if not large:
            self.core_net = nn.Sequential(
                nn.Linear(self.noise_size, 4 * 4 * 512, bias=False), nn.BatchNorm1d(4 * 4 * 512), nn.ReLU(), 
                Expression(lambda tensor: tensor.view(tensor.size(0), 512, 4, 4)),
                nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                WN_ConvTranspose2d(128,   3, 5, 2, 2, 1, train_scale=True, init_stdv=0.1), nn.Tanh(),
            )
        else:
            self.core_net = nn.Sequential(
                nn.Linear(self.noise_size, 2 * 2 * 1024, bias=False), nn.BatchNorm1d(2 * 2 * 1024), nn.ReLU(), 
                Expression(lambda tensor: tensor.view(tensor.size(0), 1024, 2, 2)),
                nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                WN_ConvTranspose2d(128,   3, 5, 1, 2, 0, train_scale=True, init_stdv=0.1), nn.Tanh(),
            )

    def forward(self, noise):        
        output = self.core_net(noise)

        return output

class Encoder(nn.Module):
    def __init__(self, image_size, noise_size=100, output_params=False):
        super(Encoder, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size

        self.core_net = nn.Sequential(
            nn.Conv2d(  3, 128, 5, 2, 2, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 5, 2, 2, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            Expression(lambda tensor: tensor.view(tensor.size(0), 512 * 4 * 4)),
        )
        
        if output_params:
            self.core_net.add_module(str(len(self.core_net._modules)), WN_Linear(4 * 4 * 512, self.noise_size*2, train_scale=True, init_stdv=0.1))
            self.core_net.add_module(str(len(self.core_net._modules)), Expression(lambda x: torch.chunk(x, 2, 1)))
        else:
            self.core_net.add_module(str(len(self.core_net._modules)), WN_Linear(4 * 4 * 512, self.noise_size, train_scale=True, init_stdv=0.1))

    def forward(self, input):
        
        output = self.core_net(input)

        return output
