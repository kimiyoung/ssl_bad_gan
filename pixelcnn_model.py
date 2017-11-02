import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter

old_version = False

class GaussianConditional(nn.Module):
    def __init__(self, dim, clip=False):
        super(GaussianConditional, self).__init__()
        self.dim = dim

        self.sp = nn.Softplus()
        self.clip = clip

    def forward(self, input):
        mu = input.narrow(1, 0, self.dim)
        log_sigma = input.narrow(1, self.dim, self.dim)

        if self.clip:
            output = Variable(input.data.new(input.size(0), self.dim).normal_()) * torch.exp(torch.min(log_sigma, Variable(log_sigma.data.new(log_sigma.size()).fill_(10.0)))) + mu
        else:
            output = Variable(input.data.new(input.size(0), self.dim).normal_()) * torch.exp(log_sigma) + mu

        return output

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

def assert_nan(x):
    assert not np.isnan(x.data.cpu().numpy().sum())

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
        if old_version:
            bias = self.bias
        else:
            bias = None
        activation = F.conv2d(input, norm_weight, bias=bias, 
                              stride=self.stride, padding=self.padding, 
                              dilation=self.dilation, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-8).squeeze()
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
        if old_version:
            bias = self.bias
        else:
            bias = None
        activation = F.conv_transpose2d(input, norm_weight, bias=bias, 
                                        stride=self.stride, padding=self.padding, 
                                        output_padding=output_padding, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-8).squeeze()
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

