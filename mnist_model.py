
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

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

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, b=True, g=True):
        super(BatchNorm1d, self).__init__()
        self.b = b
        self.g = g
        self.core = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=(b and g))
        print self.core
        if (not b) and g:
            self.g = Parameter(torch.Tensor(num_features))
        elif (not g) and b:
            self.b = Parameter(torch.Tensor(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        if (not self.b) and self.g:
            self.g.data.fill_(1)
        elif (not self.g) and self.b:
            self.b.data.zero_()

    def forward(self, input):
        output = self.core(input)
        if (not self.b) and self.g:
            output = output * self.g.expand_as(output)
        elif (not self.g) and self.b:
            output = output + self.b.expand_as(output)

        return output

class WN_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, train_scale=False):
        super(WN_Linear, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.weight_scale = Parameter(torch.ones(self.out_features))
        else:
            self.register_buffer('weight_scale', Variable(torch.ones(self.out_features)))
        self.init_mode = False

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.1)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        # normalize weight matrix and linear projection
        norm_weight = self.weight * (self.weight_scale.unsqueeze(1) / torch.sqrt((self.weight ** 2).sum(1) + 1e-6)).expand_as(self.weight)
        activation = F.linear(input, norm_weight)

        if self.init_mode == True:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.expand_as(activation)

            stdv_act = torch.sqrt((activation ** 2).mean(0) + 1e-6).squeeze(0)
            activation = activation / stdv_act.expand_as(activation)

            self.weight_scale.data = self.weight_scale.data / stdv_act.data
            self.bias.data = - mean_act.data / stdv_act.data

        else:
            if self.bias is not None:
                activation = activation + self.bias.expand_as(activation)

        return activation
            

class Generator(nn.Module):
    def __init__(self, image_size, noise_size=100):
        super(Generator, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size

        self.core_net = nn.Sequential(
            nn.Linear(noise_size, 500, bias=False), nn.BatchNorm1d(500), nn.Softplus(), 
            nn.Linear(500, 500, bias=False),    nn.BatchNorm1d(500), nn.Softplus(), 
            WN_Linear(500, self.image_size, train_scale=True),    nn.Sigmoid()
        )

    def forward(self, noise):
        output = self.core_net(noise)

        return output

class Discriminative(nn.Module):
    def __init__(self, noise_size, num_label, image_size=28*28):
        super(Discriminative, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size
        self.num_label  = num_label

        self.feat_net = nn.Sequential(
            GaussianNoise(0.3), WN_Linear(self.image_size, 1000), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear(1000, 500), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear( 500, 250), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear( 250, 250), nn.ReLU(),
            GaussianNoise(0.5), WN_Linear( 250, 250), nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            GaussianNoise(0.5), 
            WN_Linear(250, self.num_label, train_scale=True)
        )

    def forward(self, X, feat=False):
        if X.dim() == 4:
            X = X.view(X.size(0), -1)

        if feat:
            return self.feat_net(X)
        else:
            return self.out_net(self.feat_net(X))

class Encoder(nn.Module):
    def __init__(self, image_size=28*28, noise_size=100, output_params=True):
        super(Encoder, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size

        self.core_net = nn.Sequential(
            nn.Linear(self.image_size, 500, bias=False), nn.BatchNorm1d(500), nn.Softplus(), 
            nn.Linear(500, 500, bias=False), nn.BatchNorm1d(500), nn.Softplus()
        )

        if output_params:
            self.core_net.add_module(str(len(self.core_net._modules)), WN_Linear(500, self.noise_size*2, train_scale=True))
            self.core_net.add_module(str(len(self.core_net._modules)), Expression(lambda x: torch.chunk(x, 2, 1)))
        else:
            self.core_net.add_module(str(len(self.core_net._modules)), WN_Linear(500, self.noise_size, train_scale=True))

    def forward(self, input):
        if input.dim() == 4:
            input = input.view(-1, self.image_size)
        return self.core_net(input)
