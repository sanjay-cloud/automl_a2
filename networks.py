###################################################################################
# DO NOT CHANGE THIS FILE
###################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(self, indim=3, out_channels=64):
        super().__init__()
        self.out_channels = out_channels
        self.cl = nn.Conv2d(in_channels=indim, out_channels=self.out_channels,
                            kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels, momentum=1)
        self.mp = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        
    def forward(self, x, weights=None):
        if weights is None:
            x = self.cl(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.mp(x)
            return x
        
        # Apply conv2d
        x = F.conv2d(x, weights[0], weights[1], padding=1) 

        # Manual batch normalization followed by ReLU
        running_mean =  torch.zeros(self.out_channels).to(x.device)
        running_var = torch.ones(self.out_channels).to(x.device)
        x = F.batch_norm(x, running_mean, running_var, 
                         weights[2], weights[3], momentum=1, training=True)
        x = F.max_pool2d(F.relu(x), kernel_size=2)
        return x


class Conv4(nn.Module):
    
    def __init__(self, num_ways, img_size, rgb=False):
        super().__init__()
        self.eval()
        self.num_ways = num_ways

        rnd_input = torch.rand((2,1,img_size,img_size))

        d = OrderedDict([])
        for i in range(4):
            if i == 0:
                if rgb:
                    indim = 3
                else:
                    indim = 1
            else:
                indim = 64
            d.update({'conv_block%i'%i: ConvBlock(indim=indim, out_channels=64)})
        
        d.update({'flatten': nn.Flatten()})
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})

        self.in_features = self.get_infeatures(rnd_input).size()[1]
        print("In-features:", self.in_features)
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.num_ways)})
        self.train()

    def get_infeatures(self,x):
        for i in range(4):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        return x

    def forward(self, x, weights=None):
        if weights is None:
            for i in range(4):
                x = self.model.features[i](x)
            x = self.model.features.flatten(x)
            x = self.model.out(x)
            return x
        
        for i in range(4):
            x = self.model.features[i](x, weights=weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        return x
