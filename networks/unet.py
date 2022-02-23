# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:58:19 2019

@author: ASUS
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu,inplace=True)
class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)  
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.activation(x)
        output = self.bn2(x)
    
        return output
        
class conv_block_dilated(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_dilated, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, dilation=3, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, dilation=3, padding=3, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)  
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.activation(x)
        output = self.bn2(x)
    
        return output


class UNet(nn.Module):


    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv5_0 = conv_block_nested(filters[4] + filters[3], filters[3], filters[3])
        self.conv6_0 = conv_block_nested(filters[3] + filters[2], filters[2], filters[2])
        self.conv7_0 = conv_block_nested(filters[2] + filters[1], filters[1], filters[1])
        self.conv8_0 = conv_block_nested(filters[1] + filters[0], filters[0], filters[0])
        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x_ = self.Up(x)
        x0 = self.conv0_0(x_)  # h,w,n1
        x1 = self.conv1_0(self.pool(x0))  # h/2,w/2,n2
        x2 = self.conv2_0(self.pool(x1))  # h/4,w/4,n3
        x3 = self.conv3_0(self.pool(x2))  # h/8,w/8,n4
        x4 = self.conv4_0(self.pool(x3))  # h/16,w/16,n5
        x5 = self.conv5_0(torch.cat([self.Up(x4), x3], 1))  # h/8,w/8,n4
        x6 = self.conv6_0(torch.cat([self.Up(x5), x2], 1))  # h/4,w/4,n3
        x7 = self.conv7_0(torch.cat([self.Up(x6), x1], 1))  # h/2,w/2,n2
        x8 = self.conv8_0(torch.cat([self.Up(x7), x0], 1))  # h,w,n1

        output = self.final(x8)

        perc_map0 = torch.max(x5, dim=1)[0]
        perc_map1 = torch.max(x6, dim=1)[0]
        perc_map2 = torch.max(x7, dim=1)[0]

        return F.sigmoid(output), (perc_map0, perc_map1, perc_map2)


class UNetNeighbor(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(UNetNeighbor, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(in_ch, filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv5_0 = conv_block_nested(filters[4] + filters[3], filters[3], filters[3])
        self.conv6_0 = conv_block_nested(filters[3] + filters[2], filters[2], filters[2])
        self.conv7_0 = conv_block_nested(filters[2] + filters[1], filters[1], filters[1])
        self.conv8_0 = conv_block_nested(filters[1] + filters[1], filters[0], filters[0])
        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        # x0 = self.conv0_0(x) # h,w,n1
        x1 = self.conv1_0(x)  # h,w,n2
        x2 = self.conv2_0(self.pool(x1))  # h/2,w/2,n3
        x3 = self.conv3_0(self.pool(x2))  # h/4,w/4,n4
        x4 = self.conv4_0(self.pool(x3))  # h/8,w/8,n5
        x5 = self.conv5_0(torch.cat([self.Up(x4), x3], 1))  # h/4,w/4,n4
        x6 = self.conv6_0(torch.cat([self.Up(x5), x2], 1))  # h/2,w/2,n3
        x7 = self.conv7_0(torch.cat([self.Up(x6), x1], 1))  # h,w,n2
        x8 = self.conv8_0(torch.cat([self.Up(x7), self.Up(x1)], 1))  # h*2,w*2,n1

        output = self.final(x8)
        # perc_map = torch.max(x2,dim=1)[0]
        # perc_map1 = torch.min(x2,dim=1)[0]

        perc_map0 = torch.max(x5, dim=1)[0]
        perc_map1 = torch.max(x6, dim=1)[0]
        perc_map2 = torch.max(x7, dim=1)[0]

        # print(perc_map.shape)
        return F.sigmoid(output), (perc_map0, perc_map1, perc_map2)