# -*- coding : utf-8 -*-
# @Author    : Thirteen
# @Time      : 2019/11/28
# @Site      :
# @File      : VGG.py
from torch import nn


class VGG(object):
    def __init__(self, input_channels, n_class):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=64,
                      kernel_size=3,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=26,
                      out_channels=256,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=26,
                      out_channels=512,
                      kernel_size=2,
                      padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(512*7*7,4096),nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(4096,4096),nn.Dropout())
        self.out = nn.Linear(4096,n_class)


    def VGG16(self, inputs):
        net = self.conv1(inputs)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net) 
        net = net.view(net.size(0),-1)
        net = self.fc1(net)
        net = self.fc2(net)
        net = self.out(net)

        return net