# -*- conding: utf-8 -*-
# @Time      : 2019/11/28
# @Author    : Thirteen
# @Site      :
# @File      : AlexNet.py

import torch
from torch import nn


class AlexNet(object):
    def __init__(self, input_channels, n_class):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=96,
                      kernel_size=11,
                      stride=2,
                      padding="same"), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=5,
                      stride=2,
                      padding='same'), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding='same'), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding='same'), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding='same'), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(4 * 4 * 256, 1024), nn.ReLU(),
                                 nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(),
                                 nn.Dropout(p=0.6))
        self.out = nn.Linear(1024, n_class)

    def network(self, inputs):
        net = self.conv1(inputs)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)
        net = net.view(net.size(0), -1)
        net = self.fc1(net)
        net = self.fc2(net)
        net = self.out(net)

        return net
