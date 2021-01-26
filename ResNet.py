import torch
from torch import nn


# 用于ResNet18和34的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_channel,out_channel,stride=1,down_sample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,
                               stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channel,out_channel,kernel_size=3,
                               stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def forward(self,x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)

        net = self.conv2(net)
        net = self.bn2(net)

        net += identity
        out = self.relu(net)

        return out


# 用于ResNet5和ResNet101的残差结构
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,in_channel,out_channel,stride=1,down_sample=None):
        super(Bottleneck,self).__init__()
        "********************************************************4"
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1,
                               stride=stride,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        "********************************************************"
        self.conv2 = nn.Conv2d(in_channel,out_channel,kernel_size=3,
                               stride=stride,bias=False,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        "********************************************************"
        self.conv3 = nn.Conv2d(in_channel,out_channel * self.expansion,kernel_size=1,
                               stride=stride,bias=False,padding=1)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

    def forward(self,x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)

        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)

        net = self.conv2(net)
        net = self.bn2(net)
        net = self.relu(net)

        net = self.conv3(net)
        net = self.bn3(net)

        net += identity
        out = self.relu(net)
        return out


class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=1000,include_top=True):
        super(ResNet,self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 特征矩阵深度
        # 512*512*3 -> 256*256*64
        self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=7,
                               stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # 256*256*64 -> 128*128*64
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # 128*128*64 -> 128*128*256
        self.layer1 = self._make_layer(block,64,blocks_num[0])
        # 128*128*256 -> 64*64*512
        self.layer2 = self._make_layer(block,128,blocks_num[1],stride=2)
        # 64*64*512 -> 32*32*1024
        self.layer3 = self._make_layer(block,256,blocks_num[2],stride=2)
        # 32*32*1024 -> 16*16*2048
        self.layer4 = self._make_layer(block,512,blocks_num[3],stride=2)

        """
        通过上述过程获得2048个有效特征层
        """

        if self.include_top:
            self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def _make_layer(self,block,channel,block_num,stride=1):
        down_sample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel,channel * block.expansion,kernel_size=1,stride=stride),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel,channel,down_sample=down_sample,stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel))

        return nn.Sequential(*layers)

    def forward(self,x):
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)
        net = self.max_pool(net)

        net = self.layer1(net)
        net = self.layer2(net)
        net = self.layer3(net)
        out = self.layer4(net)

        if self.include_top:
            net = self.avg_pool(out)
            net = torch.flatten(net,1)
            out = self.fc(net)

        return out


def ResNet34(num_classes=1000,include_top=True):
    return ResNet(BasicBlock,[3,4,6,3],
                  num_classes=num_classes,include_top=include_top)


def ResNet50(num_classes=1000,include_top=True):
    return ResNet(Bottleneck,[3,4,6,3],
                  num_classes=num_classes,include_top=include_top)


def ResNet101(num_classes=1000,include_top=True):
    return ResNet(Bottleneck,[3,4,23,3],
                  num_classes=num_classes,include_top=include_top)
