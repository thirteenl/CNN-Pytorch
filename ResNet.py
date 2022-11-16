from torch import nn
import config


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1), down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=(3, 3),
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu1(net)

        net = self.conv2(net)
        net = self.bn2(net)
        net = self.relu2(net)

        net = self.conv3(net)
        net = self.bn3(net)

        if self.down_sample is not None:
            identity = self.down_sample(x)
        net += identity

        return self.relu(net)


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes, classify=False):
        super(ResNet, self).__init__()
        self.classify = classify

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7),
                               stride=(2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)

        self.layer1 = self._maker_layer(block, 64, blocks_num[0])
        self.layer2 = self._maker_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._maker_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._maker_layer(block, 512, blocks_num[3], stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def _make_layer(self, block, channels, block_num, stride=(1, 1)):
        down_sample = None
        if stride != 1 or self.in_channels != channels * block.expansion:  # 只需要判断一个大类的block是否需要降采样
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = [block(self.in_channels, channels, down_sample=down_sample, stride=stride)]

        self.in_channels = channels * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        net = self.conv1(x)
        net = self.bn1(net)
        net = self.relu(net)
        net = self.maxpool(net)

        net = self.layer1(net)
        net = self.layer2(net)
        net = self.layer3(net)
        net = self.layer4(net)

        if self.claasify:
            net = net.contiguous().view(x.size()[0], -1)
            net = self.classifier(net)

        return net


def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], config.classes)  # 1000=len(config.classes)
