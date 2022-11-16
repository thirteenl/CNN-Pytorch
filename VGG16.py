from torch import nn


class Block(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)


class VGG16(nn.Module):

    def __init__(self, classes, classify=False) -> None:
        super(VGG16, self).__init__()
        self.classify = classify

        self.in_channel = 3
        self.layer1 = self._make_layer(Block, 64, 2)
        self.layer2 = self._make_layer(Block, 128, 2)
        self.layer3 = self._make_layer(Block, 256, 3)
        self.layer4 = self._make_layer(Block, 512, 3)
        self.layer5 = self._make_layer(Block, 512, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, classes)
        )

    def _make_layer(self, block, out_channel, block_num):
        conv = []
        for _ in range(block_num):
            conv.append(block(self.in_channel, out_channel))
            self.in_channel = out_channel
        return nn.Sequential(*conv)

    def forward(self, x):
        net = self.layer1(x)
        net = self.maxpool(net)
        net = self.layer2(net)
        net = self.maxpool(net)
        net = self.layer3(net)
        net = self.maxpool(net)
        net = self.layer4(net)  # 图像太小了，直接变成1*1了
        net = self.maxpool(net)
        net = self.layer5(net)
        net = self.maxpool(net)
        if self.classify:
            net = net.contiguous().view(x.size()[0], -1)

            net = self.classifier(net)
        return net
