from torch import nn
import VGG16
import config


class Siamese(nn.Module):
    def __init__(self, in_channel, out_channel):
        # forward networks from VGG16
        super().__init__()
        self.net = VGG16.VGG16(config.classes)

    def forward_once(self, x):
        out = self.net.forward(x)
        return out

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        return out1, out2
