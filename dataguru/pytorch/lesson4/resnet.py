import torch
from torch import nn
from torch.nn import functional as F


def conv_unit(in_channels, out_channels, kernel_size, stride, use_bn=False, act_type=None):
    pad = kernel_size // 2
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if act_type is 'prelu':
        layers.append(nn.PReLU())
    elif act_type is 'relu':
        layers.append(nn.ReLU())
    return layers


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, act_type='relu'):
        super(ResidualBlock, self).__init__()
        layers = [nn.BatchNorm2d(in_channels), ]
        layers += conv_unit(in_channels, out_channels, kernel_size=3, stride=1, act_type=act_type)
        layers += conv_unit(out_channels, out_channels, kernel_size=3, stride=stride, act_type=None)
        self.fx = nn.Sequential(*layers)
        self.stride = stride
        if self.stride == 2:
            self.gx = nn.Sequential(*conv_unit(in_channels, out_channels, kernel_size=3, stride=2, act_type=None))

    def forward(self, x):
        if self.stride == 1:
            return self.fx(x) + x
        else:
            return self.fx(x) + self.gx(x)


class ResNet(nn.Module):
    """
    ResNet 50 [3, 4, 14, 3]
    filter_list = [64, 64, 128, 256, 512]
    """

    def __init__(self, units=(3, 4, 14, 3), filter_list=(64, 64, 128, 256, 512), act_type='prelu'):
        super(ResNet, self).__init__()
        # [B, 64, 112, 112]
        self.res0 = nn.Conv2d(3, filter_list[0], (3, 3), 1, 1, bias=False)
        residual = []
        # [B, 64, 56, 56]
        self.res1 = nn.Sequential(ResidualBlock(filter_list[0], filter_list[1], 2, act_type=act_type),
                                  *[ResidualBlock(filter_list[1], filter_list[1], 1, act_type) for _ in
                                    range(units[0] - 1)])
        # [B, 128, 28, 28]
        layers = [ResidualBlock(filter_list[1], filter_list[2], 2, act_type=act_type)]
        layers += [ResidualBlock(filter_list[2], filter_list[2], 1, act_type) for _ in range(units[1] - 1)]
        self.res2 = nn.Sequential(*layers)
        # [B, 256, 14, 14]
        self.res3 = nn.Sequential(ResidualBlock(filter_list[2], filter_list[3], 2, act_type=act_type),
                                  *[ResidualBlock(filter_list[3], filter_list[3], 1, act_type) for _ in
                                    range(units[2] - 1)])
        # [B, 512, 7, 7]
        self.res4 = nn.Sequential(ResidualBlock(filter_list[3], filter_list[4], 2, act_type=act_type),
                                  *[ResidualBlock(filter_list[4], filter_list[4], 1, act_type) for _ in
                                    range(units[3] - 1)])
        # end block
        self.end_block1 = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout2d(0.4))
        self.end_block2 = nn.Sequential(nn.Linear(7 * 7 * 512, 512, bias=False), nn.BatchNorm1d(512))

    def forward(self, x):
        x = (x - 127.5) / 256.0
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.end_block1(x)
        x = x.view(-1, 7 * 7 * 512)
        x = self.end_block2(x)
        return x


if __name__ == '__main__':
    net=ResNet()
    print(net)