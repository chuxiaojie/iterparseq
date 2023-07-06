import math
from typing import Optional, Callable

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class BasicBlock(resnet.BasicBlock):

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.conv1 = resnet.conv1x1(inplanes, planes)
        self.conv2 = resnet.conv3x3(planes, planes, stride)


class ResNet(nn.Module):

    def __init__(self, block, layers, output_channels=512):
        super().__init__()
        channels = [output_channels//(2**i) for i in reversed(range(5))]
        self.channels = channels
        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=1)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=1)
        self.layer5 = self._make_layer(block, channels[4], layers[4], stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, extra_feats=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if extra_feats is not None:
            if extra_feats[0].shape[1]>0:
                x = x+F.interpolate(extra_feats[0], x.shape[2:], mode='nearest')
        x = self.layer1(x)
        if extra_feats is not None:
            if extra_feats[1].shape[1]>0:
                x = x+F.interpolate(extra_feats[1], x.shape[2:], mode='nearest')
        x = self.layer2(x)
        if extra_feats is not None:
            if extra_feats[2].shape[1]>0:
                x = x+F.interpolate(extra_feats[2], x.shape[2:], mode='nearest')
        x = self.layer3(x)
        if extra_feats is not None:
            if extra_feats[3].shape[1]>0:
                x = x+F.interpolate(extra_feats[3], x.shape[2:], mode='nearest')
        x = self.layer4(x)
        if extra_feats is not None:
            if extra_feats[4].shape[1]>0:
                x = x+F.interpolate(extra_feats[4], x.shape[2:], mode='nearest')
        x = self.layer5(x)
        if extra_feats is not None:
            if extra_feats[5].shape[1]>0:
                x = x+F.interpolate(extra_feats[5], x.shape[2:], mode='nearest')
        return x


def resnet45(layers=[3, 4, 6, 6, 3], output_channels=512):
    print(layers, output_channels)
    return ResNet(BasicBlock, layers=layers, output_channels=output_channels)