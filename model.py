# -*- coding: utf-8 -*-
# @Author   : zqian9
import torch
import torch.nn as nn

from frontend import LogSpectrogram


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.pre_act = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
        )

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.pre_act(x)

        residual = self.residual_function(out)
        out = residual + self.shortcut(out)

        return out


class BottleNeck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, drop_path_rate=0.0):
        super().__init__()
        self.pre_act = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
        )

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.pre_act(x)

        residual = self.residual_function(out)
        out = residual + self.shortcut(out)

        return out


class ResNetV2(nn.Module):

    def __init__(self, block, num_block, num_classes=2):
        super().__init__()

        self.in_channels = 16
        self.frontend = LogSpectrogram(512, 512, 256)

        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)

        self.layer1 = self._make_layer(block, 16, num_block[0], 1)
        self.layer2 = self._make_layer(block, 32, num_block[1], 2)
        self.layer3 = self._make_layer(block, 64, num_block[2], 2)
        self.layer4 = self._make_layer(block, 128, num_block[3], 2)
        self.post_act = nn.Sequential(
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.frontend(x)
        out = out.unsqueeze(1)
        out = self.conv1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.post_act(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def resnetv2_10():
    return ResNetV2(BasicBlock, [1, 1, 1, 1])


def resnetv2_18():
    return ResNetV2(BasicBlock, [2, 2, 2, 2])


def resnetv2_34():
    return ResNetV2(BasicBlock, [3, 4, 6, 3])


def resnetv2_50():
    return ResNetV2(BottleNeck, [3, 4, 6, 3])


if __name__ == '__main__':
    m = resnetv2_10()
    print(m)
