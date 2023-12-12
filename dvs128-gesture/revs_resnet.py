import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, surrogate
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from revtorch import ReversibleBlock, ReversibleSequence

surrogate_function = surrogate.ATan()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicResidualBlock(nn.Module):
    def __init__(self, planes, enable_amp=False):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes)
        )
        self.sn1 = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes)
        )
        self.sn2 = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        self.enable_amp = enable_amp

    def inner_forward(self, x):
        out = self.conv1(self.sn1(x))
        out = self.conv2(self.sn2(out))
        return out

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            return self.inner_forward(x)


def DownSampleConv(inplanes, planes, stride):
    return nn.Sequential(
        MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy'),
        layer.SeqToANNContainer(
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
            conv1x1(inplanes, planes, 1),
            nn.BatchNorm2d(planes),
        ))

class RevSResNet(nn.Module):

    def __init__(self, block, layers, planes, num_classes=10, norm_layer=None, T=4, enable_amp=False):
        super(RevSResNet, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = planes[0] * 2
        self.dilation = 1
        self.enable_amp = enable_amp

        self.conv1 = layer.SeqToANNContainer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False))
        self.bn1 = layer.SeqToANNContainer(norm_layer(self.inplanes))

        self.maxpool = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, planes[0], layers[0])

        self.ds1 = DownSampleConv(planes[0] * 2, planes[1] * 2, 2)
        self.layer2 = self._make_layer(block, planes[1], layers[1])

        self.ds2 = DownSampleConv(planes[1] * 2, planes[2] * 2, 2)
        self.layer3 = self._make_layer(block, planes[2], layers[2])

        self.sn1 = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(planes[2] * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks):

        layers = []
        for _ in range(0, blocks):
            f_func = block(planes, enable_amp=self.enable_amp)
            g_func = block(planes, enable_amp=self.enable_amp)
            layers.append(ReversibleBlock(f_func, g_func, 2))

        return ReversibleSequence(nn.ModuleList(layers))

    def _forward_impl(self, x, return_feature=False):
        x = x.permute(1, 0, 2, 3, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.ds1(x)
        x = self.layer2(x)

        x = self.ds2(x)
        x = self.layer3(x)
        x = self.sn1(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x.mean(dim=0)

    def forward(self, x, return_feature=False):
        return self._forward_impl(x, return_feature)


def revs_resnet(block, layers, planes, **kwargs):
    model = RevSResNet(block, layers, planes, **kwargs)
    return model


arch = [16, 32, 48]
# rt. r18
def revs_resnet24(**kwargs):
    return revs_resnet(BasicResidualBlock, [1, 2, 2], arch, **kwargs)
