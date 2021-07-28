import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 activation='relu'):
        super(Conv, self).__init__()
        # Always pad to the same shape
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Activation(nn.Module):
    def __init__(self, activation_type='relu'):
        super(Activation, self).__init__()
        if activation_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation_type == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        elif activation_type == 'none':
            self.act = nn.Identity()
        else:
            raise ValueError('Unkown type %s' % activation_type)

    def forward(self, x):
        return self.act(x)


class Padding(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(Padding, self).__init__()
        num_channels = out_channels - in_channels
        self.padding = [num_channels // 2,
                        num_channels // 2 + num_channels % 2]
        self.stride = stride

    def forward(self, x):
        return F.pad(x[:, :, ::self.stride, ::self.stride],
                     [0, 0, 0, 0, *self.padding],
                     mode='constant',
                     value=0)


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 down_sample=False,
                 use_residual=False,
                 shortcut_conv=True,
                 activation='relu'):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        if not use_residual:
            shortcut_conv = True
        self.shortcut = None
        if in_channels != out_channels:
            if shortcut_conv:
                self.shortcut = Conv(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=stride,
                                     activation='none')
            else:
                self.shortcut = Padding(in_channels,
                                        out_channels,
                                        stride=stride)
        if use_residual:
            layers = [Conv(in_channels,
                           out_channels,
                           kernel_size=3,
                           stride=stride,
                           activation=activation),
                      Conv(out_channels,
                           out_channels,
                           kernel_size=3,
                           stride=1,
                           activation='none')]
        else:
            mid_channels = out_channels // 4
            layers = [Conv(in_channels,
                           mid_channels,
                           kernel_size=1,
                           stride=1,
                           activation=activation),
                      Conv(mid_channels,
                           mid_channels,
                           kernel_size=3,
                           stride=stride,
                           activation=activation),
                      Conv(mid_channels,
                           out_channels,
                           kernel_size=1,
                           stride=1,
                           activation='none')]
        self.m = nn.Sequential(*layers)
        self.act = Activation(activation)

    def forward(self, x):
        shortcut = x
        x = self.m(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        x = x + shortcut
        return self.act(x)


class GlobalPool(nn.Module):
    def __init__(self, pooling_type='avg'):
        super(GlobalPool, self).__init__()
        if pooling_type == 'max':
            self.use_max_pool = True
        elif pooling_type == 'avg':
            self.use_max_pool = False
        else:
            raise ValueError('Unknown type %s' % pooling_type)

    def forward(self, x):
        if len(x.shape) != 4 or (x.shape[2] == 1 and x.shape[3] == 1):
            return x
        if self.use_max_pool:
            return F.max_pool2d(x,
                                x.shape[2:4],
                                stride=1,
                                padding=0)
        return F.avg_pool2d(x,
                            x.shape[2:4],
                            stride=1,
                            padding=0)
