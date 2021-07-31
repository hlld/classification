import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 activation='relu'):
        super(Conv, self).__init__()
        # Always pad to the same shape
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HardSigmoid(nn.Module):
    @staticmethod
    def forward(x):
        return 0.16667 * F.relu6(x + 3.0, inplace=True)


class HardSwish(nn.Module):
    @staticmethod
    def forward(x):
        return 1.0 / 6.0 * x * F.relu6(x + 3.0, inplace=True)


class Activation(nn.Module):
    def __init__(self, activation_type='relu'):
        super(Activation, self).__init__()
        if activation_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation_type == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        elif activation_type == 'hard-swish':
            self.act = HardSwish()
        elif activation_type == 'hard-sigmoid':
            self.act = HardSigmoid()
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


class SeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 activation='relu'):
        super(SeparableConv, self).__init__()
        self.m = nn.Sequential(*[
            Conv(in_channels,
                 in_channels,
                 kernel_size=kernel_size,
                 stride=stride,
                 groups=in_channels,
                 activation=activation),
            Conv(in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 activation=activation),
        ])

    def forward(self, x):
        return self.m(x)


class SqueezeExcite(nn.Module):
    def __init__(self,
                 in_channels,
                 squeeze_factor=4,
                 divisible_by=8):
        super(SqueezeExcite, self).__init__()
        squeezed_channels = self.make_divisible(in_channels / squeeze_factor,
                                                divisor=divisible_by)
        self.m = nn.Sequential(*[
            GlobalPool('avg'),
            nn.Conv2d(in_channels,
                      squeezed_channels,
                      kernel_size=1,
                      stride=1),
            Activation('relu'),
            nn.Conv2d(squeezed_channels,
                      in_channels,
                      kernel_size=1,
                      stride=1),
            Activation('hard-sigmoid')
        ])

    def make_divisible(self,
                       value,
                       divisor,
                       min_value=None):
        if min_value is None:
            min_value = divisor
        new_value = max(min_value,
                        int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def forward(self, x):
        return self.m(x) * x


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=6.0,
                 kernel_size=3,
                 stride=1,
                 activation='relu6',
                 use_squeeze_excite=False):
        super(InvertedResidual, self).__init__()
        expanded_channels = int(float(in_channels) * expand_ratio)
        if expanded_channels != in_channels:
            in_module = Conv(in_channels,
                             expanded_channels,
                             kernel_size=1,
                             stride=1,
                             activation=activation)
        else:
            in_module = nn.Identity()
        if use_squeeze_excite:
            se_module = SqueezeExcite(expanded_channels)
        else:
            se_module = nn.Identity()
        self.m = nn.Sequential(*[
            in_module,
            Conv(expanded_channels,
                 expanded_channels,
                 kernel_size=kernel_size,
                 stride=stride,
                 groups=expanded_channels,
                 activation=activation),
            se_module,
            Conv(expanded_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 activation='none')
        ])
        self.shortcut = stride == 1 and in_channels == out_channels

    def forward(self, x):
        y = self.m(x)
        if self.shortcut:
            y = y + x
        return y


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
        assert len(x.shape) == 4
        # Static kernel size to avoid onnx export error
        kernel_h, kernel_w = int(x.shape[2]), int(x.shape[3])
        if self.use_max_pool:
            return F.max_pool2d(x,
                                kernel_size=(kernel_h, kernel_w),
                                stride=1,
                                padding=0)
        return F.avg_pool2d(x,
                            kernel_size=(kernel_h, kernel_w),
                            stride=1,
                            padding=0)
