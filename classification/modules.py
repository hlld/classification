import torch
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
        elif activation_type == 'gelu':
            self.act = nn.GELU()
        elif activation_type == 'none':
            self.act = nn.Identity()
        else:
            raise ValueError('Unkown type %s' % activation_type)

    def forward(self, x):
        return self.act(x)


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


class Msa(nn.Module):
    # https://github.com/rwightman/pytorch-image-models
    def __init__(self,
                 in_channels,
                 num_heads=8,
                 qkv_bias=True,
                 attn_drop=0,
                 proj_drop=0):
        super(Msa, self).__init__()
        head_dim = in_channels // num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(in_channels,
                             in_channels * 3,
                             bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape((b, n, 3, self.num_heads, c // self.num_heads))
        # [3, b, num_heads, n, c // num_heads]
        qkv = qkv.permute((2, 0, 3, 1, 4))
        # Avoid torch-script export error
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [b, num_heads, n, n]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # [b, n, num_heads, c // num_heads]
        x = (attn @ v).transpose(1, 2).reshape((b, n, c))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 activation='gelu',
                 dropout=0):
        super(Mlp, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = Activation(activation)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    # Stochastic depth per sample
    def __init__(self, drop_prob=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with different dim tensors
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape,
                                               dtype=x.dtype,
                                               device=x.device)
        x = x.div(keep_prob) * random_tensor.floor()
        return x


class ViTBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0,
                 attn_drop=0,
                 drop_path=0,
                 activation='gelu'):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.msa = Msa(in_channels,
                       num_heads=num_heads,
                       qkv_bias=qkv_bias,
                       attn_drop=attn_drop,
                       proj_drop=dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(in_channels)
        hidden_channels = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_channels=in_channels,
                       hidden_channels=hidden_channels,
                       activation=activation,
                       dropout=dropout)

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels=3,
                 image_size=224,
                 patch_size=16,
                 embed_dim=768,
                 flatten=True,
                 layer_norm=False):
        super(PatchEmbed, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.flatten = flatten
        self.conv = nn.Conv2d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=True)
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        if self.flatten:
            # [n, c, h, w] to [b, n, c] format
            x = x.flatten(2, -1).transpose(1, 2)
        x = self.norm(x)
        return x


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
