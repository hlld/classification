import argparse
import math
import torch
import torch.nn as nn
from thop import profile
from copy import deepcopy
from classification.modules import Conv, Bottleneck, GlobalPool
from classification.tools import select_device, ModelEMA


class Model(object):
    def __init__(self,
                 in_channels,
                 num_classes,
                 model_type,
                 **kwargs):
        super(Model, self).__init__()
        if model_type == 'mlp':
            self._model = MLP(in_channels,
                              num_classes,
                              model_type,
                              **kwargs)
        elif 'vgg' in model_type:
            self._model = VGGNet(in_channels,
                                 num_classes,
                                 model_type,
                                 **kwargs)
        elif 'resnet' in model_type:
            self._model = ResNet(in_channels,
                                 num_classes,
                                 model_type)
        else:
            raise ValueError('Unknown type %s' % model_type)
        self._model_ddp = None
        self.model_ema = None

    def __call__(self, x):
        return self.model(x)

    @property
    def module(self):
        if self._model_ddp is not None:
            return self._model_ddp.module
        return self._model

    @property
    def model(self):
        if self._model_ddp is not None:
            return self._model_ddp
        return self._model

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self,
                        state_dict,
                        strict=True):
        self.module.load_state_dict(state_dict, strict=strict)

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def modules(self):
        return self.module.modules()

    def frozen_bn(self):
        # Frozen batchnorm layers before training
        for m in self.module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def apply_sync_bn(self, local_rank):
        # Should call before apply_ddp()
        if local_rank != -1:
            self._model = \
                torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)

    def apply_ddp(self, local_rank):
        if self._model_ddp is None and local_rank != -1:
            self._model_ddp = torch.nn.parallel.DistributedDataParallel(
                self._model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False)

    def apply_ema(self, local_rank):
        if self.model_ema is None and local_rank in [-1, 0]:
            self.model_ema = ModelEMA(self._model)

    def update_ema(self,
                   state_dict=None,
                   updates=0):
        if self.model_ema is not None:
            if state_dict is None:
                self.model_ema.update(self.module)
            else:
                self.model_ema.module.load_state_dict(state_dict)
                self.model_ema.updates = updates

    def profile(self,
                device,
                input_size=224,
                verbose=False):
        if self._model.model_type == 'mlp':
            input_size = 1
        inputs = torch.rand((1,
                             self._model.in_channels,
                             input_size,
                             input_size), device=device)
        # Backup model to avoid distributed training error
        flops, params = profile(deepcopy(self._model),
                                inputs=(inputs,),
                                verbose=verbose)
        # Flops in billion, params in million
        flops = flops / 1000 ** 3 * 2
        params /= 1000 ** 2
        print('Total FLOPs %.5f B, Params %.5f M\n' % (flops, params))
        return flops, params


class _BaseModel(nn.Module):
    def __init__(self):
        super(_BaseModel, self).__init__()
        self.stem = nn.Sequential()
        self.layers = nn.Sequential()
        self.pool = nn.Sequential(*[GlobalPool('avg'),
                                    nn.Flatten(1, -1)])
        self.logits = nn.Sequential()
        self.max_stride = -1
        self.model_type = ''
        self.data_type = ''
        self.in_channels = -1
        self.num_classes = -1
        self.classes = []

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x)
        x = self.logits(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MLP(_BaseModel):
    def __init__(self,
                 in_channels,
                 num_classes,
                 model_type='mlp',
                 hidden_channels=2048,
                 dropout=0):
        super(MLP, self).__init__()
        self.pool = nn.Flatten(1, -1)
        # Dropout must operate with inplace disabled
        logits = [nn.Linear(in_channels, hidden_channels),
                  nn.ReLU(inplace=True),
                  nn.Dropout(dropout),
                  nn.Linear(hidden_channels, num_classes)]
        self.logits = nn.Sequential(*logits)
        self.max_stride = 1
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model_type = model_type


class VGGNet(_BaseModel):
    def __init__(self,
                 in_channels,
                 num_classes,
                 model_type='vgg16',
                 hidden_channels=2048,
                 dropout=0):
        super(VGGNet, self).__init__()
        if model_type == 'vgg16':
            block_num = [2, 2, 3, 3, 3]
        elif model_type == 'vgg19':
            block_num = [2, 2, 4, 4, 4]
        else:
            raise ValueError('Unknown type %s' % model_type)
        num_outputs = [64, 128, 256, 512, 512]
        max_pool = [True, True, True, True, False]
        layers = []
        self.in_channels = in_channels
        for num, out, pool in zip(block_num,
                                  num_outputs,
                                  max_pool):
            layers.append(self._make_layer(in_channels,
                                           out_channels=out,
                                           num_blocks=num,
                                           max_pool=pool))
            in_channels = out
        self.layers = nn.Sequential(*layers)
        # Dropout must operate with inplace disabled
        logits = [nn.Linear(in_channels, hidden_channels),
                  nn.ReLU(inplace=True),
                  nn.Dropout(dropout),
                  nn.Linear(hidden_channels, num_classes)]
        self.logits = nn.Sequential(*logits)
        self.max_stride = 16
        self.num_classes = num_classes
        self.model_type = model_type
        self._initialize_weights()

    @staticmethod
    def _make_layer(in_channels,
                    out_channels,
                    num_blocks,
                    max_pool):
        layer = [Conv(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      activation='relu')]
        for _ in range(1, num_blocks):
            layer.append(Conv(out_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              activation='relu'))
        if max_pool:
            layer.append(nn.MaxPool2d(kernel_size=2,
                                      stride=2,
                                      padding=0))
        return nn.Sequential(*layer)


class ResNet(_BaseModel):
    def __init__(self,
                 in_channels,
                 num_classes,
                 model_type='resnet50'):
        super(ResNet, self).__init__()
        if model_type == 'resnet18':
            block_num = [2, 2, 2, 2]
        elif model_type == 'resnet34':
            block_num = [3, 4, 6, 3]
        elif model_type == 'resnet50':
            block_num = [3, 4, 6, 3]
        elif model_type == 'resnet101':
            block_num = [3, 4, 23, 3]
        elif model_type == 'resnet152':
            block_num = [3, 8, 36, 3]
        elif model_type == 'resnet20':
            block_num = [3, 3, 3]
        elif model_type == 'resnet32':
            block_num = [5, 5, 5]
        elif model_type == 'resnet44':
            block_num = [7, 7, 7]
        elif model_type == 'resnet56':
            block_num = [9, 9, 9]
        elif model_type == 'resnet110':
            block_num = [18, 18, 18]
        else:
            raise ValueError('Unknown type %s' % model_type)
        if model_type in ['resnet18', 'resnet34']:
            num_outputs = [64, 128, 256, 512]
            use_residual = True
        elif model_type in ['resnet20', 'resnet32', 'resnet44',
                            'resnet56', 'resnet110']:
            num_outputs = [16, 32, 64]
            use_residual = True
        else:
            num_outputs = [256, 512, 1024, 2048]
            use_residual = False
        self.in_channels = in_channels
        if model_type in ['resnet20', 'resnet32', 'resnet44',
                          'resnet56', 'resnet110']:
            self.max_stride = 8
            down_sample = [False, True, True]
            shortcut_conv = False
            self.stem = Conv(in_channels,
                             out_channels=16,
                             kernel_size=3,
                             stride=1,
                             activation='relu')
            in_channels = 16
        else:
            self.max_stride = 32
            down_sample = [False, True, True, True]
            shortcut_conv = True
            self.stem = nn.Sequential(*[
                Conv(in_channels,
                     out_channels=64,
                     kernel_size=7,
                     stride=2,
                     activation='relu'),
                nn.MaxPool2d(kernel_size=3,
                             stride=2,
                             padding=1)
            ])
            in_channels = 64
        layers = []
        for num, out, down in zip(block_num,
                                  num_outputs,
                                  down_sample):
            layers.append(self._make_layer(in_channels,
                                           out_channels=out,
                                           down_sample=down,
                                           num_blocks=num,
                                           use_residual=use_residual,
                                           shortcut_conv=shortcut_conv))
            in_channels = out
        self.layers = nn.Sequential(*layers)
        self.logits = nn.Linear(in_channels, num_classes)
        self.num_classes = num_classes
        self.model_type = model_type
        self._initialize_weights()

    @staticmethod
    def _make_layer(in_channels,
                    out_channels,
                    down_sample,
                    num_blocks,
                    use_residual,
                    shortcut_conv):
        layer = [Bottleneck(in_channels,
                            out_channels,
                            down_sample=down_sample,
                            use_residual=use_residual,
                            shortcut_conv=shortcut_conv,
                            activation='relu')]
        for _ in range(1, num_blocks):
            layer.append(Bottleneck(out_channels,
                                    out_channels,
                                    down_sample=False,
                                    use_residual=use_residual,
                                    shortcut_conv=True,
                                    activation='relu'))
        return nn.Sequential(*layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet18',
                        help='model type')
    parser.add_argument('--device', type=int, default=0,
                        help='cuda device')
    opt = parser.parse_args()

    device = select_device(opt.device)
    model = Model(in_channels=3,
                  num_classes=1000,
                  model_type=opt.model_type)
    model = model.to(device).train()
    model.profile(device, input_size=224)
    for name, val in model.named_parameters():
        print(name, val.shape)
