# Classification

This repository contains the PyTorch implementation of basic classification models. It supports multi-machine and multi-GPU training, mixed precision training, transfer learning, model export to TorchScript and ONNX. The reason I wrote this project was for review purpose, and models for various architectures will continue to be added. Hope it's useful to you guys.

## Development History

<details><summary> <b>Expand</b> </summary>

* `2021-07-28` - support basic training pipeline
* `2021-07-26` - support basic vision datasets
* `2021-07-25` - support basic convolutional models

</details>

## Training Benchmark

| Dataset | Model | FLOPs(B) | Params(M) | Top1(%) | Top5(%) |
| --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | - |

## Supported Datasets

<details><summary> <b>Expand</b> </summary>

- [x] MNIST
- [x] SVHN
- [x] CIFAR10
- [x] CIFAR100
- [x] ILSVRC2012
- [x] CUSTOM

</details>

## Supported Models

<details><summary> <b>Expand</b> </summary>

- [x] MLP
- [x] ResNet20
- [x] ResNet32
- [x] ResNet44
- [x] ResNet56
- [x] ResNet110
- [x] VGG16
- [x] VGG19
- [x] ResNet18
- [x] ResNet34
- [x] ResNet50
- [x] ResNet101
- [x] ResNet152

</details>

## Requirements
```
pip install -r requirements.txt
```

## Getting Start Training

### Training MLP on MNIST dataset
```
python train.py               \
    --data_type='mnist'       \
    --model_type='mlp'        \
    --hidden_channels=4096    \
    --dropout=0.5             \
    --epochs=40               \
    --batch_size=256
```

### Training ResNet20 on CIFAR10 dataset
```
python train.py               \
    --data_type='cifar10'     \
    --model_type='resnet20'   \
    --epochs=180              \
    --batch_size=128
```

### Training ResNet50 on ILSVRC2012 dataset
```
python train.py               \
    --data_root='path'        \
    --data_type='ilsvrc2012'  \
    --model_type='resnet50'   \
    --epochs=90               \
    --batch_size=256
```
- The ILSVRC2012 data set needs to be prepared before training. For details, see [examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)

### Fine-tuning ResNet50 on CUSTOM dataset
```
python train.py               \
    --data_root='path'        \
    --data_type='custom'      \
    --model_type='resnet50'   \
    --weights='pre-trained'   \
    --epochs=40               \
    --batch_size=128
```
- The custom dataset format should be consistent with ILSVRC2012. More specially, the reference format of a custom dataset is as follows:
```
dataset_name
├── test
│   ├── class_name_1
│   │   └── image.jpg
│   ├── class_name_2
│   │   └── image.jpg
│   └── class_name_3
│       └── image.jpg
├── train
│   ├── class_name_1
│   │   └── image.jpg
│   ├── class_name_2
│   │   └── image.jpg
│   └── class_name_3
│       └── image.jpg
└── val
    ├── class_name_1
    │   └── image.jpg
    ├── class_name_2
    │   └── image.jpg
    └── class_name_3
        └── image.jpg
```

### Training with multi-GPU or multi-machine
```
export CUDA_VISIBLE_DEVICES='0,1,2,3'
python -m torch.distributed.launch --nproc_per_node=4 train.py
```
Or we have a simpler way to start with:
```
python train.py --device='0,1,2,3'
```

## About The Author

A boring master student from CQUPT. Email `hlldmail@qq.com`

## Acknowledgements

* [pytorch/torchvision](https://github.com/pytorch/vision/tree/master/torchvision)
* [pytorch/examples](https://github.com/pytorch/examples)
* [akamaster/pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
