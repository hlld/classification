# Classification

Implementation of common classification model

## Development History

<details><summary> <b>Expand</b> </summary>

* `2021-07-28` - support basic training pipeline
* `2021-07-26` - support basic vision datasets
* `2021-07-25` - support basic convolutional models

</details>

## Training records

| Dataset | Model | FLOPs(B) | Params(M) | Top1(%) | Top5(%) |
| --- | --- | --- | --- | --- | --- |
| MNIST | MLP | - | - | - | - |

## Requirements
```
pip install -r requirements.txt
```

## How to Use

### Training MLP on MNIST dataset
```
python train.py               \
    --data_type='mnist'       \
    --model_type='mlp'        \
    --hidden_channels=2048    \
    --dropout=0.5             \
    --epochs=20               \
    --batch_size=256
```

### Training ResNet20 on SVHN dataset
```
python train.py               \
    --data_type='svhn'        \
    --model_type='resnet20'   \
    --epochs=20
```

### Training VGG16 on CIFAR10 dataset
```
python train.py               \
    --data_type='cifar10'     \
    --model_type='vgg16'      \
    --hidden_channels=128     \
    --dropout=0.5             \
    --epochs=200
```

### Training ResNet18 on CIFAR100 dataset
```
python train.py               \
    --data_type='cifar100'    \
    --model_type='resnet18'   \
    --epochs=200
```

### Training ResNet50 on ILSVRC2012 dataset
```
python train.py               \
    --data_type='ilsvrc2012'  \
    --model_type='resnet50'   \
    --epochs=60               \
    --batch_size=256
```

## About The Author

A boring master student from CQUPT. Email: hlldmail@qq.com
