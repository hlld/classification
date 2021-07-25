import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def create_dataloader(args,
                      use_transet=True,
                      return_info=True):
    if args.data_type == 'imagenet':
        imagedir = 'train' if use_transet else 'val'
        datadir = os.path.join(args.data_path, imagedir)
        num_classes = 1000
        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        base_ops = [transforms.ToTensor(),
                    transforms.Normalize(mean, std)]
        if use_transet:
            aug_ops = [transforms.RandomResizedCrop(image_size),
                       transforms.RandomHorizontalFlip()]
        else:
            aug_ops = [transforms.Resize(256),
                       transforms.CenterCrop(image_size)]
        base_ops = aug_ops + base_ops
        transform = transforms.Compose(base_ops)
        dataset = datasets.ImageFolder(root=datadir,
                                       transform=transform)
    elif args.data_type in ['mnist', 'cifar10', 'cifar100', 'svhn']:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        image_size = 32
        if args.data_type == 'mnist':
            builder = torchvision.datasets.MNIST
            num_classes = 10
        elif args.data_type == 'cifar10':
            builder = torchvision.datasets.CIFAR10
            num_classes = 10
        elif args.data_type == 'cifar100':
            builder = torchvision.datasets.CIFAR100
            num_classes = 100
        else:
            builder = torchvision.datasets.SVHN
            num_classes = 10
        if args.data_type == 'mnist':
            base_ops = [transforms.ToTensor(),
                        transforms.Normalize(mean[0], std[0])]
        else:
            base_ops = [transforms.ToTensor(),
                        transforms.Normalize(mean, std)]
        if use_transet:
            padding = 6 if args.data_type == 'mnist' else 4
            aug_ops = [transforms.RandomCrop(image_size, padding=padding),
                       transforms.RandomHorizontalFlip()]
            base_ops = aug_ops + base_ops
        elif args.data_type == 'mnist':
            aug_ops = [transforms.Pad(padding=2)]
            base_ops = aug_ops + base_ops
        transform = transforms.Compose(base_ops)
        if args.data_type == 'svhn':
            split_set = 'train' if use_transet else 'test'
            dataset = builder(root=args.data_path,
                              split=split_set,
                              download=True,
                              transform=transform)
        else:
            dataset = builder(root=args.data_path,
                              train=use_transet,
                              download=True,
                              transform=transform)
    else:
        raise ValueError('Unknown data type %s' % args.data_type)
    batch_size = args.test_batch_size
    if use_transet:
        batch_size = args.train_batch_size
    sampler = None
    if use_transet and args.use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    # For Windows use num_workers=0
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_transet and not args.use_ddp,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler)
    if return_info:
        return dataloader, [image_size, num_classes]
    return dataloader
