import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from classification.models import Model
from classification.losses import CrossEntropyLoss
from classification.datasets import DataLoader
from classification.evaluate import evaluate
from classification.tools import load_yaml, save_yaml, find_directory, \
    ModelEMA, select_intersect, strip_optimizer, select_device, \
    decay_lambda, torch_benchmark


def train_network(local_rank, opt):
    opt.local_rank = local_rank
    init_method = 'env://'
    if opt.use_single_node_ddp:
        init_method = 'tcp://localhost:%d' % opt.local_port
        opt.global_rank = local_rank
    device = select_device(opt.device)
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda:%d' % local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=init_method,
                                             world_size=opt.world_size,
                                             rank=opt.global_rank)
        # Divide the batch size based on the total number of GPUs
        opt.workers = opt.workers // opt.world_size
        opt.batch_size = opt.total_batch_size // opt.world_size
    torch_benchmark(opt.benchmark, local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset options
    parser.add_argument('--data_root', type=str, default='../datasets',
                        help='data root')
    parser.add_argument('--data_type', type=str, default='cifar10',
                        help='data type')
    parser.add_argument('--input_size', type=int, default=None,
                        help='input size')
    parser.add_argument('--in_channels', type=int, default=None,
                        help='input channels')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='dataset classes')
    parser.add_argument('--image_mean', type=list, default=None,
                        help='image mean')
    parser.add_argument('--image_std', type=list, default=None,
                        help='image std')
    parser.add_argument('--data_augment', type=bool, default=True,
                        help='data augmentation')
    parser.add_argument('--random_flip', type=float, default=0.5,
                        help='random flip')
    parser.add_argument('--random_crop', type=float, default=0.5,
                        help='random crop')
    # Training options
    parser.add_argument('--benchmark', type=bool, default=False,
                        help='benchmark mode')
    parser.add_argument('--weights', type=str, default='',
                        help='weights path')
    parser.add_argument('--model_type', type=str, default='resnet20',
                        help='model type')
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='total batch size')
    parser.add_argument('--save_path', type=str, default='./results',
                        help='save path')
    parser.add_argument('--cosine_lr', type=bool, default=False,
                        help='cosine learning rate')
    parser.add_argument('--lr_steps', type=list, default=None,
                        help='learning rate decay steps')
    parser.add_argument('--initial_lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--final_lr', type=float, default=0.001,
                        help='final learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='warmup steps')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    # Platform options
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='DDP local rank')
    parser.add_argument('--local_port', type=int, default=1234,
                        help='DDP local port')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda device')
    parser.add_argument('--sync_bn', type=bool, default=False,
                        help='sync batchnorm')
    parser.add_argument('--workers', type=int, default=4,
                        help='dataloader workers')
    opt = parser.parse_args()

    use_single_node_ddp, global_rank = False, -1
    if opt.local_rank == -1:
        world_size = len(opt.device.strip().split(','))
        if world_size > 1:
            use_single_node_ddp = True
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        global_rank = int(os.environ['RANK'])
    # Only the first process of every node save results
    # Assume that all nodes have the same files contained
    if opt.resume:
        save_path = Path(opt.weights).parent
        if opt.local_rank in [-1, 0]:
            print('Resuming training from %s ...' % save_path)
    else:
        save_path = find_directory(opt.save_path, 'exp')
        if opt.local_rank in [-1, 0]:
            Path(save_path).mkdir(parents=True, exist_ok=True)
    opt_path = os.path.join(save_path, 'options.yaml')
    if opt.resume:
        backup = (opt.device, opt.local_rank, opt.local_port, opt.weights)
        opt_params = load_yaml(opt_path)
        opt = argparse.Namespace(**opt_params)
        opt.device, opt.local_rank, opt.local_port, opt.weights = backup
        opt.resume = True
    # Save runtime options parameters
    if opt.local_rank in [-1, 0] and not opt.resume:
        save_yaml(opt_path, vars(opt))
    opt.save_path = save_path
    opt.use_single_node_ddp = use_single_node_ddp
    opt.world_size = world_size
    opt.global_rank = global_rank
    opt.total_batch_size = opt.batch_size
    if opt.local_rank in [-1, 0]:
        print('Options ' + str(vars(opt)))
    if use_single_node_ddp:
        torch.multiprocessing.spawn(train_network,
                                    nprocs=opt.world_size,
                                    args=(opt,))
    else:
        train_network(opt.local_rank, opt)
