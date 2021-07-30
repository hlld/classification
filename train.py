import argparse
import os
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
from classification.losses import Criterion
from classification.datasets import DataLoader
from classification.evaluate import evaluate
from classification.tools import load_yaml, save_yaml, find_directory, \
    select_intersect, strip_optimizer, select_device, decay_lambda, \
    torch_benchmark, check_input_size


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

    # Infer default arguments
    if opt.data_type in ['mnist', 'svhn', 'cifar10',
                         'cifar100', 'ilsvrc2012']:
        input_size, in_channels, num_classes, image_mean, image_std = \
            DataLoader.default_params(opt.data_type)
        if opt.input_size <= 0:
            opt.input_size = input_size
        if opt.in_channels <= 0:
            opt.in_channels = in_channels
        if opt.num_classes <= 0:
            opt.num_classes = num_classes
        if len(opt.image_mean) == 0:
            opt.image_mean = image_mean
        if len(opt.image_std) == 0:
            opt.image_std = image_std
    elif opt.data_type == 'custom':
        if opt.input_size <= 0 or opt.in_channels <= 0 or \
                opt.num_classes <= 0 or len(opt.image_mean) == 0 or \
                len(opt.image_std) == 0:
            raise ValueError('Customized dataset with default arguments')
    else:
        raise ValueError('Unknown type %s' % opt.data_type)
    hyp_params = {'hsv': opt.random_hsv,
                  'flip': opt.random_flip,
                  'crop': opt.random_crop,
                  'mean': opt.image_mean,
                  'std': opt.image_std}

    # Load model into memory
    if opt.model_type == 'mlp':
        opt.in_channels *= opt.input_size ** 2
    model = Model(opt.in_channels,
                  num_classes=opt.num_classes,
                  model_type=opt.model_type,
                  hidden_channels=opt.hidden_channels,
                  dropout=opt.dropout)
    model.to(device).train()
    if opt.weights:
        ckpt = torch.load(opt.weights, map_location=device)
        state_dict = ckpt['model'].float().state_dict()
        state_dict = select_intersect(state_dict,
                                      model.state_dict(),
                                      exclude=[])
        model.load_state_dict(state_dict, strict=False)
        if local_rank in [-1, 0]:
            print('Transferred %g/%g items from %s' % (
                len(state_dict),
                len(model.state_dict()),
                opt.weights))
    # Check input size before profile
    opt.input_size = check_input_size(opt.input_size,
                                      model.module.max_stride)
    if local_rank in [-1, 0]:
        model.profile(device, opt.input_size)

    # Accumulate loss before optimizing
    accumulate = max(1, round(opt.nominal_batch_size / opt.total_batch_size))
    # Linear scale weight decay
    opt.weight_decay *= \
        opt.total_batch_size * accumulate / opt.nominal_batch_size
    if local_rank in [-1, 0]:
        print('Scaled weight decay to %g' % opt.weight_decay)

    # Optimizer parameter groups
    params_weight, params_bias, params_except = [], [], []
    for module in model.modules():
        if hasattr(module, 'weight'):
            if isinstance(module.weight, nn.Parameter):
                if isinstance(module, nn.BatchNorm2d):
                    params_except.append(module.weight)
                else:
                    params_weight.append(module.weight)
        if hasattr(module, 'bias'):
            if isinstance(module.bias, nn.Parameter):
                params_bias.append(module.bias)
    optimizer = optim.SGD(params_bias,
                          lr=opt.initial_lr,
                          momentum=opt.momentum,
                          nesterov=True)
    optimizer.add_param_group({'params': params_weight,
                               'weight_decay': opt.weight_decay})
    if len(params_except) > 0:
        optimizer.add_param_group({'params': params_except})
    del params_weight, params_bias, params_except

    if opt.cosine_lr:
        # Cosine learning rate from 1.0 to final_lr
        lr_lambda = decay_lambda(opt.final_lr,
                                 total_steps=opt.epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=lr_lambda)
    else:
        if len(opt.lr_steps) == 0:
            if opt.data_type in ['svhn', 'cifar10', 'cifar100']:
                opt.lr_steps = [round(0.50 * opt.epochs),
                                round(0.75 * opt.epochs)]
            else:
                opt.lr_steps = [round(0.33 * opt.epochs),
                                round(0.66 * opt.epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=opt.lr_steps,
                                                   gamma=0.1)
    if opt.sync_bn:
        model.apply_sync_bn(local_rank)
        if local_rank in [-1, 0]:
            print('Using synchronized batchnorm')
    if opt.model_ema:
        model.apply_ema(local_rank)
        if local_rank in [-1, 0]:
            print('Using exponential moving average')
    model.apply_ddp(local_rank)
    if local_rank == 0:
        print('Using distributed data parallel')

    start_epoch, best_accuracy = 0, 0
    if opt.weights:
        if opt.resume:
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_accuracy = ckpt['best_accuracy']
            if local_rank in [-1, 0] and ckpt.get('model_ema'):
                if opt.model_ema:
                    model.update_ema(ckpt['model_ema'].float().state_dict(),
                                     updates=ckpt['updates'])
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, 'Training is finished'
            scheduler.last_epoch = start_epoch - 1
        del ckpt, state_dict

    # Create dataset loader
    trainloader = DataLoader(opt.data_root,
                             opt.data_type,
                             data_split='train',
                             input_size=opt.input_size,
                             batch_size=opt.batch_size,
                             data_augment=opt.data_augment,
                             hyp_params=hyp_params,
                             download=True,
                             shuffle=True,
                             num_workers=opt.workers,
                             local_rank=local_rank)
    if local_rank in [-1, 0]:
        testloader = DataLoader(opt.data_root,
                                opt.data_type,
                                data_split='val',
                                input_size=opt.input_size,
                                batch_size=opt.batch_size,
                                data_augment=False,
                                hyp_params=hyp_params,
                                download=True,
                                shuffle=False,
                                num_workers=opt.workers,
                                local_rank=-1)
    # Start training network
    num_batchs = len(trainloader)
    model.module.data_type = opt.data_type
    model.module.classes = trainloader.dataset.classes
    use_cuda = device.type != 'cpu'
    scaler = amp.GradScaler(enabled=use_cuda)
    criterion = Criterion()
    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter(opt.save_path)
        print('Dataloader workers %g' % trainloader.num_workers)
        print('Input size %g, batch size %g' % (opt.input_size,
                                                opt.total_batch_size))
        print('Training %s on %s for %g epochs' % (opt.model_type,
                                                   opt.data_type,
                                                   opt.epochs))
        print('Logging results to %s\n' % opt.save_path)

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        mean_loss = torch.zeros(1, device=device)
        if local_rank in [-1, 0]:
            print(('%10s' * 5) % ('Epoch',
                                  'memory',
                                  'loss',
                                  'step',
                                  'image'))
        if local_rank != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        if local_rank in [-1, 0]:
            pbar = tqdm(pbar, total=num_batchs)
        optimizer.zero_grad()

        for index, (images, targets) in pbar:
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device)
            total_steps = num_batchs * epoch + index

            # Warmup stage learning rate
            if total_steps <= opt.warmup_steps:
                x_pos = [0, opt.warmup_steps]
                y_pos = [1.0, opt.nominal_batch_size / opt.total_batch_size]
                accumulate = max(1, np.interp(total_steps,
                                              x_pos,
                                              y_pos).round())
                for param_index, params in enumerate(optimizer.param_groups):
                    # Learning rate rise from 0 to final_lr
                    if opt.cosine_lr:
                        final_lr = opt.initial_lr * lr_lambda(epoch)
                    else:
                        final_lr = opt.initial_lr
                    y_pos = [0, final_lr]
                    params['lr'] = np.interp(total_steps, x_pos, y_pos)
                    if 'momentum' in params:
                        params['momentum'] = np.interp(
                            total_steps,
                            x_pos,
                            [opt.warmup_momentum, opt.momentum])

            with amp.autocast(enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, targets)
                # DDP gradients averaged between devices
                if local_rank != -1:
                    loss *= opt.world_size
            # Scale loss using mixed precision
            scaler.scale(loss).backward()
            if total_steps % accumulate == 0:
                # Apply gradient descent
                scaler.step(optimizer)
                scaler.update()
                # Clean accumulated gradients
                optimizer.zero_grad()
                # Update model exponential moving average
                if opt.model_ema:
                    model.update_ema()

            if local_rank in [-1, 0]:
                mean_loss = (mean_loss * index + loss.item()) / (index + 1)
                memory_used = 0
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_reserved(device) / 1e9
                desc = ('%10s' * 2 + '%10.4g' * 3) % (
                    '%g/%g' % (epoch, opt.epochs - 1),
                    '%.4gG' % memory_used,
                    mean_loss,
                    scheduler.get_last_lr()[-1],
                    images.shape[-1])
                pbar.set_description(desc)

        scheduler.step()
        if local_rank in [-1, 0]:
            results = (0, 0, 0)
            if epoch == opt.epochs - 1 or not opt.notest:
                if opt.model_ema:
                    test_model = model.model_ema.module
                else:
                    test_model = model.module
                results = evaluate(test_model,
                                   device,
                                   dataloader=testloader,
                                   criterion=criterion)
            txt_path = os.path.join(opt.save_path, 'results.txt')
            with open(txt_path, 'a') as fd:
                fd.write((desc + '%10.4g' * 3 + '\n') % results)
            if results[0] > best_accuracy:
                best_accuracy = results[0]

            # Log results using tensorboard
            num_groups = len(optimizer.param_groups)
            lr_tags = ['train/lr%d' % k for k in range(num_groups)]
            tb_tags = ['train/loss', *lr_tags, 'metrics/top1',
                       'metrics/top5', 'metrics/loss']
            lrs = [params['lr'] for params in optimizer.param_groups]
            tb_vals = list(mean_loss) + lrs + list(results)
            for tag, val in zip(tb_tags, tb_vals):
                tb_writer.add_scalar(tag, val, epoch)

            # Save model to storage
            half_model_ema, ema_updates = None, None
            if opt.model_ema:
                half_model_ema = deepcopy(model.model_ema.module).half()
                ema_updates = model.model_ema.updates
            half_model = deepcopy(model.module).half()
            saved_ckpt = {'epoch': epoch,
                          'best_accuracy': best_accuracy,
                          'model': half_model,
                          'model_ema': half_model_ema,
                          'updates': ema_updates,
                          'optimizer': optimizer.state_dict()}
            last_ckpt_path = os.path.join(opt.save_path, 'last.pt')
            best_ckpt_path = os.path.join(opt.save_path, 'best.pt')
            torch.save(saved_ckpt, last_ckpt_path)
            if best_accuracy == results[0]:
                torch.save(saved_ckpt, best_ckpt_path)
            del saved_ckpt
    # Strip optimizers
    if local_rank in [-1, 0]:
        for file_path in [last_ckpt_path, best_ckpt_path]:
            strip_optimizer(file_path)
    else:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    return best_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset options
    parser.add_argument('--data_root', type=str, default='./datasets',
                        help='dataset root')
    parser.add_argument('--data_type', type=str, default='cifar10',
                        help='dataset type')
    parser.add_argument('--input_size', type=int, default=-1,
                        help='input size')
    parser.add_argument('--in_channels', type=int, default=-1,
                        help='input channels')
    parser.add_argument('--num_classes', type=int, default=-1,
                        help='dataset classes')
    parser.add_argument('--image_mean', type=list, default=[],
                        help='image mean')
    parser.add_argument('--image_std', type=list, default=[],
                        help='image std')
    parser.add_argument('--data_augment', type=bool, default=True,
                        help='data augmentation')
    parser.add_argument('--random_flip', type=float, default=0.5,
                        help='random flip')
    parser.add_argument('--random_crop', type=float, default=0.5,
                        help='random crop')
    parser.add_argument('--random_hsv', type=list, default=[0.015, 0.7, 0.4],
                        help='random hsv')
    # Model options
    parser.add_argument('--model_type', type=str, default='resnet20',
                        help='model type')
    parser.add_argument('--hidden_channels', type=int, default=2048,
                        help='hidden channels of head')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout rate of hidden channels')
    parser.add_argument('--sync_bn', type=bool, default=False,
                        help='sync batchnorm')
    parser.add_argument('--model_ema', type=bool, default=True,
                        help='model ema')
    # Training options
    parser.add_argument('--weights', type=str, default='',
                        help='weights path')
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training')
    parser.add_argument('--epochs', type=int, default=180,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='total batch size')
    parser.add_argument('--nominal_batch_size', type=int, default=128,
                        help='nominal batch size')
    parser.add_argument('--save_path', type=str, default='./results',
                        help='save path')
    parser.add_argument('--notest', type=bool, default=False,
                        help='only test final epoch')
    parser.add_argument('--lr_steps', type=list, default=[],
                        help='learning rate decay steps')
    parser.add_argument('--cosine_lr', type=bool, default=False,
                        help='cosine learning rate')
    parser.add_argument('--initial_lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--final_lr', type=float, default=0.01,
                        help='final learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='warmup steps')
    parser.add_argument('--warmup_momentum', type=float, default=0.8,
                        help='warmup momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay')
    # Platform options
    parser.add_argument('--benchmark', type=bool, default=False,
                        help='benchmark mode')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='DDP local rank')
    parser.add_argument('--local_port', type=int, default=1234,
                        help='DDP local port')
    parser.add_argument('--device', type=str, default='0',
                        help='cuda device')
    parser.add_argument('--workers', type=int, default=4,
                        help='dataloader workers')
    opt = parser.parse_args()

    if opt.data_type == 'ilsvrc2012':
        if not opt.data_root or opt.data_root == './datasets':
            opt.data_root = '/home/ubuntu/DataSets/ILSVRC2012'
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
