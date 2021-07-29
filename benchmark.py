import argparse
from classification.datasets import DataLoader
from classification.evaluate import evaluate
from classification.tools import select_device, load_model, check_input_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='',
                        help='weights path')
    parser.add_argument('--data_root', type=str, default='./datasets',
                        help='dataset root')
    parser.add_argument('--data_type', type=str, default='cifar10',
                        help='dataset type')
    parser.add_argument('--data_split', type=str, default='test',
                        help='train, val or test')
    parser.add_argument('--image_mean', type=list, default=[],
                        help='image mean')
    parser.add_argument('--image_std', type=list, default=[],
                        help='image std')
    parser.add_argument('--input_size', type=int, default=-1,
                        help='image input size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='image batch size')
    parser.add_argument('--device', type=int, default=0,
                        help='cuda device')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of workers')
    opt = parser.parse_args()

    if opt.data_type == 'ilsvrc2012':
        if not opt.data_root or opt.data_root == '../datasets':
            opt.data_root = '/home/ubuntu/DataSets/ILSVRC2012'
    if opt.data_split not in ['train', 'val', 'test']:
        raise ValueError('Unknown type %s' % opt.data_split)

    device = select_device(opt.device)
    print('Loading model from %s ...' % opt.weights)
    model = load_model(opt.weights, device)
    # Infer default arguments
    if opt.data_type in ['mnist', 'svhn', 'cifar10',
                         'cifar100', 'ilsvrc2012']:
        input_size, _, _, image_mean, image_std = \
            DataLoader.default_params(opt.data_type)
        if opt.input_size <= 0:
            opt.input_size = input_size
        if len(opt.image_mean) == 0:
            opt.image_mean = image_mean
        if len(opt.image_std) == 0:
            opt.image_std = image_std
    elif opt.data_type == 'custom':
        if len(opt.image_mean) == 0 or len(opt.image_std) == 0 or \
                opt.input_size <= 0:
            raise ValueError('Customized dataset with default arguments')
    else:
        raise ValueError('Unknown type %s' % opt.data_type)
    hyp_params = {'mean': opt.image_mean,
                  'std': opt.image_std}
    opt.input_size = check_input_size(opt.input_size,
                                      model.max_stride)
    dataloader = DataLoader(opt.data_root,
                            opt.data_type,
                            data_split=opt.data_split,
                            input_size=opt.input_size,
                            batch_size=opt.batch_size,
                            data_augment=False,
                            hyp_params=hyp_params,
                            download=True,
                            shuffle=False,
                            num_workers=opt.workers,
                            local_rank=-1)
    evaluate(model,
             device,
             dataloader=dataloader)
