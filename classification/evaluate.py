import torch
from tqdm import tqdm


def evaluate(model,
             device,
             dataloader,
             criterion=None):
    half_precision = device.type != 'cpu'
    if half_precision:
        model.half()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # Switch to evaluate mode
    model.eval()
    with torch.no_grad():
        desc = ('%10s' * 3) % ('top1',
                               'top5',
                               'loss')
        for images, targets in tqdm(dataloader, desc=desc):
            images = images.to(device, non_blocking=True)
            if half_precision:
                images = images.half()
            targets = targets.to(device)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, targets)
            else:
                loss = torch.zeros(1, device=device)
            if half_precision:
                outputs = outputs.float()
                loss = loss.float()
            # Measure accuracy and record loss
            acc = accuracy(outputs, targets)
            losses.update(loss.item(), targets.size(0))
            top1.update(acc[0].item(), targets.size(0))
            top5.update(acc[1].item(), targets.size(0))
        print(('%10.4g' * 3 + '\n') % (top1.avg,
                                       top5.avg,
                                       losses.avg))
    if half_precision:
        model.float()
    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val * num
        self.cnt += num
        self.avg = self.sum / self.cnt


def accuracy(output, target):
    # Computes the top1 and top5 precision
    topk = [1, 5]
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output.topk(maxk, 1, True, True)[1]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
