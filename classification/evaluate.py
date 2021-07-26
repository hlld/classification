import torch


def eval_network(model,
                 dataloader,
                 device):
    correct = 0
    total = 0
    # Switch model to eval mode
    model.eval()
    with torch.no_grad():
        for k, (images, targets) in enumerate(dataloader, 0):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            outputs = model(images)
            index = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            correct += torch.eq(index.indices, targets).sum().item()
    top1_accuracy = correct / total * 100
    return top1_accuracy
