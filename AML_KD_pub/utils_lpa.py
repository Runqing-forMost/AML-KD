import torch
import time
from torchvision import transforms
from data.half_cifar_dataloader import CIFAR100
from torch.utils.data import DataLoader
from utils import *
from models import model_dict
import torch.nn.functional as F
from wrapper import wrapper
import data.cifar_dataloader

__all__ = ['cross_entropy', 'create_dataloader_student', 'test', 'create_model', 'create_a2adataloader']


def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(
        1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def create_dataloader_student(args):
    # return train && test dataloader
    transform_train_light = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    train_data = CIFAR100(root=args.data_root, train=True, heavy_transform=transform_train_light,
                          light_transform=transform_train_light, noise_mode=args.noise_mode,
                          noise_ratio=args.r, noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode),
                          t_file='%s/teacher.json' % args.data_path, s_file='%s/student.json' % args.data_path,
                          teacher=False)

    # print('dataset scale: ', len(train_data))
    test_data = CIFAR100(root=args.data_root, train=False, test_transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    return train_loader, test_loader


def test(model, test_loader, epoch, args):
    model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for x, target, _ in test_loader:
            x = x.cuda()
            target = target.cuda()
            with torch.no_grad():
                output, _, _ = model(x, is_weight=False)
                loss = F.cross_entropy(output, target)

            batch_acc = accuracy(output, target, topk=(1,))[0]
            loss_record.update(loss.item(), x.size(0))
            acc_record.update(batch_acc.item(), x.size(0))

    run_time = time.time() - start

    info = 'test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_loss:{:.3f}\t cls_acc:{:.2f}\n'.format(
        epoch + 1, args.epoch, run_time, loss_record.avg, acc_record.avg) if epoch is not None else 'teacher acc : %.2f' % acc_record.avg

    print(info)


def create_model(args):
    model_s = model_dict[args.arch_s](num_classes=args.num_class)
    model_t = model_dict[args.arch_t](num_classes=args.num_class)
    return wrapper(model_s).cuda(), wrapper(model_t).cuda()


def create_a2adataloader(args):
    transform_train_light = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    train_data = data.cifar_dataloader.CIFAR100(root=args.data_root, train=True, heavy_transform=transform_train_light,
                                                light_transform=transform_train_light, noise_mode=args.noise_mode,
                                                noise_ratio=args.r,
                                                noise_file='%s/%.1f_%s_whole2.json' % (
                                                    args.data_path, args.r, args.noise_mode),
                                                )

    test_data = data.cifar_dataloader.CIFAR100(root=args.data_root, train=False, test_transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    return train_loader, test_loader

