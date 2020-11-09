import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from model import models
import dataset.getdata as getdata


def get_network(net_name, nc, device, IN=True):
    if net_name == "vgg19":
        net = models.VGG19(nc, IN)
    elif net_name == "resnet50":
        net = models.ResNet50(nc, IN)
    else:
        print("Not support net - {}".format(net_name))
        raise ModuleNotFoundError
    net = net.to(device)
    return net


def initialize_loader(istrainset, batchsize=8, input_size=448, worker=4, shuffle=None, corrupt=None):
    """Make some default variable settings"""
    root_dir = "/workspace/datasets/FG/CUB_200_2011/CUB_200_2011"
    list_txt = "train_list.txt" if istrainset else "test_list.txt"
    aug = istrainset
    if shuffle is None:
        shuffle = istrainset 
    return getdata.getloader(root_dir, list_txt, input_size, shuffle=shuffle, bs=batchsize, aug=aug, worker=worker, corrupt=corrupt)


def accuracy(dataloader, net, num_classes, device, disp=False):

    net = net.to(device)
    net.eval()
    correct = 0
    total = 0
    class_correct = list(0 for _ in range(num_classes))
    class_total = list(0 for _ in range(num_classes))
    conf_matrix = np.zeros([num_classes, num_classes])  # (i, j): i-Gt; j-Pr
    for data in iter(dataloader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            # It is to considered whether outputs is tensor or tuple of tensors
            outputs = net(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            conf_matrix[label, predicted[i].item()] += 1
    accr = correct / total
    if disp:
        print('Total number of images={}'.format(total))
        print('Total number of correct images={}'.format(correct))

    return accr, class_correct, class_total, conf_matrix


def information_entropy(scores, ita=1e-10):
    ps = nn.functional.softmax(scores, dim=1)
    results = -ps * torch.log(ps + ita)
    results = results.sum(1).mean()
    return results.item()


class CE_IELoss(nn.Module):
    """
    CrossEntropy Loss with Information Entropy as a regularization
    """
    def __init__(self, eps=3e-4, reduction='mean', ita=1e-10):
        super(CE_IELoss, self).__init__()
        self.eps = eps
        self.ita = ita
        self.nll = nn.NLLLoss(reduction=reduction)
        self.softmax = nn.Softmax(1)

    def update_eps(self):
        self.eps = self.eps * 0.9

    def forward(self, outputs, labels):
        """
        :param outputs: [b, c]
        :param labels: [b,]
        :return: a loss (Variable)
        """
        outputs = self.softmax(outputs)  # probabilities
        ce = self.nll(outputs.log(), labels)
        reg = outputs * torch.log(outputs + self.ita)
        reg = reg.sum(1).mean()
        loss_total = ce + reg * self.eps
        return loss_total, ce, reg


class LabelSmoothing(nn.Module):
    def __init__(self, nc, eps=0.1, reduction='mean', ita=1e-10):
        super(LabelSmoothing, self).__init__()
        self.nc = nc
        self.eps = eps
        self.ita = ita
        self.nll = nn.NLLLoss(reduction=reduction)
        self.softmax = nn.Softmax(1)

    def forward(self, outputs, labels):
        """
        :param outputs: [b, c]
        :param labels: [b,]
        :return: loss (Variable)
        """
        outputs = self.softmax(outputs)  # probabilities
        ce = self.nll(outputs.log(), labels)
        logs = torch.log(outputs + self.ita)
        sum_log = logs.sum(1).mean()
        loss_total = (1 - self.eps) * ce - self.eps * sum_log / self.nc
        return loss_total, ce


def save_state(save_filepath, epoch, net, optimizer, lr_scheduler, best_acc):
    save_state_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict()
        },
    }
    torch.save(save_state_dict, save_filepath)
    print("=> saving checkpoint at %s.\n" % save_filepath)


def load_state(save_filepath, net, optimizer=None, lr_schedular=None):
    if os.path.isfile(save_filepath):
        checkpoint = torch.load(save_filepath)
        net.load_state_dict(checkpoint["state_dict"]["net"])
        if optimizer is not None:
            opt_state_dict = checkpoint["state_dict"]["optimizer"]
            optimizer.load_state_dict(opt_state_dict)
        if lr_schedular is not None:
            sch_state_dict = checkpoint["state_dict"]["scheduler"]
            lr_schedular.load_state_dict(sch_state_dict)
        print("=> restore checkpoint from %s finished." % save_filepath)
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        return start_epoch, best_acc
    else:
        print("=> no checkpoint found at %s." % save_filepath)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
