import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from torch.utils.tensorboard import SummaryWriter
import utils


def train(epoch):
    net.train()
    loss_sum = 0.0
    ie_sum = 0.0
    for batch_index, (images, labels) in enumerate(train_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        labels = labels.cuda()
        images = images.cuda()

        outputs = net(images)
        if args.reg:
            loss, ce, reg = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            ie = -reg.item()
        elif args.lsr:
            loss, ce = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            ie = utils.information_entropy(outputs)
        else:
            ce = ce_loss(outputs, labels)
            optimizer.zero_grad()
            ce.backward()
            ie = utils.information_entropy(outputs)

        optimizer.step()
        loss_sum += ce.item()
        ie_sum += ie
        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1
        if batch_index % PRINT_ITER == PRINT_ITER - 1:
            loss_avg = loss_sum / PRINT_ITER
            ie_avg = ie_sum / PRINT_ITER
            loss_sum = 0.0
            ie_sum = 0.0
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]  Loss: {:0.4f}  IE: {:0.4f}  lr: {:0.6f}'.format(
                loss_avg,
                ie_avg,
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(train_loader.dataset)
            ))

            # update training loss for each iteration
            writer.add_scalar('Train_loss', loss_avg, n_iter)
            writer.add_scalar('Information_entropy', ie_avg, n_iter)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], n_iter)


def eval_training(epoch):
    net.eval()
    test_loss = 0.0
    ie = 0.0
    correct = 0.0
    for (images, labels) in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = net(images)
        loss = ce_loss(outputs, labels)
        test_loss += loss.item()
        ie += utils.information_entropy(outputs)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    acc = correct.float() / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Average IE: {:.4f}, Accuracy: {:.2%}, FormerBestAccuracy: {:.2%}'.format(
        test_loss / len(test_loader),
        ie / len(test_loader),
        acc,
        best_acc,
    ))

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(test_loader), epoch)
    writer.add_scalar('Test/Average IE', ie / len(test_loader), epoch)
    writer.add_scalar('Test/Accuracy', acc, epoch)
    return acc


if __name__ == '__main__':
    # import time
    # time.sleep(1200)
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-nc', type=int, default=200, help='number of classes')
    parser.add_argument('-size', type=int, default=448, help='input image size')
    parser.add_argument('-lr', type=float, default=1e-3, help='the learning rate')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-reg', action='store_true', help='enable to use regularization term')
    parser.add_argument('-lsr', action='store_true', help='enable to use label smoothing regularization')
    parser.add_argument('-pretrain', type=str, default='')
    parser.add_argument('-start', type=int, default=1, help='start id (epoch)')
    parser.add_argument('-IN', action='store_false', help="disable to use models pretrained on ImageNet")
    parser.add_argument('-corrupt', type=float, default=0.0, help="The probability that we mislabel a training image")
    parser.add_argument('-eps', type=float, default=0.5)
    args = parser.parse_args()
    print(args)
    torch.manual_seed(100)
    random.seed(100)
    MILESTONES = [40, 60, 70] if args.lr < 8e-3 else [20, 50, 70]  # [60, 110]
    GAMMA = 0.3 if args.lr < 8e-3 else 0.2  # 0.1
    CHECKPOINT_PATH = 'checkpoint1'
    LOG_DIR = 'runs1'
    EPOCH = 80  # 150
    SAVE_EPOCH = 10
    PRINT_ITER = 40
    if args.reg or args.lsr:
        eps1 = args.eps  # 0.5

    device = torch.device('cuda:0')

    net = utils.get_network(args.net, args.nc, device, args.IN)
    train_loader = utils.initialize_loader(istrainset=True, batchsize=args.b, input_size=args.size, worker=args.w, corrupt=args.corrupt)
    test_loader = utils.initialize_loader(istrainset=False, batchsize=32, input_size=args.size, worker=args.w)

    if args.reg:
        # loss_function = utils.CE_IELoss(eps=5e-1)
        loss_function = utils.CE_IELoss(eps=eps1)
    elif args.lsr:
        loss_function = utils.LabelSmoothing(args.nc, eps=eps1)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    iter_per_epoch = len(train_loader)
    if args.IN:
        if args.reg:
            sub_name = 'regIN'
            sub_name = "{}-{}-{}".format(sub_name, str(args.corrupt), str(args.eps))
        elif args.lsr:
            sub_name = 'lsrIN'
            sub_name = "{}-{}-{}".format(sub_name, str(args.corrupt), str(args.eps))
        else:
            sub_name = 'baselineIN'
            sub_name = "{}-{}".format(sub_name, str(args.corrupt))
    else:
        sub_name = 'reg' if args.reg else 'baseline'
    checkpoint_path = os.path.join(CHECKPOINT_PATH, args.net, sub_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    best_model_path = checkpoint_path.format(net=args.net, epoch='last', type='best')
    regular_model_path = checkpoint_path.format(net=args.net, epoch='tmp', type='regular')

    # use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    writer = SummaryWriter(os.path.join(LOG_DIR, args.net, sub_name))

    start_epoch = args.start
    best_acc = 1. / args.nc
    if args.pretrain:
        start_epoch, best_acc = utils.load_state(args.pretrain, net, optimizer, train_scheduler)
        args.warm = 0
    if args.warm == 0:
        warmup_scheduler = None
    else:
        warmup_scheduler = utils.WarmUpLR(optimizer, iter_per_epoch * args.warm)
    for epoch in range(start_epoch, EPOCH + 1):
        train(epoch)
        acc = eval_training(epoch)
        train_scheduler.step()
        if best_acc < acc:
            utils.save_state(best_model_path, epoch, net, optimizer, train_scheduler, best_acc)
            # torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch='last', type='best'))
            best_acc = acc

        if not epoch % SAVE_EPOCH:
            utils.save_state(regular_model_path, epoch, net, optimizer, train_scheduler, best_acc)
        # If we use reg term
        # if args.reg:
        #     if epoch == 30:
        #         loss_function.eps = 0.5
        #     elif epoch == 50:
        #         loss_function.eps = 0.2
        #     elif epoch == 70:
        #         loss_function.eps = 0.1
        # if args.lsr:
        #     if epoch == 30:
        #         loss_function.eps = 0.59
        #     elif epoch == 50:
        #         loss_function.eps = 0.23
        #     elif epoch == 70:
        #         loss_function.eps = 0.01

    writer.close()
