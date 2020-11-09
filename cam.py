import os
import torch
import argparse
import utils
import cv2
import numpy as np
import pandas as pd
import torchvision as tv 
# import pdb


def write_pred_gt(pred, gt, fn, class_map):
    num = pred.size(0)
    with open(fn, 'w') as f:
        for i in range(num):
            p_name = class_map.iloc[pred[i].item(), 1]
            g_name = class_map.iloc[gt[i].item(), 1]
            judge = "True" if p_name == g_name else "False"
            s = "{:02d} #{:#<5}# pred: {:25} gt: {}".format(i + 1, judge, p_name, g_name)
            print(s)
            f.writelines(s + '\n')


def tensor2img_batch(image_tensor, denormalize=True):
    image_tensor = tv.utils.make_grid(image_tensor.cpu(), nrow=4)
    if denormalize:
        image_tensor = (image_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) +
                        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    return image_tensor.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255


def main():
    net.eval()
    dataiter = iter(test_loader)
    imgs, labels = dataiter.next()
    imgs, labels = imgs.to(device), labels.to(device)
    with torch.no_grad():
        # It is to be considered whether outputs is tensor or tuple of tensors
        outputs, feature_map = net(imgs, cam=True)
    _, predicted = torch.max(outputs.data, 1)
    write_pred_gt(predicted, labels, txt_path, class_map)
    b, c, h, w = feature_map.size()
    activation_map = net.get_cam(feature_map)
    # pdb.set_trace()
    activation_map = activation_map.gather(dim=1, index=labels.view(-1, 1, 1, 1).expand(-1, 1, h, w)).expand(b, 3, h, w)  # [b, 3, h, w]
    activation_map = torch.nn.UpsamplingBilinear2d(448)(activation_map)
    activation_map.detach_()
    origin = tensor2img_batch(imgs)
    attention_map = tensor2img_batch(activation_map, denormalize=False)
    attention_map = attention_map.astype(np.uint8)
    attention_map = cv2.applyColorMap(attention_map, 2)
    final = origin * 0.4 + attention_map * 0.6
    cv2.imwrite(map_path, final.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet50', help='net type')
    parser.add_argument('-size', type=int, default=448, help='input image size')
    parser.add_argument('-pretrain', type=str, default='checkpoint1/resnet50/baselineIN/resnet50-last-best.pth')
    parser.add_argument('-nc', type=int, default=200, help='number of classes')
    parser.add_argument('-fn', type=str, default='result')
    parser.add_argument('-seed', type=int, default=64)
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:0')
    net = utils.get_network(args.net, args.nc, device, False)
    utils.load_state(args.pretrain, net)

    class_map = pd.read_table('/home/datasets/FG/CUB_200_2011/CUB_200_2011/class_map.txt', header=None, delim_whitespace=True)
    test_loader = utils.initialize_loader(istrainset=False, batchsize=16, input_size=args.size, shuffle=True)
    sub_root = 'cam/'
    if not os.path.exists(sub_root):
        os.mkdir(sub_root)
    sub_root = os.path.join(sub_root, str(args.seed))
    if not os.path.exists(sub_root):
        os.mkdir(sub_root)
    txt_path = os.path.join(sub_root, args.fn + '.txt')
    map_path = os.path.join(sub_root, args.fn + '.png')
    main()
