import argparse
import torch
import utils


parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-size', type=int, default=448, help='input image size')
parser.add_argument('-pretrain', type=str, required=True)
parser.add_argument('-nc', type=int, default=200, help='number of classes')
args = parser.parse_args()
print(args)

device = torch.device('cuda:0')

net = utils.get_network(args.net, args.nc, device, False)
utils.load_state(args.pretrain, net)

test_loader = utils.initialize_loader(istrainset=False, batchsize=64, input_size=args.size)
accr, class_correct, class_total, conf_matrix = utils.accuracy(test_loader, net, args.nc, device, disp=True)
print("Accr: {:.2%}.".format(accr))
