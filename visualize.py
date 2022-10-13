import io
import numpy as np
import torchvision
import torchvision.transforms as transforms

from torch import nn
#from torchviz import make_dot
import argparse
import torch.onnx
from torch.utils.data import DataLoader

import models.resnet_last_down_extract as resnet_down
import hiddenlayer as hl

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training with KD')
parser.add_argument('--pair_keys', type=int, required=True,
                    help='---Indicate pair of keys unique for teacher and student---')
parser.add_argument('--model', type=str, required=True, help='---Model type: resnet18, resnet34, resnet50---')
args, unparsed = parser.parse_known_args()
model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(
        root='./cifar10', train=True, download=True, transform=transform_train)

trainLoader = DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=0)

def build_model():
    if args.model == 'resnet18':
        return resnet_down.__dict__[model_names[0]]()
    elif args.model == 'resnet34':
        return resnet_down.__dict__[model_names[1]]()
    elif args.model == 'resnet50':
        return resnet_down.__dict__[model_names[2]]()
    elif args.model == 'resnet101':
        return resnet_down.__dict__[model_names[3]]()
    elif args.model == 'resnet152':
        return resnet_down.__dict__[model_names[4]]()

device = 'cuda:0'

torch_model = build_model().to(device)

model_path = f'./vanilla_kd_model_saved_base/{args.model}_student_{args.pair_keys}.pth'
batch_size = 1

# map_location = lambda storage, loc: storage
# if torch.cuda.is_available():
#     map_location = None

state_dict = torch.load(model_path, map_location = None)
torch_model.load_state_dict(state_dict)

torch_model.eval()


batch = next(iter(trainLoader))

batch = batch.to(device)
torch_out = torch_model(batch.text)

# make_dot(torch_out, params=dict(torch_model.named_parameters())).render("cnn_torchviz2", format="png")

transforms = [hl.transforms.Prune('Constant')]

graph = hl.build_graph(torch_model, batch.text, transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('cnn_hiddenlayer', format='png')
