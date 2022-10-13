import io
import numpy as np 

from torch import nn
import argparse
import torch.onnx
import models.resnet_last_down_extract as resnet_down

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training with KD')
parser.add_argument('--pair_keys', type=int, required=True,
                    help='---Indicate pair of keys unique for teacher and student---')
parser.add_argument('--model', type=str, required=True, help='---Model type: resnet18, resnet34, resnet50---')
args, unparsed = parser.parse_known_args()
model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

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

torch_model = build_model()

model_path = f'./vanilla_kd_model_saved_base/{args.model}_student_{args.pair_keys}.pth'
batch_size = 1

map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None

state_dict = torch.load(model_path, map_location = map_location)
torch_model.load_state_dict(state_dict)

torch_model.eval()

x = torch.randn(batch_size, 3, 32, 32)

torch_out = torch_model(x)

torch.onnx.export(torch_model,
                  x,
                  f"./vanilla_kd_ECA_ONNX_model/{args.model}_Student_model_{args.pair_keys}.onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})
