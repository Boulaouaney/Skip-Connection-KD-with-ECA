
from torchviz import make_dot
import argparse
import torch.onnx

import models.resnet_ECA_parallel_SC as resnet_ECA_parallel
import models.resnet_last_down_extract as resnet_down_origin
import models.resnet_ECA_last_block_SC as resnet_ECA_last
import hiddenlayer as hl

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training with KD')
parser.add_argument('--pair_keys', type=int, required=True,
                    help='---Indicate pair of keys unique for teacher and student---')
parser.add_argument('--model', type=str, required=True, help='---Model type: resnet18, resnet34, resnet50---')
parser.add_argument('--ECA', type=str, default='yes', help='---model with ECA or without')
parser.add_argument('--ECA_block', type=str, default='parallel', help='---ECA block in parallel or last')
args, unparsed = parser.parse_known_args()
model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def build_model_ECA_parallel():
    if args.model == 'resnet18':
        return resnet_ECA_parallel.__dict__[model_names[0]]()
    elif args.model == 'resnet34':
        return resnet_ECA_parallel.__dict__[model_names[1]]()
    elif args.model == 'resnet50':
        return resnet_ECA_parallel.__dict__[model_names[2]]()
    elif args.model == 'resnet101':
        return resnet_ECA_parallel.__dict__[model_names[3]]()
    elif args.model == 'resnet152':
        return resnet_ECA_parallel.__dict__[model_names[4]]()


def build_model_ECA_last():
    if args.model == 'resnet18':
        return resnet_ECA_last.__dict__[model_names[0]]()
    elif args.model == 'resnet34':
        return resnet_ECA_last.__dict__[model_names[1]]()
    elif args.model == 'resnet50':
        return resnet_ECA_last.__dict__[model_names[2]]()
    elif args.model == 'resnet101':
        return resnet_ECA_last.__dict__[model_names[3]]()
    elif args.model == 'resnet152':
        return resnet_ECA_last.__dict__[model_names[4]]()


def build_model_origin():
    if args.model == 'resnet18':
        return resnet_down_origin.__dict__[model_names[0]]()
    elif args.model == 'resnet34':
        return resnet_down_origin.__dict__[model_names[1]]()
    elif args.model == 'resnet50':
        return resnet_down_origin.__dict__[model_names[2]]()
    elif args.model == 'resnet101':
        return resnet_down_origin.__dict__[model_names[3]]()
    elif args.model == 'resnet152':
        return resnet_down_origin.__dict__[model_names[4]]()


if args.ECA == 'no':
    torch_model = build_model_origin()
elif args.ECA == 'yes':
    if args.ECA_block == 'parallel':
        torch_model = build_model_ECA_parallel()
    elif args.ECA_block == 'last':
        torch_model = build_model_ECA_last()

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

make_dot(torch_out, params=dict(torch_model.named_parameters())).render(f"cnn_torchviz_{args.model}_{args.pair_keys}", format="png")


