import io
import numpy as np 

from torch import nn
import torch.onnx
import models.resnet_last_down_extract as resnet_down
model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

torch_model = resnet_down.__dict__[model_names[0]]()

model_path = 'C:/Users/82109/Desktop/Projects/Thesis/ResNet-Skip-Connection-KD/vanilla_kd_model_saved_base/resnet18_student.pth'
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
                  "Student_model.onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})
