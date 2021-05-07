# import torch
# import torchvision
# from IPython import embed
#
# # An instance of your model.
# model = torchvision.models.resnet18()
#
# # An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 224, 224)
#
# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(model, example)
# embed()

import torch
import torch.nn as nn
import torchvision
from IPython import embed

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

net = Net()
example_weight = torch.rand(1, 1, 3, 3)
example_forward_input = torch.rand(1, 3, 224, 224)

# 使用resnet18测试
model = torchvision.models.resnet18()
# Trace a specific method and construct `ScriptModule` with
# a single `forward` method
traced_model = torch.jit.trace(model, example_forward_input)
embed()
