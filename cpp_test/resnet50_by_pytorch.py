import torch
import torchvision.models as models
from PIL import Image
import numpy as np
#加载测试图片调整大小
image = Image.open("test.jpg") #图片发在了build文件夹下
image = image.resize((224, 224),Image.ANTIALIAS)
#进行预处理
image = np.asarray(image)
image = image / 255
image = torch.Tensor(image).unsqueeze_(dim=0)
#变换维度
image = image.permute((0, 3, 1, 2)).float()
#加载使用pytorch自带resnet50模型
model = models.resnet50(pretrained=True)
#模型用于测试
model = model.eval()

# torch.jit.trace
# Trace a function and return an executable or ScriptFunction
# that will be optimized using just-in-time compilation.
resnet = torch.jit.trace(model, torch.rand(1,3,224,224))
# output=resnet(torch.ones(1,3,224,224))
#使用测试模型转换
output = resnet(image)
max_index = torch.max(output, 1)[1].item()
print(max_index) # ImageNet1000类的类别序
#保存转化后的模型
resnet.save('pytorch_resnet.pt')
