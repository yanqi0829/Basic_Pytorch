# 开发人员   wyq
# 开发时间：2023/1/3 11:36

import torch

# 保存方式1-->  加载模型
import torchvision

model = torch.load("vgg16_method1.pth")
# print(model)#输出模型结构
# 保存方式2-->  加载模型
model = torch.load("vgg16_method2.pth")
print(model)#字典格式 的网络参数

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16) #此时打印的就是模型结构了

#陷阱 当使用自定义的模型时，加载时；要引入定义的模型类