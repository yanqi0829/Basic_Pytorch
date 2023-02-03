# 开发人员   wyq
# 开发时间：2023/1/3 11:36

import torch

import torchvision
# 保存方式1-->对应的  加载模型
model = torch.load("vgg16_method1.pth")
print(model)#输出模型结构

# 保存方式2-->  加载模型
model = torch.load("vgg16_method2.pth")
print(model) #此时打印的内容是   字典格式 的网络参数；就不是网络结构了

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)  #此时打印的就是模型结构了^_^

'''
陷阱
    当使用方式1进行加载时，并且模型是自定义的
        这时，要引入定义的继承nn.Module的模型类
'''
