# 开发人员   wyq
# 开发时间：2023/1/3 11:16
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

#保存方式1  模型结构+模型参数
torch.save(vgg16,"vgg16_method1.pth")

#保存方式2 模型参数 （官方推荐）
torch.save(vgg16.state_dict(),"vgg16_method2.pth")