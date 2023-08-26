# 开发人员   wyq
# 开发时间：2023/1/3 9:02
'''
torchvision下 Models and pre-trained weights
以分类预训练模型VGG为例 最常用的是VGG16  VGG19
'''
import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

# ImageNet  训练集太大 147.9G，所以不用
# train_data = torchvision.datasets.ImageNet("./dataset", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())

# vgg16=torchvision.models.vgg16(pretrained=True,progress=True)
vgg16_true = torchvision.models.vgg16(weights="DEFAULT",
                                      progress=True)  # 下载带参数的模型到  C:\Users\wangyq\.cache\torch\hub\checkpoints
vgg16_false = torchvision.models.vgg16(progress=True)  # 默认没有预训练权重，只加载模型结构,参数都是正太分布下的初始化
print(vgg16_true)  #VGG16的网路架构   可看出模型最终分类是1000个

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 如何将ImageNet1000分类的模型应用在  CIFAR 10分类数据集上面？

# 修改网络模型方式一，再添加一层
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

# 修改网络模型方式二  直接进行替换
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
