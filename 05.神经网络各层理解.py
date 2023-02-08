# 开发人员   wyq
# 开发时间：2023/1/2 9:54


import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Linear
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''
参考官网 torch.nn的 Container骨架，同时下方还有各种层
        Container下的Module是非常重要的
'''


# ----------------------------模型标准写法 ，这仅仅是个小例子，继续往下看-----------------------------------

class Model(nn.Module):  # 继承Module类
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):  # 前向传播  参数 x：输入
        x = F.relu(self.conv1(x))  # 先卷积，后非线性处理
        return F.relu(self.conv2(x))  # 先卷积，后非线性处理


# ----------------------------卷积操作的理解，不用记忆 -----------------------------------
'''
https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
torch.nn.functional 的方法有助于了解卷积，不用过多记忆
    input和kernel的数据选取没有什么特殊，和视频讲解保持一致 
'''
# 距离 ：输入是5*5的图像
input = torch.Tensor([[1, 2, 0, 3, 1], [0, 1, 2, 3, 1], [1, 2, 1, 0, 0], [5, 2, 3, 1, 1], [2, 1, 0, 1, 1]])
# 卷积核：是3*3的
kernel = torch.Tensor([[1, 2, 1], [0, 1, 0], [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))  # （batch_size,channel,宽高)
kernel = torch.reshape(kernel, (1, 1, 3, 3))
# F.conv2d     input 输入   kernel 卷积核  stride 移动步长（水平、垂直方向）默认1 ，padding默认 0不填充；1表示四周填充1层，填充0
output = F.conv2d(input, kernel, stride=1)
print(output)
output = F.conv2d(input, kernel, stride=2)
print(output)
output = F.conv2d(input, kernel, stride=1, padding=1)  # padding:四周填充1层，再进行卷积操作
print(output)


# ----------------------------创建完整模型 -----------------------------------
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        '''
         卷积  in_channels输入通道   out_channels输出通道      kernel_size卷积核大小  stride横向纵向步长
              卷积核，偏置的值无需指定大小，具体的值是通过分布采样得到的；其实初始值是多少并不重要，训练的过程就是对这些值的调整
              输出图像的宽高 官网有计算公式 
        '''
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        '''
        最大池化：减少计算量
        下采样:通过池化等方式减少特征图的宽和高就叫下采样  
            kernel_size 池化核   stride步长默认值是池化核大小   ceil_mode 默认False：步长移动后超出边界不计算  
        '''
        self.maxpool1 = MaxPool2d(kernel_size=3)
        '''
        非线性 激活函数  Non-linear Activations  
        非线性变换主要目的：引入些非线性特征，非线性越多，才能训练出符合各种特征的模型
        Relu:   
              inplace  默认False 保留原始数据
                            Input是-1   计算后 Input为-1   Output为0
                       True
                            Input是-1   计算后 Input为0
        
        '''
        self.relu1 = ReLU()
        '''
        线性层（隐藏层） nn.Linear
        '''
        self.linear1 = Linear(172800, 10)  # 172800=64*3*30*30

    def forward(self, x):
        # 分别测试 卷积层  池化层   非线性激活，可以自己造输入数据，分别注释进行测试，TenserBoard查看图片效果
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        return x


test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
model = MyModel()
# print(model)

step = 0
writer = SummaryWriter("CIFAR10_logs")
for data in dataloader:
    imgs, targets = data
    output = model(imgs)  # 每一批的数据都扔到模型里
    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # print(output.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("CIFR10_model_input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))  # 大于3通道无法显示，-1代表不确定，由计算得出具体值
    writer.add_images("CIFR10_model_conv", output, step)  # 实际上显示的图片个数是64*2  相当于将深度拆了
    step += 1
