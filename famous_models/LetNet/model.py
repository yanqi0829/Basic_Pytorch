# 开发人员   wyq
# 开发时间：2023/5/31 14:36
'''
Pytorch  Tesor 的通道排序  [batch,channel,height,width]

矩阵的尺寸由卷积核大小和步长计算得到
矩阵的深度为卷积核的个数
'''

import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):  # 定义类，继承nn.Module
    def __init__(self):  # 定义初始化函数
        super(LeNet, self).__init__()
        '''
        nn.Conv2d
            ctrl点击进去查看init， 
                in_channels：输入特征矩阵的深度
                out_channels：卷积核的个数，即生成深度为多少维的特征矩阵
                kernel_size：卷积核大小
                stride：步长，默认1
                padding：四周填充 ，默认不填充 0 ，按上下和左右进行填充
                bias：偏置，默认使用    
            矩阵尺寸大小由公式计算可得： N=（W-F+2P）/S+1          
        '''
        self.conv1 = nn.Conv2d(3, 16, 5)
        '''
        MaxPool2d:
                kernel_size池化核大小
                stride  步长
        '''
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 输出为10类别

    def forward(self, x):  # x代表输入数据  [batch,channel,height,width]
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        # 全连接层的输入是一维向量
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


import torch

input1 = torch.rand([32, 3, 32, 32])
model = LeNet()
print(model)
output = model(input1)
