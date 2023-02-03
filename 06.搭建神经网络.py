# 开发人员   wyq
# 开发时间：2023/1/2 15:08
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

'''
数据集为CIFAR10 
使用的分类网络模型使用 详见  img.png
并且使用 Sequential搭建模型
'''


# 使用Sequential会让代码更加简洁
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 既然网络结构图有了，使用公式反推padding的值
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)
        '''
        使用Sequential
        self.model1=Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        '''

    def forward(self, x):  # 前向传播
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
        # 使用Sequential
        # x=self.model1(x)
        # return x


model = Model()
print(model)  # 查看网络结构
'''
对于自己写的模型可以   进行验证
input = torch.ones((64, 3, 32, 32))
output = model(input)
print(output.shape) 

writer = SummaryWriter("CIFAR10_logs")
writer.add_graph(model,input)
writer.close()
'''

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
loss = nn.CrossEntropyLoss()  # 分类问题损失函数
'''
torch.optim 
优化器：
    SGD   随机梯度下降
'''
optim = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, target = data
        outputs = model(imgs)  # 根据调优的参数再训练
        result_loss = loss(outputs, target)
        # print(result_loss)
        optim.zero_grad()  # 梯度设置为0
        result_loss.backward()  # 反向传播，【求梯度】  卷积层或线性层的 weight和bias 下看到梯度值grad
        optim.step()  # 参数调优， 参数会进行变化，有的值可能变化不大，多执行几次断点
        running_loss = running_loss + result_loss #这里计算每轮的loss
    print(running_loss)
    '''
    print(running_loss) 每轮的输出结果如下
        tensor(360.5676, grad_fn=<AddBackward0>)
        tensor(356.7508, grad_fn=<AddBackward0>)
        tensor(341.7326, grad_fn=<AddBackward0>)
        tensor(318.2594, grad_fn=<AddBackward0>)
        tensor(309.9871, grad_fn=<AddBackward0>)
        tensor(301.5600, grad_fn=<AddBackward0>)
        tensor(293.2404, grad_fn=<AddBackward0>)
        tensor(284.1416, grad_fn=<AddBackward0>)
        tensor(278.7745, grad_fn=<AddBackward0>)
        tensor(272.6930, grad_fn=<AddBackward0>)
        tensor(267.6430, grad_fn=<AddBackward0>)
        tensor(262.0797, grad_fn=<AddBackward0>)
        tensor(257.9688, grad_fn=<AddBackward0>)
        tensor(253.5258, grad_fn=<AddBackward0>)
        tensor(247.9718, grad_fn=<AddBackward0>)
        tensor(243.9755, grad_fn=<AddBackward0>)
        tensor(240.3479, grad_fn=<AddBackward0>)
        tensor(236.4041, grad_fn=<AddBackward0>)
        tensor(232.9967, grad_fn=<AddBackward0>)
        tensor(230.7793, grad_fn=<AddBackward0>)
    '''
