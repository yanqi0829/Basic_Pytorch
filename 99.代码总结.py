# 开发人员   wyq
# 开发时间：2023/8/26 8:20
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, ReLU, Linear, CrossEntropyLoss
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(3, 64, 3, stride=1, padding=0),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(14400, 1024),
            Linear(1024, 10)
        )

    def forward(self, x):
        out = self.model1(x)
        return out


test = Test()
print(test)
ones = torch.ones((3, 32, 32))
ones = torch.reshape(ones, (1, 3, 32, 32))
output = test(ones)
print(output)

dataset = torchvision.datasets.CIFAR10("./dataset", download=True, transform=transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=24, shuffle=False)

loss = CrossEntropyLoss()
optim=torch.optim.SGD(test.parameters(),lr=0.01,)

for echo in range(20):
    running_loss=0
    print(f"#############第{echo}轮训练##############")
    for datas in dataloader:  # 目前为止，整个数据集计算的损失函数没有明显变小；当前图像我们只看了一遍,需要echo多轮
        imgs, targets = datas
        outputs = test(imgs)
        # print(outputs.shape)
        item_loss = loss(outputs, targets)
        # print(item_loss)
        optim.zero_grad()
        item_loss.backward()#反向传播计算梯度
        optim.step()
        running_loss+=item_loss
    print(running_loss)