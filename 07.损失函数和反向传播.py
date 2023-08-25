# 开发人员   wyq
# 开发时间：2023/1/2 16:33
import torch
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss

'''
torch.nn  Loss Functions
    损失函数：
        1.计算实际输出和目标之间的差距
        2.为我们更新输出提供一定的依据（反向传播），对（卷积层或线性层的 weight和bias）每一个需要调整的参数提供了梯度 grad
    nn.L1Loss ：比较简单   差值的和 求均值     为什么是求均值，不是求和   reduction='mean'默认是均值
    nn.MSELoss：均方差     差值的平方 求均值    为什么是求均值，不是求和   reduction='mean'默认是均值
    nn.CrossEntropyLoss  交叉熵，用于N个类别的分类问题，
            
'''

# ----------------------L1Loss--------------
import torch

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)#输入值
target = torch.tensor([2, 2, 2], dtype=torch.float32)#目标值
inputs = torch.reshape(inputs, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))
loss = L1Loss()
out = loss(inputs, target)
print(out)

# ----------------------------均方差MSE-----------------------------------
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss_mse = MSELoss()
result_mse = loss_mse(inputs, target)
print(result_mse)  # tensor(1.3333)
# ----------------------------交叉熵-----------------------------------
x = torch.tensor([0.1, 0.2, 0.3]) #三分类，给出了每个分类的当前概率
y = torch.tensor([1])  # target 实际是1号类别，即x中的第二个位置
x = torch.reshape(x, (1, 3)) #  1为batch size  3 为number of  class
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)  # tensor(1.1019)   -1*0.2+ln(exp(0.1)+exp(0.2)+exp(0.3))   可复制到谷歌进行计算；0.2为1号类别的概率
