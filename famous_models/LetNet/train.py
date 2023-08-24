# 开发人员   wyq
# 开发时间：2023/5/31 17:04
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    # 图像预处理
    '''
            Ctrl点击ToTensor()函数，主要内容描述如下
                Converts a PIL Image or numpy.ndarray (H x W x C) in the range
                [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            Normalize标准化:使用均值和标准差；数据变成[-1,1]区间
            '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 5W张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # train为True，会导入训练集样本
    # torchvision.datasets. 中有很多现成的数据
    train_set = torchvision.datasets.CIFAR10(root='../../dataset', train=True,
                                             download=False, transform=transform)
    #
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='../../dataset', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # 训练集迭代5轮

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            # print statistics
            running_loss += loss.item()  # 累加损失函数
            if step % 500 == 499:  # 每隔500步打印信息
                with torch.no_grad():  # with是上下文管理器   with torch.no_grad()：作用域内的代码不会计算梯度
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]  # dim=1，第0个维度是batch
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path) #保存模型


if __name__ == '__main__':
    main()
