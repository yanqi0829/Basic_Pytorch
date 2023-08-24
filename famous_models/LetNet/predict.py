# 开发人员   wyq
# 开发时间：2023/6/2 15:04

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('1.jpeg')  # 通过PIL或者numpy导入的是 [H,W,C]
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # 再加一个维度 [N, C, H, W]  由于Pytorch  Tesor 的通道排序  [batch,channel,height,width]

    with torch.no_grad():
        outputs = net(im)
        # predict = torch.max(outputs, dim=1)[1].numpy()
        predict=torch.softmax(outputs,dim=1)  #直接得到每个分类的概率
        print(predict)
    # print(classes[int(predict)])


if __name__ == '__main__':
    main()
