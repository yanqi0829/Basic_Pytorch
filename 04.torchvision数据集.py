# 开发人员   wyq
# 开发时间：2022/12/30 15:21
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

'''
官网提供数据集和模型   https://pytorch.org/vision/
数据集保存在root目录下，download下载，下载之后解压并校验
                    如果手动下载的数据集，解压并校验
如果下载的比较慢，将源码中的地址粘贴到迅雷中
        如：url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
'''
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
print(train_set[0])
print(type(train_set[0][0]))  # <class 'PIL.Image.Image'>  直接读取的格式；如果进行了转换就是Tensor类型
img, target = train_set[0]  # Tensor，label
print(test_set.classes)  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
writer = SummaryWriter("CIFAR10_logs")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("CIFAR10取了前十张图片", img, i)

# ----------------------------Dataloader 进行批处理-----------------------------------
'''
windows下 多进程num_worker只能为0;shuffle随机洗牌打乱顺序；drop_last丢弃不足batch_size的数据
'''
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
'''1W张图片，batch_size:64，需要157批次'''
img, targe = test_set[0]  # 测试集中第一张图片及target
print(img.shape)  # torch.Size([3, 32, 32])
print(targe)

step = 0
for data in test_loader:
    #dataLoader根据batch_size分批取数据 将照片数据和label数据分别做为集合
    imgs, targets = data
    # print(imgs.shape)   #torch.Size([64, 3, 32, 32])
    # print(targets)
    writer.add_images("test_data_dataloader", imgs, step)
    step = step + 1
writer.close()
