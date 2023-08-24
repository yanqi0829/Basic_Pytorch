# 开发人员   wyq
# 开发时间：2022/12/30 8:53
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
'''
对图像进行变换
transforms的结构及用法
    transforms.py相当于一个工具箱，看其Structure 有许多class类
'''
# 执行 tensorboard --logdir=logs/transforms

# ----------------------------①ToTensor-----------------------------------
'''
Convert a ``PIL Image`` or ``numpy.ndarray(Opencv读取的)`` to tensor
'''
image_path = "images/haizei.jpg"
img = Image.open(image_path)
# print(type(img))
trans_totensor = transforms.ToTensor()  # 创建实例对象
tensor_img = trans_totensor(img)  # 名称()”可以理解为是“对象.__call__()”的简写
# print(tensor_img)

# TensorBoard 使用Tensor展示图片
writer = SummaryWriter("logs/transforms")
writer.add_image("ToTensor", tensor_img)

# ----------------------------Normalize归一化-----------------------------------
'''
Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels
平均值  标准差
output[channel] = (input[channel] - mean[channel]) / std[channel]
    __init__(self, mean, std, inplace=False):
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
         input 是[0,1]  则经过归一化的结果output[-1,1]
'''
print(f'归一化之前 :{tensor_img[0][0][0]}')
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(f'归一化之后 :{img_norm[0][0][0]}')
writer.add_image("Normalize归一化", img_norm)
# ----------------------------Resize-----------------------------------
# print(img.size)
trans_resize = transforms.Resize((312, 312))
#img PIL  ->  resize  ->img_resize PIL
img_resize = trans_resize(img)
#img_resize PIL ->  resize  ->img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
# ----------------------------Compose-----------------------------------
trans_resize2 = transforms.Resize((100, 100))
trans_compose = transforms.Compose([trans_resize2, trans_totensor])
trans_resize2 = trans_compose(img)
writer.add_image("Resize", trans_resize2, 1)
# ----------------------------RandomCrop随机裁剪-----------------------------------
trans_random = transforms.RandomCrop(100, 100)
trans_compose2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose2(img)
    writer.add_image("RandomCrop随机裁剪", img_crop, i)

writer.close()
