# 开发人员   wyq
# 开发时间：2022/12/29 14:27
from torch.utils.data import Dataset
import os
from PIL import Image



'''
继承Dataset类
__getitem__    __len__ 为必须实现的方法
根据index 获取数据及其label      数据集数量
'''
'''
当然，数据集的格式是多样的，如label写在txt中
'''

class MyData(Dataset):
    def __init__(self, root_dir, label_dir): #该数据集 label按文件夹名称区分
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, index):
        image_name = self.image_path[index]
        #每张图片路径
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        img = Image.open(image_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.image_path)


root_dir = "hymenoptera_data/train"
ant_label_dir = "ants"
ant_dataset = MyData(root_dir, ant_label_dir)

img, label = ant_dataset[0]
img.show() #PIL类型直接显示图片

bee_label_dir = "bees"
bee_dataset = MyData(root_dir, bee_label_dir)
#可以对数据集进行拼接
train_dataset = ant_dataset + bee_dataset

