# 开发人员   wyq
# 开发时间：2022/12/29 16:42

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

#启动服务：  tensorboard --logdir=logs  --port=8888
writer = SummaryWriter("logs") #文件夹路径
'''
常用的方法：
        writer.add_image()
        writer.add_images()
        writer.add_scalar()
        等  
注意：如果页面没有显示，注意close
        writer.close()
'''

# 可以是numpy，Tensor等类型
image_path = "hymenoptera_data/train/ants/6240329_72c01e663e.jpg"
image_PIL = Image.open(image_path)
# image_PIL.show()
img_array = np.array(image_PIL)
print(type(img_array))
img_array.shape
'''
①add_image注释中可见
    照片类型  Tensor  或  numpy-array
'''
#从PIL到numpy，需要在add_image中指定 shape中每个数字表示的含义
# writer.add_image("test1234", img_array, 0, dataformats='HWC') ,要指定shape格式
writer.add_image("测试tensorboard存照片", img_array, 1, dataformats='HWC')
writer.add_image("测试tensorboard存照片", img_array, 2, dataformats='HWC')

for i in range(100):
# ②add_scalar   tag（标题）, scalar_value（y轴）,global_step(x轴步长)
    writer.add_scalar("y=2x", 2 * i, i)
