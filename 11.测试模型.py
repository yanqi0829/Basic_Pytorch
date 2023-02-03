# 开发人员   wyq
# 开发时间：2023/1/3 16:34
'''
一般为  test.py
'''
import time

import torch
from PIL import Image
from torchvision import transforms

image_path = "./images/dog.png"
image = Image.open(image_path)
print(image)
image = image.convert("RGB")
print(image)

trans = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
image = trans(image)
print(image.shape)

# model = torch.load("model50.pth")  # GPU模式
model = torch.load("model50.pth",map_location=torch.device('cpu'))  # CPU
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
# image = image.cuda()
start=time.time()
model.eval()
with torch.no_grad():  # 节约内存 性能
    output = model(image)
print(output)
print(output.argmax(1))
end=time.time()
print(f"cost time {(end-start)}s") #GPU  1.67s   CPU:   0.007s
