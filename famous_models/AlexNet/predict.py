import torch
from model import AlexNet
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

'''
图片的预处理  可以直接使用numpy，但pytorch更加方便
'''
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# load Image
img = Image.open("./test.jpg")
# [N,C,H,W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)  # shape  【1,3,224,224】

try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create Model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = './AlexNet.pth'
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    output = model(img)
    output = torch.squeeze(output)  # 压缩掉batch维度
    predict = torch.softmax(output, dim=0)  # 各节点概率
    cla = torch.argmax(predict, dim=0).numpy()  # tensor(2)
print(class_indict[str(cla)], predict[cla].item())
