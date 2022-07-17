# _*_coding:utf-8_*_
# 完整走一遍流程

# 使用pytorch内置的库来创建dataset

import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt

# Dataset
training_data = datasets.FashionMNIST(
    root="data",  # 设置存放数据的根路径
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),


)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor())

# Iterating and visualizing the Dataset
# matplotlib的应用

# 创建标签映射图
labels_map = {
    0: "T_Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 设定figure的大小
figure = plt.figure(figsize=(8,8))
# cols 列 rows 行
cols , rows = 3,3

# 开始一个循环
for i in range(1, cols * rows +1):
    sample_idx = torch.randint(len(training_data),size=(1,)).item()  # 随机取出一个标签 用item()转化成数字
    img,label = training_data[sample_idx]  # 将标签传入sample_idx中 取回图片和标签
    figure.add_subplot(rows, cols, i)   #添加子图  以行，列来整合 i为第几个图片
    plt.title(labels_map[label])  # 设定标题 标题通过labels_map来进行映射
    plt.axis("off")  #设定轴
    plt.imshow(img.squeeze(),cmap="gray") # 设定颜色显示
plt.show()  #打开图片


# Dataloader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Iterate through the DataLoader
# Display image and label
train_features , train_labels = next(iter(train_dataloader))  #使用next函数取出所有的划分好batch的训练样本 和 训练标签
                                                              # iter方法得到迭代器 next方法依次取得一个一个的mini_batch
print(f"Feature batch shape: {train_features.size()}")   #训练数据的大小 一共有多少张图片
print(f"Feature batch shape: {train_labels.size()}")     #训练标签的大小 一共有多少个具体的标签
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label:{label}")



# Transform
# transform
# traget_transform
    # 对于得到的0-10的标签，要计算交叉熵损失的时候 处理成one-hot的形式 也就是对标签值利用scatter函数进行赋值

# 将0-9 转换成 one-hot向量的形式
target_transform = torchvision.transforms.Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

# 确定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 自定义一个网络
# 需要设定__init__ forward 两个函数
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()  #声明父类
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    # 前向运算的过程
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)   #在pytorch中Linear层的输出就是logits 另一种理解的角度就是能量
        return logits
# 得到logits之后讲过softmax 然后就可以得到预测的概率 之后就可以进行计算损失

# 实例化一个网络
            # 并将网络传入一个设备之中
model = NeuralNetwork().to(device)

# 输出网络的信息
# 方式一 pytorch内置的方法 print
print(model)

# 方式二 第三方库 pytorch summary
from torchsummary import summary
summary(model=model,input_size=(1,28,28),)

# 使用网络进行推理
x = torch.rand(1,28,28,device=device)
logits = model(x)
pred_probab = nn.Softmax(dim=1)(logits)  #按第一维传入logits
y_pred = pred_probab.argmax(1) # 利用argmax得到最大的一个概率值的索引
print(f"Predicted class:{y_pred}")  #得到 y预测的标签的索引

# 得到预测的标签的索引 然后将索引传入到对应的映射图之中 得到具体的语义上的映射
'''
pred_label = labels_map[y_pred]
print(f"Label:{pred_label}")
'''
'''
报错：
Traceback (most recent call last):
  File "D:\pytorch code\deep_thoughts\6_transforms_torch_nn\dataloader_dataset_transform.py", line 140, in <module>
    pred_label = labels_map[y_pred]
KeyError: tensor([6], device='cuda:0')
'''
# 分析：传回的是一个tensor labels_map里映射不到tensor 所以使用item()方法 将tensor转化为int
pred_label = labels_map[y_pred.item()]
print(f"Label:{pred_label}")
# Label:Ankle Boot 成功解决！


