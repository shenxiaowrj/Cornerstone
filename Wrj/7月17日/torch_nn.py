# _*_coding:utf-8_*_
# Model Layers
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设置一个输入
input_image = torch.rand(3,28,28)
print(input_image.size())
# 再次回顾一下 pytorch中图片的维度数 为batch_size channels height weight

# nn.Flattern
flattern = nn.Flatten()
flat_image = flattern(input_image)
print(flat_image.size())
'''
class torch.nn.Flatten(start_dim=1, end_dim=- 1)
凡是class类，都需要实例化

默认将tensor的除了第0个维度外 其他的维度进行展平的处理 
在图像中也就是 保留第0维batch_size 将其它维度进行展平 相乘即可

>>> input = torch.randn(32, 1, 5, 5)
>>> # With default parameters
>>> m = nn.Flatten()
>>> output = m(input)
>>> output.size()
torch.Size([32, 25])
>>> # With non-default parameters
>>> m = nn.Flatten(0, 2)
>>> output = m(input)
>>> output.size()
torch.Size([160, 5])

'''

# nn.Linear 构造线性层的方法 也就是一个多元函数
layer1 = nn.Linear(in_features=28*28,out_features=20)
hidden1 = layer1(flat_image)
hidden2 = hidden1
print(hidden1.size())
'''
class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None) 默认是有bias的 没有bias不好进行拟合
Variables 变量
有两个成员变量 .weight .bias 可以直接使用 进行输出和查看

m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
'''

print(f"layer1 weight:{layer1.weight}")
print(f"layer1.bias:{layer1.bias}")

# nn.Relu 非线性层 将负数全部变为0
print(f"Before Relu:{hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After Relu: {hidden1}")

'''
class torch.nn.ReLU(inplace=False)
  >>> m = nn.ReLU()
  >>> input = torch.randn(2)
  >>> output = m(input)


An implementation of CReLU - https://arxiv.org/abs/1603.05201    这是个啥东西呀？？？

  >>> m = nn.ReLU()
  >>> input = torch.randn(2).unsqueeze(0)
  >>> output = torch.cat((m(input),m(-input)))
'''
# CReLU()
m = nn.ReLU()
input = hidden2
print(f"Before CReLU:{input}\n\n")
output = torch.cat((m(input),m(-input)))
print(f"After CReLU:{output}")


# nn.Sequential
seq_modules = nn.Sequential(
    flattern,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)

input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
print(f"logits:{logits},logits:{logits.shape}")
'''
class torch.nn.Sequential(*args)
A sequential container.
# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
这是啥意思啊？我不理解
'''

# nn.Softmax 利用指数求出每个结果代表的概率
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"pred_probab:{pred_probab}, pred_probab:{pred_probab.shape}")
'''
class torch.nn.Softmax(dim=None)

m = nn.Softmax(dim=1)
input = torch.randn(2, 3)
output = m(input)
'''

# Model Parameters
# 前面构建模型的代码
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

# nn.Parameter 每一个部分的具体的名字 size 数值values

print(f"Model structure:{model}\n\n")

for name,param in model.named_parameters():
    print(f"Layer: {name} | Size:{param.size()} | Values: {param[:2]} \n")

