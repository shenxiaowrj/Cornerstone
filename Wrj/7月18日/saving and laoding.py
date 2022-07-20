# _*_coding:utf-8_*_
'''简单的保存模型:只保存模型的权重'''

import torch
import torchvision.models as models

# 创建模型
wights = models.VGG16_Weights
model = models.vgg16(wights=wights,process=True)

# 保存模型
torch.save(obj=model.state_dict(),f='model_weights.pth')

# 加载模型
model = models.vgg16()
model.load_state_dict(state_dict=torch.load(f='model_wights.pth'))
model.eval()


# 创建和保存模型的程序通常是放在一起 加载模型的程序应该另设一个py文件

# 重点：state_dict：状态字典  这个每一个Module的子类 其作用是保存对应Module之中所有的parameter 和 buffer




'''
标准的保存模型的方式：模型的权重 模型的训练状态：epoch loss optimizer 
Steps

    Import all necessary libraries for loading our data
    Define and initialize the neural network
    Initialize the optimizer
    Save the general checkpoint
    Load the general checkpoint

'''
# initialize 初始化

#   Import all necessary libraries for loading our data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define and initialize the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

# Initialize the optimizer
optimizer = optim.SGD(params=net.parameters(),lr=0.01,momentum=0.9)

# Save the general checkpoint
# Additional information
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
    'epoch':EPOCH,
    'model_state_dict':net.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'loss':LOSS
},PATH
)





# Load the general checkpoint
# Remember to first initialize the model and optimizer, then load the dictionary locally.
model = Net()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#导入所有的参数
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()

'''
You must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
Failing to do this will yield inconsistent inference results.

If you wish to resuming training, call model.train() to ensure these layers are in training mode.

Congratulations! You have successfully saved and loaded a general checkpoint for inference and/or resuming training in PyTorch.
'''

