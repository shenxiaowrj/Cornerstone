# _*_coding:utf-8_*_
'''
class torch.nn.Module

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module): # 继承父类
    def __init__(self):
        super().__init__() #声明
        # nn.Module也可以作为父类中的子类
        self.conv1 = nn.Conv2d(1,20,kernel_size=5,)  # 这里是2d 所以kernel_size 自动就设置为5*5
        self.conv2 = nn.Conv2d(20,20,kernel_size=5,)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

# Variables:
    # training

# add_module(name,module)
    # 另外增添一个module 模块儿

# apply(fn)
    # 对Model里边的某些module进行操作 比如对linear进行归一化

@torch.no_grad()
def init_weight(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)
net = nn.Sequential(
    nn.Linear(2,2),
    nn.Linear(2,2)
)
net.apply(init_weight)
'''
#print(m)

Parameter containing:
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
Parameter containing:
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)

print(m)   多出一个Linear 多出一个Sequential

Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
'''

# bfloat16()
    # Casts all floating point parameters and buffers to bfloat16 datatype.
    # buffers槽变量 缓冲区

# buffers(recurse=True)
    # Returns an iterator over module buffers.
model = Model()
for buf in model.buffers():
    print(type(buf),buf.size())

# children()
    # Returns an iterator over immediate children modules.

# cpu()
    # Moves all model parameters and buffers to the CPU.

# cuda(device=None)
    # Moves all model parameters and buffers to the GPU.

# double()
    # Casts all floating point parameters and buffers to double datatype.

# eval()
    # Sets the module in evaluation mode.
    # This has any effect only on certain modules.
    # See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g. Dropout, BatchNorm, etc.

# float()
    # Casts all floating point parameters and buffers to float datatype.

# forward(*input)
    # Defines the computation performed at every call. 前向计算
    # Should be overridden by all subclasses.

# get_parameter(target)
    # Returns the parameter given by target if it exists, otherwise throws an error.

# load_state_dict(state_dict, strict=True)
    #Copies parameters and buffers from state_dict into this module and its descendants.
    # If strict is True, then the keys of state_dict must exactly match the keys returned by this module’s state_dict() function.

# modules()
    #Returns an iterator over all modules in the network.

l = nn.Linear(2,2)
net = nn.Sequential(
    l,
    l
)
for idx,m in enumerate(net.modules()):
    print(idx,'->',m)
'''
0 -> Sequential(
  (0): Linear(in_features=2, out_features=2, bias=True)
  (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)
为什么会有两个索引？index

'''

# parameters(recurse=True)
    # Returns an iterator over module parameters.
    # This is typically passed to an optimizer.
        # recurse (bool) – if True, then yields parameters of this module and all submodules.
        # Otherwise, yields only parameters that are direct members of this module.

for param in model.parameters():
    print(type(param),param.size())

# 若是Linear 那么就有weight bias


# requires_grad_(requires_grad=True)
    # Change if autograd should record operations on parameters in this module.
    # This method sets the parameters’ requires_grad attributes in-place.
#requires_grad (bool) – whether autograd should record operations on parameters in this module. Default: True.
# 如果需要进行训练，那么就需要进行反向传播，如果需要进行反向传播，那么就需要autograd记录parameters的相关操作，也就是requires_grad=True


# state_dict(*args, destination=None, prefix='', keep_vars=False)
    # Returns a dictionary containing a whole state of the module.
    # Both parameters and persistent buffers (e.g. running averages) are included.
    # Keys are corresponding parameter and buffer names. Parameters and buffers set to None are not included.

# 以字典的形式 保存上module的全部的状态  Keys是parameter 和 buffer的名称
# Values就是当前parameter 和 buffer的状态

