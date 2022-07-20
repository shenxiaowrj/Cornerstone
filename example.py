# _*_coding:utf-8_*_
import torch
import torch.nn as nn
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # 同过for循环定义
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        # 通过for循环展开
        for i, l in enumerate(self.linears):  # enumerate枚举函数 i为index 0,1,2,3... l是什么呢？  ###
            x = self.linears[i // 2](x) + l(x)
        return x
Test = MyModule()
print(Test)

'''
MyModule(
  (linears): ModuleList(
    (0): Linear(in_features=10, out_features=10, bias=True)
    (1): Linear(in_features=10, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=10, bias=True)
    (3): Linear(in_features=10, out_features=10, bias=True)
    (4): Linear(in_features=10, out_features=10, bias=True)
    (5): Linear(in_features=10, out_features=10, bias=True)
    (6): Linear(in_features=10, out_features=10, bias=True)
    (7): Linear(in_features=10, out_features=10, bias=True)
    (8): Linear(in_features=10, out_features=10, bias=True)
    (9): Linear(in_features=10, out_features=10, bias=True)
  )
)
'''
