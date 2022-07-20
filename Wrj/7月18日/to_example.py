# _*_coding:utf-8_*_
import torch
import torch.nn as nn
# 创建一个测试网络
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.linear1 = torch.nn.Linear(2,3)
        self.linear2 = torch.nn.Linear(3,4)
        self.batchnorm = torch.nn.BatchNorm2d(4)

test_module = Test()

# _modules
print(test_module._modules)
'''
OrderedDict([('linear1', Linear(in_features=2, out_features=3, bias=True)),
 ('linear2', Linear(in_features=3, out_features=4, bias=True)), ('batchnorm', BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))])
'''
# 由字典构成 Key为自己设定的名字 Value为自己设定的Module

# 定位到linear1函数 使用字典的方式
print(test_module._modules['linear1'])
# Linear(in_features=2, out_features=3, bias=True)

# 使用Linear的variables 变量 weight bias
print(test_module._modules['linear1'].weight)
'''
Parameter containing:
tensor([[ 0.4040, -0.0632],
        [ 0.4741,  0.0847],
        [ 0.3573,  0.0244]], requires_grad=True)
'''

# dtype查看数据类型
print(test_module._modules['linear1'].weight.dtype)
# torch.float32

# 使用to函数转换精度
print(test_module._modules['linear1'].weight.to(torch.double).dtype)
# torch.float64 转换成功

# 再次使用to函数转换回来
print(test_module._modules['linear1'].weight.to(torch.float).dtype)
# torch.float32

# 使用to函数将模型转换到gpu上 并同时实现精度变换
gpu = torch.device('cuda:0')
test_module._modules['linear1'].to(device=gpu,dtype=torch.half,non_blocking=True)
# print(test_module._modules['linear1'].device)
'''
Traceback (most recent call last):
  File "D:\pytorch code\deep_thoughts\7_torch nn module\to_example.py", line 50, in <module>
    print(test_module._modules['linear1'].device)
  File "E:\anaconda\envs\py1_16\lib\site-packages\torch\nn\modules\module.py", line 1207, in __getattr__
    raise AttributeError("'{}' object has no attribute '{}'".format(
AttributeError: 'Linear' object has no attribute 'device'
'''
print(test_module._modules['linear1'].weight.device,'\n',test_module._modules['linear1'].weight.dtype)
# cuda:0
#  torch.float16

# 再次转换到cpu上
cpu = torch.device('cpu')
test_module._modules['linear1'].to(cpu)
print(test_module._modules['linear1'].weight.device)
# cpu

# getattr魔法方法
    # _modules
    # _parameters
    # _buffers

print(f"test_module._modules:{test_module._modules}")
print(f"test_module._parameters:{test_module._parameters}")
print(f"test_module._buffers:{test_module._buffers}")

'''
test_module._modules:OrderedDict([('linear1', Linear(in_features=2, out_features=3, bias=True)), ('linear2', Linear(in_features=3, out_features=4, bias=True)), ('batchnorm', BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))])
test_module._parameters:OrderedDict()
test_module._buffers:OrderedDict()
'''
# 仅在Test()的子模块儿中进行遍历 不去更下一层进行遍历 所以返回是空的字典

print(test_module._modules['linear1']._parameters)
'''
OrderedDict([('weight', Parameter containing:
tensor([[ 0.5088,  0.7002],
        [ 0.2295, -0.0842], 
        [-0.1604,  0.0043]], dtype=torch.float16, requires_grad=True)), ('bias', Parameter containing:
tensor([-0.1665, -0.6836, -0.2002], dtype=torch.float16, requires_grad=True))])
'''
# 这样定位到之后再进行检索就可以了

# 所以要记住的是getattr这个魔法方法只会向下一层进行查询


# state_dict()
print(test_module.state_dict())
'''
OrderedDict([('linear1.weight', tensor([[ 0.4380,  0.1281],
        [-0.5967,  0.2642],
        [ 0.1620,  0.4932]], dtype=torch.float16)), ('linear1.bias', tensor([-0.0457,  0.4065, -0.4844], dtype=torch.float16)), ('linear2.weight', tensor([[-0.1451, -0.4641,  0.3215],
        [-0.4890,  0.1731,  0.3324],
        [-0.5628,  0.4249,  0.1527],
        [-0.3408, -0.1856,  0.0129]])), ('linear2.bias', tensor([-0.2101,  0.5644,  0.1853, -0.5167])), ('batchnorm.weight', tensor([1., 1., 1., 1.])), ('batchnorm.bias', tensor([0., 0., 0., 0.])), ('batchnorm.running_mean', tensor([0., 0., 0., 0.])), ('batchnorm.running_var', tensor([1., 1., 1., 1.])), ('batchnorm.num_batches_tracked', tensor(0))])
'''
# 有了这个就可以使用字典的功能 找出对应的数据 然后利用其他的语言进行运算之类的 就十分的方便了 可拓展性大大增加
# print(test_module['linear1.weight'])
'''
Traceback (most recent call last):
  File "D:\pytorch code\deep_thoughts\7_torch nn module\to_example.py", line 109, in <module>
    print(test_module['linear1.weight'])
TypeError: 'Test' object is not subscriptable
'''

print(test_module.state_dict()['linear1.weight'])
'''
tensor([[-0.4309,  0.0344],
        [ 0.6934, -0.3245],
        [ 0.0434,  0.0063]], dtype=torch.float16)
'''

# 这种方法就相当于之上的那种，先使用._modules定位 然后使用._parameters ._buffers 进行检索 的方式 的更加简单的快捷的版本
# 这种方法的实现 是将上边那种方法整合起来，形成一种自动化的形式，并且更加规范化了，适用于更多的情形

# ('batchnorm.weight', tensor([1., 1., 1., 1.])), ('batchnorm.bias', tensor([0., 0., 0., 0.])),
# ('batchnorm.running_mean', tensor([0., 0., 0., 0.])), ('batchnorm.running_var', tensor([1., 1., 1., 1.])),
# ('batchnorm.num_batches_tracked', tensor(0))

# batch_norm会有上边的五个算子 数据
# weight bias 一维batch norm需要做一个 w*x+b 的操作
# running_mean running_bais 是运算过程中的统计量
# num_batches_tracked 是运算过程中的对第几个batch的一个计算

# _xyz xyz() named_xyz() 的区别
# _xyz 只向下检索一层 返回的是一个字典 带有名字 本身
# xyz() 全部遍历 返回的是一个迭代器 需要用for循环进行拆解 仅有值 无名字
# named_xyz() 全部遍历 返回的是一个迭代器 需要用for循环进行拆解 有名字 有值

# _modules, modules(), named_modules()
print(test_module._modules) # 仅有下一层的对象
'''
OrderedDict([('linear1', Linear(in_features=2, out_features=3, bias=True)), ('linear2', Linear(in_features=3, out_features=4, bias=True)), ('batchnorm', BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))])
'''
print(test_module.modules())
'''
<generator object Module.modules at 0x0000026DFFDB02E0>
'''
for modules in test_module.modules(): # 这一层的对象 和 下一层的对象
    print(modules)
'''
Test(
  (linear1): Linear(in_features=2, out_features=3, bias=True)
  (linear2): Linear(in_features=3, out_features=4, bias=True)
  (batchnorm): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
Linear(in_features=2, out_features=3, bias=True)
Linear(in_features=3, out_features=4, bias=True)
BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
'''
print(test_module.named_modules())
'''
<generator object Module.named_modules at 0x0000026DFFDB02E0>
'''
for named_modules in test_module.named_modules():  # 这一层的对象 和 下一层的对象
    print(named_modules)
'''
('', Test(
  (linear1): Linear(in_features=2, out_features=3, bias=True)
  (linear2): Linear(in_features=3, out_features=4, bias=True)
  (batchnorm): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
))
('linear1', Linear(in_features=2, out_features=3, bias=True))
('linear2', Linear(in_features=3, out_features=4, bias=True))
('batchnorm', BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
'''

# children(), named_children()
print(test_module.children())
print('\n',11111)
'''
<generator object Module.children at 0x00000214A4C5D890>
'''
for p in test_module.children():
    print(p)
'''
Linear(in_features=2, out_features=3, bias=True)
Linear(in_features=3, out_features=4, bias=True)
BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
<generator object Module.named_children at 0x0000022334BEDAC0>
'''

print(test_module.named_children())
print('\n',22222)
'''
<generator object Module.named_children at 0x00000214A4C5D890>
'''
for p1 in test_module.named_children():
    print(p1)
'''
('linear1', Linear(in_features=2, out_features=3, bias=True))
('linear2', Linear(in_features=3, out_features=4, bias=True))
('batchnorm', BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
OrderedDict()
'''



# _parameters parameters() named_parameters()
print(test_module._parameters) # 仅检索下一层 检索不到
print('\n')
'''
OrderedDict()
'''

for parameters in test_module.parameters(): # 检索下一层 再下一层 可以检索的到
    print(parameters)
print('\n')
'''
Parameter containing:
tensor([[-0.0427, -0.3650],
        [-0.2013,  0.6567],
        [-0.2045, -0.3450]], dtype=torch.float16, requires_grad=True)
Parameter containing:
tensor([-0.7026,  0.2214, -0.6890], dtype=torch.float16, requires_grad=True)
Parameter containing:
tensor([[-0.0876,  0.1539, -0.4298],
        [-0.1438,  0.5769,  0.3897],
        [ 0.4653,  0.4236,  0.4190],
        [-0.2059, -0.5718, -0.5219]], requires_grad=True)
Parameter containing:
tensor([ 0.1093,  0.2364, -0.1805, -0.1084], requires_grad=True)
Parameter containing:
tensor([1., 1., 1., 1.], requires_grad=True)
Parameter containing:
tensor([0., 0., 0., 0.], requires_grad=True)
'''

for named_parameters in test_module.named_parameters(): # 检索下一层 再下一层 可以检索的到
    print(named_parameters)
print('\n')
'''
('linear1.weight', Parameter containing:
tensor([[-0.0427, -0.3650],
        [-0.2013,  0.6567],
        [-0.2045, -0.3450]], dtype=torch.float16, requires_grad=True))
('linear1.bias', Parameter containing:
tensor([-0.7026,  0.2214, -0.6890], dtype=torch.float16, requires_grad=True))
('linear2.weight', Parameter containing:
tensor([[-0.0876,  0.1539, -0.4298],
        [-0.1438,  0.5769,  0.3897],
        [ 0.4653,  0.4236,  0.4190],
        [-0.2059, -0.5718, -0.5219]], requires_grad=True))
('linear2.bias', Parameter containing:
tensor([ 0.1093,  0.2364, -0.1805, -0.1084], requires_grad=True))
('batchnorm.weight', Parameter containing:
tensor([1., 1., 1., 1.], requires_grad=True))
('batchnorm.bias', Parameter containing:
tensor([0., 0., 0., 0.], requires_grad=True))
'''

# _buffers, buffers(), named_buffers()
print(test_module._buffers) # 仅向下检索一层 检索不到
'''
OrderedDict()
'''
for buffers in test_module.buffers(): # 向下检索一层 再向下检索一层
    print(buffers)
'''
tensor([0., 0., 0., 0.])
tensor([1., 1., 1., 1.])
tensor(0)
'''
for named_buffers in test_module.named_buffers():
    print(named_buffers)
'''
('batchnorm.running_mean', tensor([0., 0., 0., 0.]))
('batchnorm.running_var', tensor([1., 1., 1., 1.]))
('batchnorm.num_batches_tracked', tensor(0))
'''

### 请你总结出 所有的可以实现获取test_module中所有module的不同方法 不要求返回的对象的类型 可以是一整个字典 也可以是一个个的module
# 2022-07-19


# 魔法方法 __repr__
print(str(test_module))
'''
Test(
  (linear1): Linear(in_features=2, out_features=3, bias=True)
  (linear2): Linear(in_features=3, out_features=4, bias=True)
  (batchnorm): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
'''
print(repr(test_module))
'''
Test(
  (linear1): Linear(in_features=2, out_features=3, bias=True)
  (linear2): Linear(in_features=3, out_features=4, bias=True)
  (batchnorm): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
'''

# 魔法方法 __dir__
print(dir(test_module))
'''
['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__',
 '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', 
 '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__',
  '__weakref__', '_apply', '_backward_hooks', '_buffers', '_call_impl', '_forward_hooks', '_forward_pre_hooks', '_get_backward_hooks', '_get_name',
   '_is_full_backward_hook', '_load_from_state_dict', '_load_state_dict_pre_hooks', '_maybe_warn_non_full_backward_hook', '_modules', '_named_members', 
   '_non_persistent_buffers_set', '_parameters', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel',
    '_save_to_state_dict', '_slow_forward', '_state_dict_hooks', '_version', 'add_module', 'apply', 'batchnorm', 'bfloat16', 'buffers', 'children', 'cpu', 
    'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'float', 'forward', 'half', 'linear1', 'linear2', 'load_state_dict', 'modules', 'named_buffers', 
    'named_children', 'named_modules', 'named_parameters', 'parameters', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 
    'register_full_backward_hook', 'register_parameter', 'requires_grad_', 'share_memory', 'state_dict', 'to', 'train', 'training', 'type', 'xpu', 'zero_grad']
'''


