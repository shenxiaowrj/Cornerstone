from collections import OrderedDict, namedtuple
import itertools
import warnings
import functools

import torch
from ..parameter import Parameter
import torch.utils.hooks as hooks

from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from ...utils.hooks import RemovableHandle

_grad_t = Union[Tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Module')

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


r"""This tracks hooks common to all modules that are executed before/after
calling forward and backward. This is global state used for debugging/profiling
purposes"""
_global_backward_hooks: Dict[int, Callable] = OrderedDict()
_global_is_full_backward_hook: Optional[bool] = None
_global_forward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks: Dict[int, Callable] = OrderedDict()

_EXTRA_STATE_KEY_SUFFIX = '_extra_state'


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a forward pre-hook common to all modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a global forward hook for all the modules

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the output. It can modify the input inplace but
    it will not have effect on forward since this is called after
    :func:`forward` is called.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    handle = hooks.RemovableHandle(_global_forward_hooks)
    _global_forward_hooks[handle.id] = hook
    return handle

def register_module_backward_hook(
    hook: Callable[['Module', _grad_t, _grad_t], Union[None, Tensor]]
) -> RemovableHandle:
    r"""Registers a backward hook common to all the modules.

    This function is deprecated in favor of
    :func:`torch.nn.modules.module.register_module_full_backward_hook`
    and the behavior of this function will change in future versions.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is True:
        raise RuntimeError("Cannot use both regular backward hooks and full backward hooks as a "
                           "global Module hook. Please use only one of them.")

    _global_is_full_backward_hook = False

    handle = hooks.RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle

def register_module_full_backward_hook(
    hook: Callable[['Module', _grad_t, _grad_t], Union[None, Tensor]]
) -> RemovableHandle:
    r"""Registers a backward hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time the gradients with respect to module
    inputs are computed. The hook should have the following signature::

        hook(module, grad_input, grad_output) -> Tensor or None

    The :attr:`grad_input` and :attr:`grad_output` are tuples. The hook should
    not modify its arguments, but it can optionally return a new gradient with
    respect to the input that will be used in place of :attr:`grad_input` in
    subsequent computations. :attr:`grad_input` will only correspond to the inputs given
    as positional arguments and all kwarg arguments will not appear in the hook. Entries
    in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
    arguments.

    For technical reasons, when this hook is applied to a Module, its forward function will
    receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
    of each Tensor returned by the Module's forward function.

    Global hooks are called before hooks registered with `register_backward_hook`

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is False:
        raise RuntimeError("Cannot use both regular backward hooks and full backward hooks as a "
                           "global Module hook. Please use only one of them.")

    _global_is_full_backward_hook = True

    handle = hooks.RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
def _forward_unimplemented(self, *input: Any) -> None:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

# 1 开始
class Module:
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.

    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """

    dump_patches: bool = False

    _version: int = 1
    r"""This allows better BC support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""

    training: bool
    _is_full_backward_hook: Optional[bool]

    # 定义一些成员变量 一般都是字典
    def __init__(self) -> None:
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")

        # 默认情况下 state = train
        # 也就是batchnorm dropout是开启的
        self.training = True
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        # parameters buffers 是字典的形式
        self._non_persistent_buffers_set: Set[str] = set()

        # 钩子函数 高级操作
        self._backward_hooks: Dict[int, Callable] = OrderedDict()
        self._is_full_backward_hook = None
        self._forward_hooks: Dict[int, Callable] = OrderedDict()
        self._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._state_dict_hooks: Dict[int, Callable] = OrderedDict()
        self._load_state_dict_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._load_state_dict_post_hooks: Dict[int, Callable] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()

    forward: Callable[..., Any] = _forward_unimplemented

    # 作用：向module之中添加buffer
    # register 注册
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""Adds a buffer to the module.


        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``  #对每个批次的均值做的滑动平均 统计的累积量
        is not a parameter, but is part of the module's state.# 状态 Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        # persistent 执着的 持久的
        # 默认persistent = True 也就是代表是module的一部分
        # persitent = False 的时候，使用state_dict时，不会将其和module一块儿存储在硬盘之中

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            如果你要自己去实现一个module 而不是使用pytorch自带的module 比如说构建一个新的batchnorm的算法 那么就要用到buffer的相关操作
            class BacthNorm(nn.Module):
                def __init__(self):
                self.register_buffer('running_mean',torch.zeros(num_features))
                self.register_buffer('running_variance',torch.ones(num_features))

            >>> self.register_buffer('running_mean', torch.zeros(num_features))
                                     buffer-name     buffer-tensor       buffer-persistent default = True
        """
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(torch.typename(tensor), name))
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    # 更加常用的一个方法 parameter比buffer多了很多
    # register 注册
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name

            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.

            # param是parameter类型的 parameter继承自tensor
            class torch.nn.parameter.Parameter(data=None, requires_grad=True)

            A kind of Tensor that is to be considered 考虑 认为 a module parameter.

            Parameters are Tensor subclasses 子类,
            that have a very special property 属性 when used with Module s -
            when they’re assigned as 被分配为 Module attributes they are automatically added to the list of its parameters,  # 若是你给Module传入的是一个parameter类型的数据 它就会被自动添加到Module的列表之中
            and will appear e.g. in parameters() iterator.

            Assigning a Tensor doesn’t have such effect.  # 若是你要自己添加 就会挺麻烦的
             This is because one might want to cache 缓存 some temporary 时间 state,
             like last hidden state of the RNN, in the model.
              If there was no such class as Parameter,
             these temporaries would get registered too.

            parameter类型 也就是 tensor类型的变种版本 多了一点功能
            这个功能是，当它被传入到Module之中的时候，它可以自动的被添加到Module的列表之中

        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            self._parameters[name] = param
            # 最后会被存放parameters字典之中  Keys=name Values=param

    # e.g.:https://stackoverflow.com/questions/63047762/correct-way-to-register-a-parameter-for-model-in-pytorch
    '''
    I tried to define a simple model in Pytorch. The model computes negative log prob for a gaussian distribution: 
                                                # 自定义一个新的模型 该模型计算 负对数似然估计 对于一个高斯分布来说
    
    import torch
    import torch.nn as nn
    
    class GaussianModel(nn.Module):
    
        def __init__(self):
            super(GaussianModel, self).__init__()
            
            # 注册一个parameter
            self.register_parameter('mean', nn.Parameter(torch.zeros(1),requires_grad=True))
                                    name     tensor
                                    
            self.pdf = torch.distributions.Normal(self.state_dict()['mean'],  向高斯分布Normal 这个Module之中传入该参数
                                                  torch.tensor([1.0]))
                            
        def forward(self, x):
            return -self.pdf.log_prob(x)
    
    model = GaussianModel()
    
    '''

    # 添加新的module
    def add_module(self, name: str, module: Optional['Module']) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                torch.typename(module)))
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("module name should be a string. Got {}".format(
                torch.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module
        # 添加到modules这个字典之中 Key=name Values=module

    # 注册一个module
    def register_module(self, name: str, module: Optional['Module']) -> None:
        r"""Alias for :func:`add_module`."""
        # Alias 别名
        self.add_module(name, module)

    # 得到子module
    def get_submodule(self, target: str) -> "Module":
        """
        Returns the submodule given by ``target`` if it exists,
        otherwise throws an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )
        # 嵌套结构
        A
            net_b
                net_c
                    conv
                linear

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded 受限 by the degree 程度
        of module nesting 嵌套 in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive 传递 modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify 指定 a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: torch.nn.Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                     "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")

        return mod

    # 得到参数 根据一个字符串
    # 注意这个target一定要写清楚 写详细
    def get_parameter(self, target: str) -> "Parameter":
        """
        Returns the parameter given by ``target`` if it exists,
        otherwise throws an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        # 首先对target 解析出来module_path，param_name
        module_path, _, param_name = target.rpartition(".")
        '''
        rpartition函数 python之中关于字符串的函数
        (".") 返回
                .之前的字符串
                .本身
                .之后的字符串
        返回的值有你不想要的 就是用_进行拆包
        '''

        # 通过get_submodule方法找到对应的Module
        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name): # 从mod 也就是给定的Module之中 是否具有 param.name 这种属性
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + param_name + "`")
        '''
        hasattr 从一个对象中 找是否有一个属性
        mod 的定义方法：
        mod: 属于哪个类型 = self.  传入那个类型的实例
        '''

        param: torch.nn.Parameter = getattr(mod, param_name)
        '''
        getattr 给定一个Module 给定一个parameter的名称 取得该parameter
        
        '''

        if not isinstance(param, torch.nn.Parameter): # 利用isinstance方法再去审核一下 取出来的param是否是torch.nn.Parameter类型
            raise AttributeError("`" + param_name + "` is not an "
                                 "nn.Parameter")
        '''
        isinstance 审核实例
        Args:
        一个实例
        一个类型
        
        作用:判断该实例 是否是这个类型
        '''
        # 检查成功，返回parameter
        return param

    # 给定target目标地址 获取对应的buffer   和上面类似的逻辑
    def get_buffer(self, target: str) -> "Tensor":
        """
        Returns the buffer given by ``target`` if it exists,
        otherwise throws an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        # 通过rpartition进行解析
        module_path, _, buffer_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        # 通过hasattr验证buffer_name是否是对应Module下的一个属性
        if not hasattr(mod, buffer_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + buffer_name + "`")

        # 通过getattr得到对应的buffer
        buffer: torch.Tensor = getattr(mod, buffer_name)

        # 再次验证 buffer_name 是否在 mod._buffers 这个列表之中
        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer


    def get_extra_state(self) -> Any:
        """
        Returns any extra state to include in the module's state_dict.
        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be pickleable to ensure working serialization
        of the state_dict. We only provide provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug.")

    def set_extra_state(self, state: Any):
        """
        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug.")

    # 重要的函数 基础方法
    def _apply(self, fn):
        # 1 对每一个子模块进行fn操作
        for module in self.children():
            module._apply(fn)


        def compute_should_use_set_data(tensor, tensor_applied):
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False
        # 2 对parameters里的param 在torch.no_grad()的模式下进行fn操作
        for key, param in self._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
            should_use_set_data = compute_should_use_set_data(param, param_applied)
            if should_use_set_data:
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, Parameter)
                assert param.is_leaf
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param.grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param.grad)
                should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                if should_use_set_data:
                    out_param.grad.data = grad_applied
                else:
                    assert param.grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

        # 3 对于buffer里边的buf进行fn操作
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[ 1.,  1.],
                    [ 1.,  1.]])
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[ 1.,  1.],
                    [ 1.,  1.]])
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
        """
        # 先对子模块进行遍历操作
        for module in self.children():
            module.apply(fn)
        # 再对module本身进行操作
        fn(self)
        return self

    # apply函数的应用
    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cuda(device))
        # 对传入的T全部都调用_apply 进行.cuda(device)函数操作
        # .cuda(device)以tensor为尺度进行操作的 也就是将所有的由tensor组成的数据 都在cuda memory上做一个拷贝

    # 同上
    def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Moves all model parameters and buffers to the IPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on IPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.ipu(device))

    # 同上
    def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Moves all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.xpu(device))

    # 同上
    def cpu(self: T) -> T:
        r"""Moves all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    # 也是_apply的应用  转化为特定的type
    def type(self: T, dst_type: Union[dtype, str]) -> T:
        r"""Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.type(dst_type))
        # .type(dst_type)
        # 使用.type()函数进行类型转化

    # 同上 将数据类型换成float
    def float(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    # 同上 将数据类型换成double
    def double(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    # 将数据类型换成half
    def half(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    # 将数据类型换成bfloat16
    def bfloat16(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)

    # 将parameters buffers移动到指定的设备上 但是不拷贝相应内容的内存 这是啥意思嘞？ 只拷贝空的tensor还是什么？
    # 暂时不知道用处
    def to_empty(self: T, *, device: Union[str, device]) -> T:
        r"""Moves the parameters and buffers to the specified device without copying storage.

        Args:
            device (:class:`torch.device`): The desired device of the parameters
                and buffers in this module.

        Returns:
            Module: self
        """
        return self._apply(lambda t: torch.empty_like(t, device=device))

    @overload
    def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ...,
           non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        r"""Moves and/or casts the parameters and buffers.

        This can be called as
        多种使用call的方式(使用方式)

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self

        Examples::

            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            # non_blocking 不会阻塞数据 以实现并行处理
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

            >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.3741+0.j,  0.2382+0.j],
                    [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
            >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
            tensor([[0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

        """

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError('nn.Module.to only accepts floating point or complex '
                                'dtypes, but got desired dtype={}'.format(dtype))
            if dtype.is_complex:
                warnings.warn(
                    "Complex modules are a new feature under active development whose design may change, "
                    "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                    "if a complex module does not work as expected.")

        def convert(t):
            if convert_to_format is not None and t.dim() in (4, 5):
                return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                            non_blocking, memory_format=convert_to_format)
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

        return self._apply(convert)
        # 左边创建了一个案例文件：to_example.py


    def register_backward_hook(
        self, hook: Callable[['Module', _grad_t, _grad_t], Union[None, Tensor]]
    ) -> RemovableHandle:
        r"""Registers a backward hook on the module.

        This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
        the behavior of this function will change in future versions.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
        if self._is_full_backward_hook is True:
            raise RuntimeError("Cannot use both regular backward hooks and full backward hooks on a "
                               "single Module. Please use only one of them.")

        self._is_full_backward_hook = False

        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_full_backward_hook(
        self, hook: Callable[['Module', _grad_t, _grad_t], Union[None, Tensor]]
    ) -> RemovableHandle:
        r"""Registers a backward hook on the module.

        The hook will be called every time the gradients with respect to module
        inputs are computed. The hook should have the following signature::

            hook(module, grad_input, grad_output) -> tuple(Tensor) or None

        The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
        with respect to the inputs and outputs respectively. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the input that will be used in place of :attr:`grad_input` in
        subsequent computations. :attr:`grad_input` will only correspond to the inputs given
        as positional arguments and all kwarg arguments are ignored. Entries
        in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
        arguments.

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.

        .. warning ::
            Modifying inputs or outputs inplace is not allowed when using backward hooks and
            will raise an error.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
        if self._is_full_backward_hook is False:
            raise RuntimeError("Cannot use both regular backward hooks and full backward hooks on a "
                               "single Module. Please use only one of them.")

        self._is_full_backward_hook = True

        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def _get_backward_hooks(self):
        r"""Returns the backward hooks for use in the call function.
        It returns two lists, one with the full backward hooks and one with the non-full
        backward hooks.
        """
        full_backward_hooks: List[Callable] = []
        if (_global_is_full_backward_hook is True):
            full_backward_hooks += _global_backward_hooks.values()
        if (self._is_full_backward_hook is True):
            full_backward_hooks += self._backward_hooks.values()

        non_full_backward_hooks: List[Callable] = []
        if (_global_is_full_backward_hook is False):
            non_full_backward_hooks += _global_backward_hooks.values()
        if (self._is_full_backward_hook is False):
            non_full_backward_hooks += self._backward_hooks.values()

        return full_backward_hooks, non_full_backward_hooks

    def _maybe_warn_non_full_backward_hook(self, inputs, result, grad_fn):
        if not isinstance(result, torch.Tensor):
            if not (isinstance(result, tuple) and all([isinstance(r, torch.Tensor) for r in result])):
                warnings.warn("Using non-full backward hooks on a Module that does not return a "
                              "single Tensor or a tuple of Tensors is deprecated and will be removed "
                              "in future versions. This hook will be missing some of the grad_output. "
                              "Please use register_full_backward_hook to get the documented behavior.")
                return
        else:
            result = (result,)

        if not isinstance(inputs, torch.Tensor):
            if not (isinstance(inputs, tuple) and all([isinstance(i, torch.Tensor) for i in inputs])):
                warnings.warn("Using non-full backward hooks on a Module that does not take as input a "
                              "single Tensor or a tuple of Tensors is deprecated and will be removed "
                              "in future versions. This hook will be missing some of the grad_input. "
                              "Please use register_full_backward_hook to get the documented behavior.")
                return
        else:
            inputs = (inputs,)

        # At this point we are sure that inputs and result are tuple of Tensors
        out_grad_fn = {r.grad_fn for r in result if r.grad_fn is not None}
        if len(out_grad_fn) == 0 or (len(out_grad_fn) == 1 and grad_fn not in out_grad_fn):
            warnings.warn("Using a non-full backward hook when outputs are nested in python data structure "
                          "is deprecated and will be removed in future versions. This hook will be missing "
                          "some grad_output.")
        elif len(out_grad_fn) > 1:
            warnings.warn("Using a non-full backward hook when outputs are generated by different autograd Nodes "
                          "is deprecated and will be removed in future versions. This hook will be missing "
                          "some grad_output. Please use register_full_backward_hook to get the documented behavior.")
        else:
            # At this point the grad_ouput part of the hook will most likely be correct
            inputs_grad_fn = {i.grad_fn for i in inputs if i.grad_fn is not None}

            next_functions = {n[0] for n in grad_fn.next_functions}

            if inputs_grad_fn != next_functions:
                warnings.warn("Using a non-full backward hook when the forward contains multiple autograd Nodes "
                              "is deprecated and will be removed in future versions. This hook will be missing "
                              "some grad_input. Please use register_full_backward_hook to get the documented "
                              "behavior.")

    def register_forward_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.
        It should have the following signature::

            hook(module, input) -> None or modified input

        The input contains only the positional arguments given to the module.
        Keyword arguments won't be passed to the hooks and only to the ``forward``.
        The hook can modify the input. User can either return a tuple or a
        single modified value in the hook. We will wrap the value into a tuple
        if a single value is returned(unless that value is already a tuple).

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle

    def register_forward_hook(self, hook: Callable[..., None]) -> RemovableHandle:
        r"""Registers a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.
        It should have the following signature::

            hook(module, input, output) -> None or modified output

        The input contains only the positional arguments given to the module.
        Keyword arguments won't be passed to the hooks and only to the ``forward``.
        The hook can modify the output. It can modify the input inplace but
        it will not have effect on forward since this is called after
        :func:`forward` is called.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def _slow_forward(self, *input, **kwargs):
        tracing_state = torch._C._get_tracing_state()
        if not tracing_state or isinstance(self.forward, torch._C.ScriptMethod):
            return self.forward(*input, **kwargs)
        recording_scopes = torch.jit._trace._trace_module_map is not None
        if recording_scopes:
            # type ignore was added because at this point one knows that
            # torch.jit._trace._trace_module_map is not Optional and has type Dict[Any, Any]
            name = torch.jit._trace._trace_module_map[self] if self in torch.jit._trace._trace_module_map else None  # type: ignore[index, operator] # noqa: B950
            if name:
                tracing_state.push_scope(name)
            else:
                recording_scopes = False
        try:
            result = self.forward(*input, **kwargs)
        finally:
            if recording_scopes:
                tracing_state.pop_scope()
        return result

    def _call_impl(self, *input, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
            return forward_call(*input, **kwargs)
        # Do not call functions when jit is used
        full_backward_hooks, non_full_backward_hooks = [], []
        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()
        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook in (*_global_forward_pre_hooks.values(), *self._forward_pre_hooks.values()):
                result = hook(self, input)
                if result is not None:
                    if not isinstance(result, tuple):
                        result = (result,)
                    input = result

        bw_hook = None
        if full_backward_hooks:
            bw_hook = hooks.BackwardHook(self, full_backward_hooks)
            input = bw_hook.setup_input_hook(input)

        result = forward_call(*input, **kwargs)
        if _global_forward_hooks or self._forward_hooks:
            for hook in (*_global_forward_hooks.values(), *self._forward_hooks.values()):
                hook_result = hook(self, input, result)
                if hook_result is not None:
                    result = hook_result

        if bw_hook:
            result = bw_hook.setup_output_hook(result)

        # Handle the non-full backward hooks
        if non_full_backward_hooks:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
                self._maybe_warn_non_full_backward_hook(input, result, grad_fn)

        return result

    __call__ : Callable[..., Any] = _call_impl

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Support loading old checkpoints that don't have the following attrs:
        if '_forward_pre_hooks' not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if '_state_dict_hooks' not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if '_load_state_dict_pre_hooks' not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
        if '_load_state_dict_post_hooks' not in self.__dict__:
            self._load_state_dict_post_hooks = OrderedDict()
        if '_non_persistent_buffers_set' not in self.__dict__:
            self._non_persistent_buffers_set = set()
        if '_is_full_backward_hook' not in self.__dict__:
            self._is_full_backward_hook = None

# 魔法方法
    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(torch.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _register_state_dict_hook(self, hook):
        r"""These hooks will be called with arguments: `self`, `state_dict`,
        `prefix`, `local_metadata`, after the `state_dict` of `self` is set.
        Note that only parameters and buffers of `self` or its children are
        guaranteed to exist in `state_dict`. The hooks may modify `state_dict`
        inplace or return a new one.
        """
        handle = hooks.RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        # 对当前Module的子模块进行parameter buffer的遍历 遍历出来的对象存放在destination的的字典之中
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "get_extra_state", Module.get_extra_state) is not Module.get_extra_state:
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrederedDict` is created and returned.
    T_destination = TypeVar('T_destination', bound=Dict[str, Any])

    @overload
    def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
        ...

    # 所以，使用state_dict()可以来实现返回对象Module()所有的parameter buffer
    # TODO: Change `*args` to `*` and remove the copprespinding warning in docs when BC allows.
    # Also remove the logic for arg parsing together.
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """

        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        # 首先对当前模块的子模块儿遍历parameter buffer 存放在destination字典之中
        self._save_to_state_dict(destination, prefix, keep_vars)
        # 接着使用_modules获得当前模块儿所有的子模块儿
        # 对子模块儿下的模块儿进行遍历 获得其parameter buffer
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _register_load_state_dict_pre_hook(self, hook, with_module=False):
        r"""These hooks will be called with arguments: `state_dict`, `prefix`,
        `local_metadata`, `strict`, `missing_keys`, `unexpected_keys`,
        `error_msgs`, before loading `state_dict` into `self`. These arguments
        are exactly the same as those of `_load_from_state_dict`.

        If ``with_module`` is ``True``, then the first argument to the hook is
        an instance of the module.

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
            with_module (bool, optional): Whether or not to pass the module
                instance to the hook as the first parameter.
        """
        handle = hooks.RemovableHandle(self._load_state_dict_pre_hooks)
        if with_module:
            hook = functools.partial(hook, self)
        self._load_state_dict_pre_hooks[handle.id] = hook
        return handle

    def register_load_state_dict_post_hook(self, hook):
        r"""Registers a post hook to be run after module's ``load_state_dict``
        is called.

        It should have the following signature::
            hook(module, incompatible_keys) -> None

        The ``module`` argument is the current module that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearning out both missing and unexpected keys will avoid an error.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._load_state_dict_post_hooks)
        self._load_state_dict_post_hooks[handle.id] = hook
        return handle


    # 从一个state_dict之中得到相应的parameter buffer并将其赋值给对应的Module
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        # 对本地状态进行遍历 local_state 组成一个key
        for name, param in local_state.items():
            key = prefix + name
            # 查找key是否在state_dict之中 如果在就将key传入到state_dict之中得到对应的值 将值赋值给input_param
            if key in state_dict:
                input_param = state_dict[key]
                if not torch.overrides.is_tensor_like(input_param):
                    error_msgs.append('While copying the parameter named "{}", '
                                      'expected torch.Tensor or Tensor-like object from checkpoint but '
                                      'received {}'
                                      .format(key, type(input_param)))
                    continue

                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.
                is_param_lazy = torch.nn.parameter.is_lazy(param)
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if not is_param_lazy and input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue
                try:
                    with torch.no_grad():# 在不计算梯度的情况下 将input_param拷贝给param
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occurred : {}.'
                                      .format(key, param.size(), input_param.size(), ex.args))
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)
        # 包含了一些如果是strict模式下一些特殊的操作 .append() 函数

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # 对module的子模块儿进行遍历使用_load_from_state_dict 导入对应的parameter buffer
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            # 对module的子模块儿中的模块儿进行遍历 导入对应的parameter buffer
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

            # Note that the hook can modify missing_keys and unexpected_keys.
            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                    "expected to return new values, if incompatible_keys need to be modified,"
                    "it should be done inplace."
                )

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    # 基础函数 通用函数 只要是由层级的关系的 都可以进行调用查询 因为get_members_fn是自定义的
    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        # 创建备忘录
        memo = set()
        # 规范化module
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules: # 遍历取出 module的名字 和 module
            members = get_members_fn(module)  # 存放在members之中,进一步整合 规范化
            for k, v in members: #遍历members
                if v is None or v in memo:
                    continue   # 这个continue是什么逻辑啊？
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v # 返回一个迭代器 名字还有值

    # _named_members应用1
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse) # 调用_named_members函数 自定义查询函数 get_members_fn 是lambda函数 module._parameters.item()转为具体的对象
        for elem in gen:
            yield elem # 返回一个迭代器

    # _named_members应用2
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Returns an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        r"""Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            (string, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse) # 使用_named_mebers函数 get_memebers_fn = module._buffers.items()
        for elem in gen:  # 返回一个迭代器 iterator 用for循环进行拆解
            yield elem


    # 类似的代码逻辑
    def children(self) -> Iterator['Module']:
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Example::

            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set() # 创建一个备忘录
        for name, module in self._modules.items(): # 使用_modules.items()进行检索 拆包 类似这个意思
            if module is not None and module not in memo:
                memo.add(module) # 添加到备忘录之中
                yield name, module # 返回一个迭代器

    # _named_members应用3
    def modules(self) -> Iterator['Module']:
        r"""Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
                    print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:  # 如果没有备忘录就创建一个备忘录
            memo = set()
        if self not in memo:  # 如果self对象不再备忘录中 就添加进去
            if remove_duplicate:
                memo.add(self)
            yield prefix, self # 返回一个迭代器 名字 和 自身
            for name, module in self._modules.items(): # 对子模块儿的模块进行遍历
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m # 返回子模块儿的模块

    # 将程序设置为训练的模式
    def train(self: T, mode: bool = True) -> T:
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool): # 首先判断传入的值是不是bool值
            raise ValueError("training mode is expected to be boolean")
        self.training = mode # 若是bool值，则将其赋值给self.training
        # 这个是个关键
        # network opetimizer基本上都是基于nn.Module类的 所以，比如batchnorm函数，只要继承了nn.Module类，那么就可以调用self.training 对这两个状态的不同功能分别进行设置
        for module in self.children(): # 对所有的module进行遍历
            module.train(mode) # 对检索到的所有的module使用.train(mode)来调整状态
        return self

    # train()函数的反过程，这也就是直接写 model.eval() batchnorm,dropout 不会生效的原因
    def eval(self: T) -> T:
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)

    # 是否需要记录自动微分求得的偏导数值
    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        r"""Change if autograd should record operations on parameters in this
        module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters(): #使用parameters函数，检索出所有的parameter buffer
            p.requires_grad_(requires_grad)  #对这些tensor使用requires_grad_函数 requires_grad这是一个bool值
        return self  # 对每一个parameter buffer调整完之后 返回修改后的self

    # pytorch会自动的累积求得的偏导数值，所以在每一个epoch的训练时，要使用optimizer.zero_grad(), 将累积的梯度置零，这样训练才有效果
    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, '_is_replica', False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead.")

        for p in self.parameters():# 使用parameters()函数，检索出所有的parameter buffer
            if p.grad is not None:
                if set_to_none: # 不是设置成0，直接设置为无
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None: #对函数里边累积的偏导进行置零
                        p.grad.detach_() # 返回一个新的tensor 与现有的计算图分离
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def share_memory(self: T) -> T:
        r"""See :meth:`torch.Tensor.share_memory_`"""
        return self._apply(lambda t: t.share_memory_())

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    # 魔法方法
    # 对实例化的类可以直接使用str方法 返回representation
    # 作用：对类进行最全面的表示，返回所有的module 如果有附加的module 比如resister的module 也会进行返回 同时，其返回的格式是十分规范和好看的
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items(): # 首先使用_modules向下检索一层子模块儿  key:代表对应模块儿的名字
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)  # 获取子模块儿的内容
            child_lines.append('(' + key + '): ' + mod_str) # 定义子模块儿 形式 (名字): 具体子模块儿的内容
        lines = extra_lines + child_lines  # 将子模块儿和extra_lines进行拼接

        main_str = self._get_name() + '(' # 首先是类的name 然后加上左括号
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]  # 加上lines的信息
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n' # 写入子模块儿 注意这个'\n'的换行符

        main_str += ')' # 加上右括号
        return main_str

    # 魔法方法
    # 目录：返回这个实例化后的类的所有的信息
    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()

        # replicas do not have parameters themselves, the replicas reference the original
        # module.
        replica._parameters = OrderedDict()
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True  # type: ignore[assignment]

        return replica
