# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/core.py
#
# Zennit is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Zennit is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
'''Core functions and classes'''
import functools
import weakref
from contextlib import contextmanager

import torch


class Stabilizer:
    '''Class to create a stabilizer callable.

    Parameters
    ----------
    epsilon: float, optional
        Value by which to shift/clip elements of ``input``.
    clip: bool, optional
        If ``False`` (default), add ``epsilon`` multiplied by each entry's sign (+1 for 0). If ``True``, instead clip
        the absolute value of ``input`` and multiply it by each entry's original sign.
    norm_scale: bool, optional
        If ``False`` (default), ``epsilon`` is added to/used to clip ``input``. If ``True``, scale ``epsilon`` by the
        square root of the mean over the squared elements of the specified dimensions ``dim``.
    dim: tuple[int], optional
        If ``norm_scale`` is ``True``, specifies the dimension over which the scaled norm should be computed (all
        except dimension 0 by default).

    '''
    def __init__(self, epsilon=1e-6, clip=False, norm_scale=False, dim=None):
        self.epsilon = epsilon
        self.clip = clip
        self.norm_scale = norm_scale
        self.dim = dim

    def __call__(self, input):
        '''Stabilize input for safe division. This shifts zero-elements by ``+ epsilon``. For the sake of the
        *epsilon rule*, this also shifts positive values by ``+ epsilon`` and negative values by ``- epsilon``.

        Parameters
        ----------
        input: :py:obj:`torch.Tensor`
            Tensor to stabilize.

        Returns
        -------
        :py:obj:`torch.Tensor`
            Stabilized ``input``.
        '''
        return stabilize(input, self.epsilon, self.clip, self.norm_scale, self.dim)

    @classmethod
    def ensure(cls, value):
        '''Given a value, return a stabilizer. If ``value`` is a float, a Stabilizer with that epsilon ``value`` is
        returned. If ``value`` is callable, it will be used directly as a stabilizer. Otherwise a TypeError will be
        raised.

        Parameters
        ----------
        value: float, int, or callable
            The value used to produce a valid stabilizer function.

        Returns
        -------
        callable or Stabilizer
            A callable to be used as a stabilizer.

        Raises
        ------
        TypeError
            If no valid stabilizer could be produced from ``value``.
        '''
        if isinstance(value, (float, int)):
            return cls(epsilon=float(value))
        if callable(value):
            return value
        raise TypeError(f'Value {value} is not a valid stabilizer!')


def stabilize(input, epsilon=1e-6, clip=False, norm_scale=False, dim=None):
    '''Stabilize input for safe division.

    Parameters
    ----------
    input: :py:obj:`torch.Tensor`
        Tensor to stabilize.
    epsilon: float, optional
        Value by which to shift/clip elements of ``input``.
    clip: bool, optional
        If ``False`` (default), add ``epsilon`` multiplied by each entry's sign (+1 for 0). If ``True``, instead clip
        the absolute value of ``input`` and multiply it by each entry's original sign.
    norm_scale: bool, optional
        If ``False`` (default), ``epsilon`` is added to/used to clip ``input``. If ``True``, scale ``epsilon`` by the
        square root of the mean over the squared elements of the specified dimensions ``dim``.
    dim: tuple[int], optional
        If ``norm_scale`` is ``True``, specifies the dimension over which the scaled norm should be computed. Defaults
        to all except dimension 0.

    Returns
    -------
    :py:obj:`torch.Tensor`
        New Tensor copied from `input` with values shifted by epsilon.
    '''
    sign = ((input == 0.).to(input) + input.sign())
    if norm_scale:
        if dim is None:
            dim = tuple(range(1, input.ndim))
        epsilon = epsilon * ((input ** 2).mean(dim=dim, keepdim=True) ** .5)
    if clip:
        return sign * input.abs().clip(min=epsilon)
    return input + sign * epsilon


def expand(tensor, shape, cut_batch_dim=False):
    '''Expand a scalar value or tensor to a shape. In addition to torch.Tensor.expand, this will also accept
    non-torch.tensor objects, which will be used to create a new tensor. If ``tensor`` has fewer dimensions than
    ``shape``, singleton dimension will be appended to match the size of ``shape`` before expanding.

    Parameters
    ----------
    tensor : int, float or :py:obj:`torch.Tensor`
        Scalar or tensor to expand to the size of ``shape``.
    shape : tuple[int]
        Shape to which ``tensor`` will be expanded.
    cut_batch_dim : bool, optional
        If True, take only the first ``shape[0]`` entries along dimension 0 of the expanded ``tensor``, if it has more
        entries in dimension 0 than ``shape``. Default (False) is not to cut, which will instead cause a
        ``RuntimeError`` due to the size mismatch.

    Returns
    -------
    :py:obj:`torch.Tensor`
        A new tensor expanded from ``tensor`` with shape ``shape``.

    Raises
    ------
    RuntimeError
        If ``tensor`` could not be expanded to ``shape`` due to incompatible shapes.

    '''
    if not isinstance(tensor, torch.Tensor):
        # cast non-tensor scalar to 0-dim tensor
        tensor = torch.tensor(tensor)
    if tensor.ndim == 0:
        # expand scalar tensors
        return tensor.expand(shape)
    if tensor.ndim < len(shape) and all(left in (1, right) for left, right in zip(tensor.shape, shape)):
        # append singleton dimensions if tensor has fewer dimensions, and the existing ones match with shape
        tensor = tensor[(...,) + (None,) * (len(shape) - len(tensor.shape))]
    if tensor.ndim == len(shape):
        # if the dims match completely (lenghts match and zipped match), expand normally
        if all(left in (1, right) for left, right in zip(tensor.shape, shape)):
            return tensor.expand(shape)
        # if `cut_batch_dim` and dims match except first, which is larger than shape, the the first dim and expand
        if (
            cut_batch_dim
            and all(left in (1, right) for left, right in zip(tensor.shape[1:], shape[1:]))
            and tensor.shape[0] > shape[0]
        ):
            return tensor[:shape[0]].expand(shape)
    raise RuntimeError('Invalid shape! Target: {tensor.shape}; Source: {shape}')


def zero_wrap(zero_params):
    '''Create a function wrapper factory (i.e. a decorator), which takes a single function argument ``(name, param) ->
    tensor`` such that the function is only called if name is not equal to zero_params, if zero_params is a string, or
    it is not in zero_params. Otherwise return `torch.zeros_like` of that tensor.

    Parameters
    ----------
    zero_params: str or list[str]
        String or list of strings compared to `name`.

    Returns
    -------
    function
        The function wrapper to be called on the function.
    '''
    def zero_params_wrapper(modifier):
        '''Wrap a function (name, param) -> tensor such that the function is only called if name is not equal to the
        closure list zero_params, if zero_params is a string, or it is not in zero_params. Otherwise return
        `torch.zeros_like` of that tensor.

        Parameters
        ----------
        func: function
            Function to wrap.

        Returns
        -------
        function
            The wrapped function.
        '''
        if not zero_params:
            return modifier

        @functools.wraps(modifier)
        def modifier_wrapper(input, name):
            '''Wrapped function (name, param) -> tensor, where the original function is only called if name is not
            equal to the closure list zero_params, if zero_params is a string, or it is not in zero_params. Otherwise
            return `torch.zeros_like` of that tensor.

            Parameters
            ----------
            input: :py:obj:`torch.Tensor`
                The input tensor modified by the original function.
            name: str
                The name associated with the input tensor (e.g. the parameter name).

            Returns
            -------
            :py:obj:`torch.Tensor`
                The modified tensor.
            '''
            if isinstance(zero_params, str) and name == zero_params or name in zero_params:
                return torch.zeros_like(input)
            return modifier(input, name)

        return modifier_wrapper
    return zero_params_wrapper


class ParamMod:
    '''Class to produce a context manager to temporarily modify parameter attributes (all by default) of a module.

    Parameters
    ----------
    modifier: function
        A function used to modify parameter attributes. If `param_keys` is empty, this is not used.
    param_keys: list[str], optional
        A list of parameter names that shall be modified. If `None` (default), all parameters are modified (which may
        be none). If `[]`, no parameters are modified and `modifier` is ignored.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    require_params: bool, optional
        Whether existence of `module`'s params is mandatory (True by default). If the attribute exists but is `None`,
        it is not considered missing, and the modifier is not applied.
    '''
    def __init__(self, modifier, param_keys=None, zero_params=None, require_params=True):
        self.modifier = modifier
        self.param_keys = param_keys
        self.zero_params = zero_params
        self.require_params = require_params

    @classmethod
    def ensure(cls, modifier):
        '''If ``modifier`` is an instance of ParamMod, return it as-is, if it is callable, create a new instance with
        ``modifier`` as the ParamMod's function, otherwise raise a TypeError.

        Parameters
        ----------
        modifier : :py:obj:`ParamMod` or callable
            The modifier which, if necessary, will be used to construct a ParamMod.

        Returns
        -------
        :py:obj:`ParamMod`
            Either ``modifier`` as is, or a :py:obj:`ParamMod` constructed using ``modifier``.

        Raises
        ------
        TypeError
            If ``modifier`` is neither an instance of :py:obj:`ParamMod`, nor callable.
        '''
        if isinstance(modifier, cls):
            return modifier
        if callable(modifier):
            return cls(modifier)
        raise TypeError(f'{modifier} is neither an instance of {cls}, nor callable!')

    @contextmanager
    def __call__(self, module):
        '''Context manager to temporarily modify parameter attributes (all by default) of a module.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            Module of which to modify parameters. If `self.requires_params` is `True`, it must have all elements given
            in `self.param_keys` as attributes (attributes are allowed to be `None`, in which case they are ignored).

        Raises
        ------
        RuntimeError
            If `self.require_params` is `True` and `module` is missing an attribute listed in `self.param_keys`.

        Yields
        ------
        module: :py:obj:`torch.nn.Module`
            The `module` with appropriate parameters temporarily modified.
        '''
        try:
            stored_params = {}
            param_keys = self.param_keys
            zero_params = self.zero_params

            if param_keys is None:
                param_keys = [name for name, _ in module.named_parameters(recurse=False)]
            if zero_params is None:
                zero_params = []

            missing = [key for key in param_keys if not hasattr(module, key)]
            if self.require_params and missing:
                missing_str = '\', \''.join(missing)
                raise RuntimeError(f'Module {module} requires missing parameters: \'{missing_str}\'')

            modifier = zero_wrap(zero_params)(self.modifier)

            for key in param_keys:
                if key not in missing:
                    param = getattr(module, key)
                    if param is not None:
                        stored_params[key] = param
                        setattr(module, key, torch.nn.Parameter(modifier(param.data, key)))

            yield module
        finally:
            for key, value in stored_params.items():
                setattr(module, key, value)


def collect_leaves(module):
    '''Generator function to collect all leaf modules of a module.

    Parameters
    ----------
    module: :py:obj:`torch.nn.Module`
        A module for which the leaves will be collected.

    Yields
    ------
    leaf: :py:obj:`torch.nn.Module`
        Either a leaf of the module structure, or the module itself if it has no children.
    '''
    is_leaf = True

    children = module.children()
    for child in children:
        is_leaf = False
        for leaf in collect_leaves(child):
            yield leaf
    if is_leaf:  # pragma: no branch
        yield module


class Identity(torch.autograd.Function):
    '''Identity to add a grad_fn to a tensor, so a backward hook can be applied.'''
    @staticmethod
    def forward(ctx, *inputs):
        '''Forward identity.'''
        return inputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        '''Backward identity.'''
        return grad_outputs


class Hook:
    '''Base class for hooks to be used to compute layerwise attributions.'''
    def __init__(self):
        self.stored_tensors = {}

    def pre_forward(self, module, input):
        '''Apply an Identity to the input before the module to register a backward hook.'''
        hook_ref = weakref.ref(self)

        @functools.wraps(self.backward)
        def wrapper(grad_input, grad_output):
            return hook_ref().backward(module, grad_input, hook_ref().stored_tensors['grad_output'])

        if not isinstance(input, tuple):
            input = (input,)

        if input[0].requires_grad:
            # only if gradient required
            post_input = Identity.apply(*input)
            post_input[0].grad_fn.register_hook(wrapper)
            # work around to support in-place operations
            post_input = tuple(elem.clone() for elem in post_input)
        else:
            # no gradient required
            post_input = input
        return post_input[0] if len(post_input) == 1 else post_input

    def post_forward(self, module, input, output):
        '''Register a backward-hook to the resulting tensor right after the forward.'''
        hook_ref = weakref.ref(self)

        @functools.wraps(self.pre_backward)
        def wrapper(grad_input, grad_output):
            return hook_ref().pre_backward(module, grad_input, grad_output)

        if not isinstance(output, tuple):
            output = (output,)

        if output[0].grad_fn is not None:
            # only if gradient required
            output[0].grad_fn.register_hook(wrapper)
        return output[0] if len(output) == 1 else output

    def pre_backward(self, module, grad_input, grad_output):
        '''Store the grad_output for the backward hook'''
        self.stored_tensors['grad_output'] = grad_output

    def forward(self, module, input, output):
        '''Hook applied during forward-pass'''

    def backward(self, module, grad_input, grad_output):
        '''Hook applied during backward-pass'''

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        '''
        return self.__class__()

    def remove(self):
        '''When removing hooks, remove all references to stored tensors'''
        self.stored_tensors.clear()

    def register(self, module):
        '''Register this instance by registering all hooks to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_pre_hook(self.pre_forward),
            module.register_forward_hook(self.post_forward),
            module.register_forward_hook(self.forward),
        ])


class BasicHook(Hook):
    '''A hook to compute the layer-wise attribution of the module it is attached to.
    A BasicHook instance may only be registered with a single module.

    Parameters
    ----------
    input_modifiers: list[callable], optional
        A list of functions ``(input: torch.Tensor) -> torch.Tensor`` to produce multiple inputs. Default is a single
        input which is the identity.
    param_modifiers: list[:py:obj:`~zennit.core.ParamMod` or callable], optional
        A list of ParamMod instances or functions ``(obj: torch.Tensor, name: str) -> torch.Tensor``, with parameter
        tensor ``obj``, registered in the root model as ``name``, to temporarily modify the parameters of the attached
        module for each input produced with `input_modifiers`. Default is unmodified parameters for each input. Use a
        :py:obj:`~zennit.core.ParamMod` instance to specify which parameters should be modified, whether they are
        required, and which should be set to zero.
    output_modifiers: list[callable], optional
        A list of functions ``(input: torch.Tensor) -> torch.Tensor`` to modify the module's output computed using the
        modified parameters before gradient computation for each input produced with `input_modifier`. Default is the
        identity for each output.
    gradient_mapper: callable, optional
        Function ``(out_grad: torch.Tensor, outputs: list[torch.Tensor]) -> list[torch.Tensor]`` to modify upper
        relevance. A list or tuple of the same size as ``outputs`` is expected to be returned. ``outputs`` has the same
        size as ``input_modifiers`` and ``param_modifiers``. Default is a stabilized normalization by each of the
        outputs, multiplied with the output gradient.
    reducer: callable
        Function ``(inputs: list[torch.Tensor], gradients: list[torch.Tensor]) -> torch.Tensor`` to reduce all the
        inputs and gradients produced through ``input_modifiers`` and ``param_modifiers``. ``inputs`` and ``gradients``
        have the same as ``input_modifiers`` and ``param_modifiers``. Default is the sum of the multiplications of each
        input and its corresponding gradient.
    '''
    def __init__(
        self,
        input_modifiers=None,
        param_modifiers=None,
        output_modifiers=None,
        gradient_mapper=None,
        reducer=None,
        stabilizer=1e-6,
    ):
        super().__init__()
        modifiers = {
            'in': input_modifiers,
            'param': param_modifiers,
            'out': output_modifiers,
        }
        supplied = {key for key, val in modifiers.items() if val is not None}
        num_mods = len(modifiers[next(iter(supplied))]) if supplied else 1
        modifiers.update({key: (self._default_modifier,) * num_mods for key in set(modifiers) - supplied})

        if gradient_mapper is None:
            gradient_mapper = self._default_gradient_mapper
        if reducer is None:
            reducer = self._default_reducer

        self.input_modifiers = modifiers['in']
        self.param_modifiers = modifiers['param']
        self.output_modifiers = modifiers['out']
        self.gradient_mapper = gradient_mapper
        self.reducer = reducer

    def forward(self, module, input, output):
        '''Forward hook to save module in-/outputs.'''
        self.stored_tensors['input'] = input

    def backward(self, module, grad_input, grad_output):
        '''Backward hook to compute LRP based on the class attributes.'''
        original_input = self.stored_tensors['input'][0].detach()
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = modified.forward(input)
                output = out_mod(output)
            inputs.append(input)
            outputs.append(output)
        grad_outputs = self.gradient_mapper(grad_output[0], outputs)
        gradients = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs)
        relevance = self.reducer(inputs, gradients)
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        '''
        copy = BasicHook.__new__(type(self))
        BasicHook.__init__(
            copy,
            self.input_modifiers,
            self.param_modifiers,
            self.output_modifiers,
            self.gradient_mapper,
            self.reducer,
        )
        return copy

    @staticmethod
    def _default_modifier(obj, name=None):
        return obj

    @staticmethod
    def _default_gradient_mapper(out_grad, outputs):
        return tuple(out_grad / stabilize(output) for output in outputs)

    @staticmethod
    def _default_reducer(inputs, gradients):
        return sum(input * gradient for input, gradient in zip(inputs, gradients))


class RemovableHandle:
    '''Create weak reference to call .remove on some instance.'''
    def __init__(self, instance):
        self.instance_ref = weakref.ref(instance)

    def remove(self):
        '''Call remove on weakly reference instance if it still exists.'''
        instance = self.instance_ref()
        if instance is not None:
            instance.remove()


class RemovableHandleList(list):
    '''A list to hold handles, with the ability to call remove on all of its members.'''
    def remove(self):
        '''Call remove on all members, effectively removing handles from modules, or reverting canonizers.'''
        for handle in self:
            handle.remove()
        self.clear()


class CompositeContext:
    '''A context object to register a composite in a context and remove the associated hooks and canonizers afterwards.

    Parameters
    ----------
    module: :py:class:`torch.nn.Module`
        The module to which `composite` should be registered.
    composite: :py:class:`zennit.core.Composite`
        The composite which shall be registered to `module`.
    '''
    def __init__(self, module, composite):
        self.module = module
        self.composite = composite

    def __enter__(self):
        self.composite.register(self.module)
        return self.module

    def __exit__(self, exc_type, exc_value, traceback):
        self.composite.remove()
        return False


class Composite:
    '''A Composite to apply canonizers and register hooks to modules.
    One Composite instance may only be applied to a single module at a time.

    Parameters
    ----------
    module_map: callable, optional
        A function ``(ctx: dict, name: str, module: torch.nn.Module) -> Hook or None`` which maps a context, name and
        module to a matching :py:class:`~zennit.core.Hook`, or ``None`` if there is no matchin
        :py:class:`~zennit.core.Hook`.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, module_map=None, canonizers=None):
        if module_map is None:
            module_map = self._empty_module_map
        if canonizers is None:
            canonizers = []

        self.module_map = module_map
        self.canonizers = canonizers

        self.handles = RemovableHandleList()

    def register(self, module):
        '''Apply all canonizers and register all hooks to a module (and its recursive children).
        Previous canonizers of this composite are reverted and all hooks registered by this composite are removed.
        The module or any of its children (recursively) may still have other hooks attached.

        Parameters
        ----------
        module: :py:class:`torch.nn.Module`
            Hooks and canonizers will be applied to this module recursively according to ``module_map`` and
            ``canonizers``.
        '''
        self.handles.remove()

        for canonizer in self.canonizers:
            self.handles += canonizer.apply(module)

        ctx = {}
        for name, child in module.named_modules():
            template = self.module_map(ctx, name, child)
            if template is not None:
                hook = template.copy()
                self.handles.append(hook.register(child))

    def remove(self):
        '''Remove all handles for hooks and canonizers.
        Hooks will simply be removed from their corresponding Modules.
        Canonizers will revert the state of the modules they changed.
        '''
        self.handles.remove()

    def context(self, module):
        '''Return a CompositeContext object with this instance and the supplied module.

        Parameters
        ----------
        module: :py:class:`torch.nn.Module`
            Module for which to register this composite in the context.

        Returns
        -------
        :py:class:`zennit.core.CompositeContext`
            A context object which registers the composite to ``module`` on entering, and removes it on exiting.
        '''
        return CompositeContext(module, self)

    @staticmethod
    def _empty_module_map(ctx, name, module):
        '''Empty module_map, does not assign any rules.'''
        return None
