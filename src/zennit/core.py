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
from typing import Generator, Iterator
from itertools import compress, repeat
from inspect import signature

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
        # if the dims match completely (lengths match and zipped match), expand normally
        if all(left in (1, right) for left, right in zip(tensor.shape, shape)):
            return tensor.expand(shape)
        # if `cut_batch_dim` and dims match except first, which is larger than shape, the the first dim and expand
        if (
            cut_batch_dim
            and all(left in (1, right) for left, right in zip(tensor.shape[1:], shape[1:]))
            and tensor.shape[0] > shape[0]
        ):
            return tensor[:shape[0]].expand(shape)
    raise RuntimeError(f'Invalid shape! Target: {tensor.shape}; Source: {shape}')


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
        modifier: function
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


def uncompress(data, selector, compressed) -> Generator:
    '''Generator which, given a compressed iterable produced by :py:obj:`itertools.compress` and (some iterable similar
    to) the original data and selector used for :py:obj:`~itertools.compress`, yields values from `compressed` or
    `data` depending on `selector`. `True` values in `selector` skip `data` one ahead and yield a value from
    `compressed`, while `False` values yield one value from `data`.

    Parameters
    ----------
    data : iterable
        The iterable (similar to the) original data. `False` values in the `selector` will be filled with values from
        this iterator, while `True` values will cause this iterable to be skipped.
    selector : iterable of bool
        The original selector used to produce `compressed`. Chooses whether elements from `data` or from `compressed`
        will be yielded.
    compressed : iterable
        The results of :py:obj:`itertools.compress`. Will be yielded for each `True` element in `selector`.

    Yields
    ------
    object
        An element of `data` if the associated element of `selector` is `False`, otherwise an element of `compressed`
        while skipping `data` one ahead.

    '''
    its = iter(selector)
    itc = iter(compressed)
    itd = iter(data)
    for select in its:
        try:
            if select:
                next(itd)
                yield next(itc)
            else:
                yield next(itd)
        except StopIteration:
            break


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

    def state_dicts(self, module):
        '''Returns a state_dict of the modified module parameters.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module for which parameters shall be modified.

        Returns
        -------
        original_state: dict of :py:obj:`torch.Tensor`
            The original, unmodified parameters.
        modified_state: dict of :py:obj:`torch.Tensor`
            The modified parameters.

        Raises
        ------
        RuntimeError
            If parameters are missing and `self.require_params` has been set to ``True``.
        '''
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

        modified_state = {}
        original_state = {}
        for key in param_keys:
            if key not in missing:
                param = getattr(module, key)
                if param is not None:
                    original_state[key] = param.data
                    modified_state[key] = modifier(param.data, key)
        return original_state, modified_state

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
    def __call__(self, module) -> Generator:
        '''Context manager to temporarily modify parameter attributes (all by default) of a module.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            Module of which to modify parameters. If `self.requires_params` is `True`, it must have all elements given
            in `self.param_keys` as attributes (attributes are allowed to be `None`, in which case they are ignored).

        Yields
        ------
        module: :py:obj:`torch.nn.Module`
            The `module` with appropriate parameters temporarily modified.
        '''
        # assign empty dict, as either the following two functions may crash
        original_state = {}
        try:
            original_state, modified_state = self.state_dicts(module)
            module.load_state_dict(modified_state, strict=False, assign=True)
            yield module
        finally:
            module.load_state_dict(original_state, strict=False, assign=True)


def collect_leaves(module) -> Iterator[torch.nn.Module]:
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
        yield from collect_leaves(child)
    if is_leaf:  # pragma: no branch
        yield module


class Identity(torch.autograd.Function):
    '''Identity to add a grad_fn to a tensor, so a backward hook can be applied.'''
    @staticmethod
    def forward(ctx, *inputs):
        '''Forward identity.

        Parameters
        ----------
        ctx: object
            The function context.
        *inputs: tuple of :py:obj:`torch.Tensor`
            Inputs to forward.

        Returns
        -------
        inputs: tuple of :py:obj:`torch.Tensor`
            The unmodified inputs.
        '''
        ctx.mark_non_differentiable(*[elem for elem in inputs if not elem.requires_grad])
        return inputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        '''Backward identity.

        Parameters
        ----------
        ctx: object
            The function context.
        *grad_outputs: tuple of :py:obj:`torch.Tensor`
            Output gradients.

        Returns
        -------
        grad_outputs: tuple of :py:obj:`torch.Tensor`
            The unmodified output gradients.
        '''
        return grad_outputs


class HookBase:
    '''Base for Hook functionality. Every hook must implement this interface.'''
    def register(self, module):
        '''Attach this hook to a module. This modifies forward/backward computations. Returns a handle which can be
        used to call this Hook's ``.remove``.
        '''
        return RemovableHandle(self)

    def remove(self):
        '''Remove this hook. Removes all references and modifications it introduced.'''

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        '''
        return self.__class__()


class GradOutHook(HookBase):
    '''Hook to only modify the output gradient of a module. This leaves the gradient computation of the module intact.
    '''
    def post_forward(self, module, input, output):
        '''Register a backward-hook to the resulting tensor right after the forward.'''
        hook_ref = weakref.ref(self)

        @functools.wraps(self.backward)
        def wrapper(grad_input, grad_output):
            hook = hook_ref()
            if hook is not None and hook.active:
                return hook.backward(module, grad_output)
            return None

        if not isinstance(output, tuple):
            output = (output,)

        # only if gradient required
        if output[0].grad_fn is not None:
            # add identity to ensure .grad_fn exists
            post_output = Identity.apply(*output)
            # register the input tensor gradient hook
            self.tensor_handles.append(
                post_output[0].grad_fn.register_hook(wrapper)
            )
            # work around to support in-place operations
            post_output = tuple(elem.clone() for elem in post_output)
        else:
            # no gradient required
            post_output = output
        return post_output[0] if len(post_output) == 1 else post_output

    def backward(self, module, grad_output):
        '''Hook applied during backward-pass. Modifies the output gradient of module before its gradient
        computation.
        '''

    def register(self, module):
        '''Register this instance by registering the neccessary forward hook to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_hook(self.post_forward),
        ])


class Hook(HookBase):
    '''Base class for hooks to be used to compute layer-wise attributions.'''
    def __init__(self):
        self.stored_tensors = {}
        self.active = True
        self.tensor_handles = RemovableHandleList()

    @staticmethod
    def _inject_grad_fn(args):
        tensor_mask = tuple(isinstance(elem, torch.Tensor) for elem in args)
        tensors = tuple(compress(args, tensor_mask))
        # tensors = [(n, elem) for elem in enumerate(args) if isinstance(elem, torch.Tensor)]

        # only if gradient required
        if not any(tensor.requires_grad for tensor in tensors):
            return None, args, tensor_mask

        # add identity to ensure .grad_fn exists and all tensors share the same .grad_fn
        post_tensors = Identity.apply(*tensors)
        grad_fn = next((tensor.grad_fn for tensor in post_tensors if tensor.grad_fn is not None), None)
        if grad_fn is None:
            # sanity check, should never happen because the check above already catches cases in which no input tensor
            # requires a gradient, and in normal conditions, we will always obtain a grad_fn from `Identity` for each
            # tensor with requires_grad=True
            raise RuntimeError('Backward hook could not be registered!')  # pragma: no cover

        # work-around to support in-place operations
        post_tensors = tuple(elem.clone() for elem in post_tensors)
        post_args = tuple(uncompress(args, tensor_mask, post_tensors))
        return grad_fn, post_args, tensor_mask

    def pre_forward(self, module, args, kwargs):
        '''Apply an Identity to the input before the module to register a backward hook.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        args: tuple of :py:obj:`torch.Tensor`
            The input tensors passed to ``module.forward``.
        kwargs: dict
            The keyword arguments passed to ``module.forward``.

        Returns
        -------
        tuple of :py:obj:`torch.Tensor`, optional
            A tuple of the modified input tensors.

        '''
        hook_ref = weakref.ref(self)

        grad_fn, post_args, input_tensor_mask = self._inject_grad_fn(args)
        if grad_fn is None:
            return None

        @functools.wraps(self.backward)
        def wrapper(grad_input, grad_output):
            hook = hook_ref()
            if hook is not None and hook.active:
                return hook.backward(
                    module,
                    list(uncompress(
                        repeat(None),
                        input_tensor_mask,
                        grad_input,
                    )),
                    hook.stored_tensors['grad_output'],
                )
            return None

        # register the input tensor gradient hook
        self.tensor_handles.append(grad_fn.register_hook(wrapper))

        return post_args, kwargs

    def post_forward(self, module, args, kwargs, output):
        '''Register a backward-hook to the resulting tensor right after the forward.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        args: tuple of :py:obj:`torch.Tensor`
            The input tensors passed to ``module.forward``.
        kwargs: tuple of object
            The keyword arguments passed to ``module.forward``.
        output: :py:obj:`torch.Tensor`
            The output tensor.

        Returns
        -------
        tuple of :py:obj:`torch.Tensor`, optional
            A tuple of the modified output tensors.
        '''
        hook_ref = weakref.ref(self)

        single = not isinstance(output, tuple)
        if single:
            output = (output,)

        grad_fn, post_output, output_tensor_mask = self._inject_grad_fn(output)
        if grad_fn is None:
            return None

        @functools.wraps(self.pre_backward)
        def wrapper(grad_input, grad_output):
            hook = hook_ref()
            if hook is not None and hook.active:
                return hook.pre_backward(
                    module,
                    grad_input,
                    tuple(uncompress(
                        repeat(None),
                        output_tensor_mask,
                        grad_output
                    ))
                )
            return None

        # register the output tensor gradient hook
        self.tensor_handles.append(grad_fn.register_hook(wrapper))

        if single:
            return post_output[0]
        return post_output

    def pre_backward(self, module, grad_input, grad_output):
        '''Store the grad_output for the backward hook.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        grad_input: :py:obj:`torch.Tensor`
            The input gradient tensor.
        grad_output: :py:obj:`torch.Tensor`
            The output gradient tensor.
        '''
        self.stored_tensors['grad_output'] = grad_output

    def forward(self, module, args, kwargs, output):
        '''Hook applied during forward-pass.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        args: tuple of :py:obj:`torch.Tensor`
            The input tensors passed to ``module.forward``.
        kwargs: tuple of object
            The keyword arguments passed to ``module.forward``.
        output: :py:obj:`torch.Tensor`
            The output tensor.
        '''

    def backward(self, module, grad_input, grad_output):
        '''Hook applied during backward-pass.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        grad_input: :py:obj:`torch.Tensor`
            The input gradient tensor.
        grad_output: :py:obj:`torch.Tensor`
            The output gradient tensor.
        '''

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.

        Returns
        -------
        :py:obj:`BasicHook`
            A copy of this hook.

        '''
        return self.__class__()

    def remove(self):
        '''When removing hooks, remove all references to stored tensors.'''
        self.stored_tensors.clear()
        self.tensor_handles.remove()

    def register(self, module):
        '''Register this instance by registering all hooks to the supplied module.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which to register to.

        Returns
        -------
        :py:obj:`RemovableHandleList`
            A list of removable handles, one for each registered hook.

        '''
        def with_kwargs(method, has_output=True):
            '''Check whether the method uses args/kwargs, or only inputs. This ensures compatibility with rules that do
            not consider kwargs, and reduces code clutter.

            Parameters
            ----------
            method: function
                Function to check.
            has_output: bool
                Function to check.

            Returns
            -------
            bool
                True if `method` uses kwargs.
            '''
            params = signature(method).parameters
            # assume with_kwargs if forward has not 3 parameters and 3rd is not called 'output'
            if has_output:
                return len(params) != 3 and list(params)[2] != 'output'
            # e.g., pre_forward has no output, so we expect 2 parameters
            return len(params) != 2

        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_pre_hook(self.pre_forward, with_kwargs=with_kwargs(self.pre_forward, False)),
            module.register_forward_hook(self.post_forward, with_kwargs=with_kwargs(self.post_forward)),
            module.register_forward_hook(self.forward, with_kwargs=with_kwargs(self.forward)),
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
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
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

    def forward(self, module, args, kwargs, output):
        '''Forward hook to save module in-/outputs.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        args: tuple of :py:obj:`torch.Tensor`
            The input tensors passed to ``module.forward``.
        kwargs: tuple of object
            The keyword arguments passed to ``module.forward``.
        output: :py:obj:`torch.Tensor`
            The output tensor.
        '''
        self.stored_tensors['input'] = args
        self.stored_tensors['kwargs'] = kwargs

    def backward(self, module, grad_input, grad_output):
        '''Backward hook to compute LRP based on the class attributes.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        grad_input: :py:obj:`torch.Tensor`
            The input gradient tensor.
        grad_output: :py:obj:`torch.Tensor`
            The output gradient tensor.

        Returns
        -------
        tuple of :py:obj:`torch.nn.Module`
            The modified input gradient tensors.
        '''
        original_input, *original_args = self.stored_tensors['input']
        original_input = original_input.clone()
        original_kwargs = self.stored_tensors['kwargs']
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(original_input).requires_grad_()
            with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                output = modified.forward(input, *original_args, **original_kwargs)
                output = out_mod(output)
            inputs.append(input)
            outputs.append(output)
        grad_outputs = self.gradient_mapper(grad_output[0], outputs)
        gradients = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad
        )
        relevance = self.reducer(inputs, gradients)
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.

        Returns
        -------
        :py:obj:`BasicHook`
            A copy of this hook.
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
    '''Create weak reference to call .remove on some instance.

    Parameters
    ----------
    instance: object
        The instance to which to create the reference.
    '''
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
        self.hook_refs = weakref.WeakSet()

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
        self.remove()

        for canonizer in self.canonizers:
            self.handles += canonizer.apply(module)

        ctx = {}
        for name, child in module.named_modules():
            templates = self.module_map(ctx, name, child)
            try:
                templates = iter(template)
            else:
                templates = (template,)

            for template in templates:
                if template is not None:
                    hook = template.copy()
                    self.hook_refs.add(hook)
                    self.handles.append(hook.register(child))

    def remove(self):
        '''Remove all handles for hooks and canonizers.
        Hooks will simply be removed from their corresponding Modules.
        Canonizers will revert the state of the modules they changed.
        '''
        self.handles.remove()
        self.hook_refs.clear()

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

    @contextmanager
    def inactive(self) -> Generator:
        '''Context manager to temporarily deactivate the gradient modification. This can be used to compute the
        gradient of the modified gradient.

        Yields
        ------
        self: :py:obj:`Composite`
            The instance of this composite.
        '''
        try:
            for hook in self.hook_refs:
                hook.active = False
            yield self
        finally:
            for hook in self.hook_refs:
                hook.active = True

    @staticmethod
    def _empty_module_map(ctx, name, module):
        '''Empty module_map, does not assign any rules.

        Parameters
        ----------
        ctx: dict
            A dictionary containing an optional context.
        name: str
            The name of the module.
        module: :py:obj:`torch.nn.Module`
            The instance of the module.

        Returns
        -------
        NoneType
            Always `None`, as the empty module map will never register any hooks.
        '''
        return None
