'''Core functions and classes'''
import functools
from contextlib import contextmanager

import torch


def stabilize(input, epsilon=1e-6):
    '''Stabilize input for safe division.

    Parameters
    ----------
    input: obj:`torch.Tensor`
        Tensor to stabilize.
    epsilon: float, optional
        Value to replace zero elements with.

    Returns
    -------
    obj:`torch.Tensor`
        New Tensor copied from `input` with all zero elements set to epsilon.
    '''
    return input + ((input == 0.).to(input) + input.sign()) * epsilon


@contextmanager
def mod_params(module, modifier):
    '''Context manager to temporarily modify `weight` and `bias` attributes of a linear layer, or the identity of the
    module when `modifier` is `None`.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        Linear layer with mandatory attributes `weight` and `bias`, or any module if `modifier` is `None`
    modifier: function
        A function to modify attributes `weight` and `bias`, or `None`.

    Yields
    ------
    module: obj:`torch.nn.Module`
        The input temporarily modified linear layer `module`.
    '''
    try:
        if modifier is None:
            yield modifier
        else:
            if module.weight is not None:
                original_weight = module.weight.data
                module.weight.data = modifier(module.weight.data)
            if module.bias is not None:
                original_bias = module.bias.data
                module.bias.data = modifier(module.bias.data)
            yield module
    finally:
        if modifier is not None:
            if module.weight is not None:
                module.weight.data = original_weight
            if module.bias is not None:
                module.bias.data = original_bias


def collect_leaves(module):
    '''Generator function to collect all leaf modules of a module.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        A module for which the leaves will be collected.

    Yields
    ------
    leaf: obj:`torch.nn.Module`
        Either a leaf of the module structure, or the module itself if it has no children.
    '''
    is_leaf = True

    children = module.children()
    for child in children:
        is_leaf = False
        for leaf in collect_leaves(child):
            yield leaf
    if is_leaf:
        yield module


class Identity(torch.autograd.Function):
    '''Identity to add a grad_fn to a tensor, so a backward hook can be applied.'''
    @staticmethod
    def forward(ctx, *inputs):
        '''Forward identity.'''
        if len(inputs) == 1:
            return inputs[0]
        return inputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        '''Backward identity.'''
        if len(grad_outputs) == 1:
            return grad_outputs[0]
        return grad_outputs


class Hook:
    '''Base class for hooks to be used to compute layerwise attributions.'''
    def __init__(self):
        self.grad_output = None

    def pre_forward(self, module, input):
        '''Apply an Identity to the input before the module to register a backward hook.'''
        @functools.wraps(self.backward)
        def wrapper(grad_input, grad_output):
            return self.backward(module, grad_input, self.grad_output)

        output = Identity.apply(input[0])
        output.grad_fn.register_hook(wrapper)
        # work around to support in-place operations
        output = output.clone()
        return (output,)

    def pre_backward(self, module, grad_input, grad_output):
        '''Store the grad_output for the backward hook'''
        self.grad_output = grad_output

    def forward(self, module, input, output):
        '''Hook applied during forward-pass'''

    def backward(self, module, grad_input, grad_output):
        '''Hook applied during backward-pass'''

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        '''
        return self.__class__()


class LinearHook(Hook):
    '''A hook to compute the layerwise attribution of the layer it is attached to.
    A `LinearHook` instance may only be registered with a single module.

    Parameters
    ----------
    input_modifiers: list of function
        A list of functions to produce multiple inputs.
    param_modifiers: list of function
        A list of functions to temporarily modify the parameters of the attached linear layer for each input produced
        with `input_modifiers`.
    gradient_mapper: function
        Function to modify upper relevance. Call signature is of form `(grad_output, outputs)` and a tuple of
        the same size as outputs is expected to be returned. `outputs` has the same size as `input_modifiers` and
        `param_modifiers`.
    reducer: function
        Function to reduce all the inputs and gradients produced through `input_modifiers` and `param_modifiers`.
        Call signature is of form `(inputs, gradients)`, where `inputs` and `gradients` have the same as
        `input_modifiers` and `param_modifiers`
    '''
    def __init__(
        self,
        input_modifiers=None,
        param_modifiers=None,
        output_modifiers=None,
        gradient_mapper=None,
        reducer=None
    ):
        super().__init__()
        modifiers = {
            'in': input_modifiers,
            'param': param_modifiers,
            'out': output_modifiers,
        }
        supplied = {key for key, val in modifiers.items() if val is not None}
        num_mods = len(next(iter(supplied), (None,)))
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

        self.input = None
        self.output = None

    def forward(self, module, input, output):
        '''Forward hook to save module in-/outputs.'''
        self.input = input

    def backward(self, module, grad_input, grad_output):
        '''Backward hook to compute LRP based on the class attributes.'''
        inputs = []
        outputs = []
        for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
            input = in_mod(self.input[0].detach()).requires_grad_()
            with mod_params(module, param_mod) as modified, torch.autograd.enable_grad():
                output = modified.forward(input)
                output = out_mod(output)
            inputs.append(input)
            outputs.append(output)
        gradients = torch.autograd.grad(outputs, inputs, grad_outputs=self.gradient_mapper(grad_output[0], outputs))
        # relevance = self.reducer([input.detach() for input in inputs], [gradient.detach() for gradient in gradients])
        relevance = self.reducer(inputs, gradients)
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        '''
        return LinearHook(
            self.input_modifiers,
            self.param_modifiers,
            self.output_modifiers,
            self.gradient_mapper,
            self.reducer
        )

    @staticmethod
    def _default_modifier(obj):
        return obj

    @staticmethod
    def _default_gradient_mapper(out_grad, outputs):
        return tuple(out_grad / stabilize(output) for output in outputs)

    @staticmethod
    def _default_reducer(inputs, gradients):
        return sum(input * gradient for input, gradient in zip(inputs, gradients))


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
    module: obj:`torch.nn.Module`
        The module to which `composite` should be registered.
    composite: obj:`Composite`
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


class Composite:
    '''A Composite to apply canonizers and register hooks to modules.
    One Composite instance may only be applied to a single module at a time.

    Parameters
    ----------
    module_map: list[function, Hook]]
        A mapping from functions that check applicability of hooks to hook instances that shall be applied to instances
        of applicable modules.
    canonizers: list[Canonizer]
        List of canonizer instances to be applied before applying hooks.
    '''
    def __init__(self, module_map=None, canonizers=None):
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
        module: obj:`torch.nn.Module`
            Hooks and canonizers will be applied to this module recursively according to `module_map` and `canonizers`
        '''
        self.handles.remove()

        for canonizer in self.canonizers:
            self.handles += canonizer.apply(module)

        ctx = {}
        for name, child in module.named_modules():
            template = self.module_map(ctx, name, child)
            if template is not None:
                hook = template.copy()
                self.handles.append(child.register_forward_pre_hook(hook.pre_forward))
                self.handles.append(child.register_forward_hook(hook.forward))
                self.handles.append(child.register_backward_hook(hook.pre_backward))

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
        module: obj:`torch.nn.module`
            Module for which to register this composite in the context.
        '''
        return CompositeContext(module, self)
