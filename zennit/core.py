'''Core functions and classes'''
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


class Hook:
    '''Base class for hooks to be used to compute layerwise attributions.'''
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
    def __init__(self, input_modifiers, param_modifiers, gradient_mapper, reducer):
        super().__init__()
        self.input_modifiers = input_modifiers
        self.param_modifiers = param_modifiers
        self.gradient_mapper = gradient_mapper
        self.reducer = reducer

        self.input = None
        self.output = None

    def forward(self, module, input, output):
        '''Forward hook to save module in-/outputs.'''
        self.input = input
        self.output = output

    def backward(self, module, grad_input, grad_output):
        '''Backward hook to compute LRP based on the class attributes.'''
        inputs = []
        outputs = []
        for in_mod, param_mod in zip(self.input_modifiers, self.param_modifiers):
            input = in_mod(self.input[0].detach()).requires_grad_()
            with mod_params(module, param_mod) as modified:
                output = modified.forward(input)
            inputs.append(input)
            outputs.append(output)
        gradients = torch.autograd.grad(inputs, outputs, grad_outputs=self.gradient_mapper(grad_output, outputs))
        return self.reducer([input.detach_() for input in inputs], [gradient.detach_() for gradient in gradients])

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        '''
        return self.__class__(self.input_modifiers, self.param_modifiers, self.gradient_mapper, self.reducer)


class RemovableHandleList(list):
    '''A list to hold handles, with the ability to call remove on all of its members.'''
    def remove(self):
        '''Call remove on all members, effectively removing handles from modules, or reverting canonizers.'''
        for handle in self:
            handle.remove()
        self.clear()


class PresetContext:
    '''A context object to register a preset in a context and remove the associated hooks and canonizers afterwards.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        The module to which `preset` should be registered.
    preset: obj:`Preset`
        The preset which shall be registered to `module`.
    '''
    def __init__(self, module, preset):
        self.module = module
        self.preset = preset

    def __enter__(self):
        self.preset.register(self.module)
        return self.module

    def __exit__(self, exc_type, exc_value, traceback):
        self.preset.remove()


class Preset:
    '''A Preset to apply canonizers and register hooks to modules.
    One Preset instance may only be applied to a single module at a time.

    Parameters
    ----------
    module_map: list[function, Hook]]
        A mapping from functions that check applicability of hooks to hooks that shall be applied to instances of
        applicable hooks.
    canonizers: list[Canonizer]
        List of canonizers to be applied before applying hooks.
    '''
    def __init__(self, module_map=None, canonizers=None):
        self.module_map = module_map
        self.canonizers = canonizers

        self.handles = RemovableHandleList()

    def register(self, module):
        '''Apply all canonizers and register all hooks to a module (and its recursive children).
        Previous canonizers of this preset are reverted and all hooks registered by this preset are removed.
        The module or any of its children (recursively) may still have other hooks attached.

        Parameters
        ----------
        module: obj:`torch.nn.Module`
            Hooks and canonizers will be applied to this module recursively according to `module_map` and `canonizers`
        '''
        self.handles.remove()

        for canonizer in self.canonizers:
            self.handles.append(canonizer(module))

        for name, child in module.named_modules():
            for applicable, hook_template in self.module_map:
                if applicable(name, child):
                    hook = hook_template.copy()
                    self.handles.append(child.register_forward_hook(hook.forward))
                    self.handles.append(child.register_backward_hook(hook.backward))

    def remove(self):
        '''Remove all handles for hooks and canonizers.
        Hooks will simply be removed from their corresponding Modules.
        Canonizers will revert the state of the modules they changed.
        '''
        self.handles.remove()

    def context(self, module):
        '''Return a PresetContext object with this instance and the supplied module.

        Parameters
        ----------
        module: obj:`torch.nn.module`
            Module for which to register this preset in the context.
        '''
        return PresetContext(module, self)
