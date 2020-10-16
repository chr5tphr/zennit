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


def merge_batch_norm(module, batch_norm):
    '''Update parameters of a linear layer to additionally include a Batch Normalization operation and update the batch
    normalization layer to instead compute the identity.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        Linear layer with mandatory attributes `weight` and `bias`.
    batch_norm: obj:`torch.nn.Module`
        Batch Normalization module with mandatory attributes `running_mean`, `running_var`, `weight`, `bias` and `eps`

    Returns
    -------
    None
    '''
    original_weight = module.weight.data
    if module.bias is None:
        module.bias = torch.nn.Parameter(torch.zeros(1, device=original_weight.device, dtype=original_weight.dtype))
    original_bias = module.bias.data

    denominator = (batch_norm.running_var + batch_norm.eps) ** .5
    scale = (batch_norm.weight / denominator)

    # merge batch_norm into linear layer
    module.weight.data = (original_weight * scale[:, None, None, None])
    module.bias.data = (original_bias - batch_norm.running_mean) * scale + batch_norm.bias

    # change batch_norm parameters to produce identity
    batch_norm.running_mean.data = torch.zeros_like(batch_norm.running_mean.data)
    batch_norm.running_var.data = torch.ones_like(batch_norm.running_var.data)
    batch_norm.bias.data = torch.zeros_like(batch_norm.bias.data)
    batch_norm.weight.data = torch.ones_like(batch_norm.weight.data)


@contextmanager
def mod_params(module, modifier):
    '''Context manager to temporarily modify `weight` and `bias` attributes of a linear layer.

    Parameters
    ----------
    module: obj:`torch.nn.Module`
        Linear layer with mandatory attributes `weight` and `bias`.
    modifier: function
        A function to modify attributes `weight` and `bias`

    Yields
    ------
    module: obj:`torch.nn.Module`
        The input temporarily modified linear layer `module`.
    '''
    try:
        if module.weight is not None:
            original_weight = module.weight.data
            module.weight.data = modifier(module.weight.data)
        if module.bias is not None:
            original_bias = module.bias.data
            module.bias.data = modifier(module.bias.data)
        yield module
    finally:
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
