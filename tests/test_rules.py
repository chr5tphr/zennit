'''Tests for various rules. Rules are re-implemented in a slower, less complicated way, which closely follows the
definition in the original works, which makes them easier to compare and thus less likely to be wrong.
'''
from functools import wraps, partial
from copy import deepcopy

import pytest
import torch
from zennit.rules import Epsilon, ZPlus, AlphaBeta, Gamma, ZBox, Norm, WSquare, Flat


def stabilize(input, epsilon=1e-6):
    '''Replicates zennit.core.stabilize for testing.'''
    return input + ((input == 0.).to(input) + input.sign()) * epsilon


def as_matrix(module_linear, input, output):
    '''Get flat weight and bias using the jacobian.'''
    jac = torch.autograd.functional.jacobian(module_linear, input[None])
    weight = jac.reshape((output.numel(), input.numel()))
    bias = output.flatten() - weight @ input.flatten()
    return weight, bias


RULEPAIRS = []


def replicates(replicated_func, **kwargs):
    '''Decorator to indicate a replication of a function for testing.'''
    def wrapper(func):
        '''Append to ``RULEPAIRS`` as partial, given ``kwargs``.'''
        RULEPAIRS.append(
            pytest.param(
                (partial(replicated_func, **kwargs), partial(func, **kwargs)),
                id=replicated_func.__name__
            )
        )
        return func
    return wrapper


def flat_module_params(func):
    '''Decorator to to copy module and overwrite module params completely with ones (for rule_flat).'''
    @wraps(func)
    def wrapped(module_linear, *args, **kwargs):
        '''Make a deep copy of module_linear, fill all parameters inline with ones, and call func with the copy.'''
        module_copy = deepcopy(module_linear)
        for param in module_copy.parameters():
            param.requires_grad_(False).fill_(1.0)
        return func(module_copy, *args, **kwargs)
    return wrapped


def matrix_form(func):
    '''Decorator to wrap function such that weights and bias supplied in matrix-form and input and output are flattened
    appropriately.'''
    @wraps(func)
    def wrapped(module_linear, input, output, **kwargs):
        '''Get flat weight matrix and bias using the jacobian, flatten input and output, and pass arguments to func.'''
        weight, bias = as_matrix(module_linear, input[0], output[0])
        return func(
            weight,
            bias,
            input.flatten(start_dim=1),
            output.flatten(start_dim=1),
            **kwargs
        ).reshape(input.shape)
    return wrapped


@replicates(Epsilon, epsilon=1e-6)
@replicates(Epsilon, epsilon=1.0)
@replicates(Norm)
@matrix_form
def rule_epsilon(weight, bias, input, relevance, epsilon=1e-6):
    '''Replicates the Epsilon rule.'''
    return input * ((relevance / stabilize(input @ weight.t() + bias, epsilon)) @ weight)


@replicates(ZPlus)
@matrix_form
def rule_zplus(weight, bias, input, relevance):
    '''Replicates the ZPlus rule.'''
    wplus = weight.clamp(min=0)
    wminus = weight.clamp(max=0)
    xplus = input.clamp(min=0)
    xminus = input.clamp(max=0)
    zval = xplus @ wplus.t() + xminus @ wminus.t() + bias.clamp(min=0)
    rfac = relevance / stabilize(zval)
    return xplus * (rfac @ wplus) + xminus * (rfac @ wminus)


@replicates(Gamma, gamma=0.25)
@replicates(Gamma, gamma=0.5)
@matrix_form
def rule_gamma(weight, bias, input, relevance, gamma):
    '''Replicates the Gamma rule.'''
    wgamma = weight + weight.clamp(min=0) * gamma
    bgamma = bias + bias.clamp(min=0) * gamma
    return input * ((relevance / stabilize(input @ wgamma.t() + bgamma)) @ wgamma)


@replicates(AlphaBeta, alpha=2.0, beta=1.0)
@replicates(AlphaBeta, alpha=1.0, beta=0.0)
@matrix_form
def rule_alpha_beta(weight, bias, input, relevance, alpha, beta):
    '''Replicates the AlphaBeta rule.'''
    wplus = weight.clamp(min=0)
    wminus = weight.clamp(max=0)
    xplus = input.clamp(min=0)
    xminus = input.clamp(max=0)
    zalpha = xplus @ wplus.t() + xminus @ wminus.t() + bias.clamp(min=0)
    zbeta = xplus @ wminus.t() + xminus @ wplus.t() + bias.clamp(max=0)
    ralpha = relevance / stabilize(zalpha)
    rbeta = relevance / stabilize(zbeta)
    result_alpha = xplus * (ralpha @ wplus) + xminus * (ralpha @ wminus)
    result_beta = xplus * (rbeta @ wminus) + xminus * (rbeta @ wplus)
    return alpha * result_alpha - beta * result_beta


@replicates(ZBox, low=-3.0, high=3.0)
@matrix_form
def rule_zbox(weight, bias, input, relevance, low, high):
    '''Replicates the ZBox rule.'''
    wplus = weight.clamp(min=0)
    wminus = weight.clamp(max=0)
    low = torch.tensor(low).expand_as(input).to(input)
    high = torch.tensor(high).expand_as(input).to(input)
    zval = input @ weight.t() - low @ wplus.t() - high @ wminus.t()
    rfac = relevance / stabilize(zval)
    return input * (rfac @ weight) - low * (rfac @ wplus) - high * (rfac @ wminus)


@replicates(WSquare)
@matrix_form
def rule_wsquare(weight, bias, input, relevance):
    '''Replicates the WSquare rule.'''
    wsquare = weight ** 2
    zval = torch.ones_like(input) @ wsquare.t() + bias ** 2
    rfac = relevance / stabilize(zval)
    return rfac @ wsquare


@replicates(Flat)
@flat_module_params
@matrix_form
def rule_flat(wflat, bias, input, relevance):
    '''Replicates the Flat rule.'''
    zval = torch.ones_like(input) @ wflat.t()
    rfac = relevance / stabilize(zval)
    return rfac @ wflat


@pytest.fixture(scope='session', params=RULEPAIRS)
def rule_pair(request):
    '''Fixture to supply ``RULEPAIRS``.'''
    return request.param


def test_linear_rule(module_linear, data_input, rule_pair):
    '''Test whether replicated and original implementations of rules for linear layers agree.'''
    rule_hook, rule_replicated = rule_pair

    input = data_input.clone().requires_grad_()
    handle = rule_hook().register(module_linear)
    try:
        output = module_linear(input)
        relevance_hook, = torch.autograd.grad(output, input, grad_outputs=output)
    finally:
        handle.remove()

    with torch.no_grad():
        relevance_replicated = rule_replicated(module_linear, input, output)

    assert torch.allclose(relevance_hook, relevance_replicated, atol=1e-5)
