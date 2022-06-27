'''Tests for various rules. Rules are re-implemented in a slower, less complicated way, which closely follows the
definition in the original works, which makes them easier to compare and thus less likely to be wrong.
'''
from functools import wraps, partial
from copy import deepcopy

import pytest
import torch
from zennit.rules import Epsilon, ZPlus, AlphaBeta, Gamma, ZBox, Norm, WSquare, Flat
from zennit.rules import Pass, ReLUDeconvNet, ReLUGuidedBackprop
from zennit.rules import zero_bias as name_zero_bias


def stabilize(input, epsilon=1e-6):
    '''Replicates zennit.core.stabilize for testing.'''
    return input + ((input == 0.).to(input) + input.sign()) * epsilon


def as_matrix(module_linear, input, output):
    '''Get flat weight and bias using the jacobian.'''
    jac = torch.autograd.functional.jacobian(module_linear, input[None])
    weight = jac.reshape((output.numel(), input.numel()))
    bias = output.flatten() - weight @ input.flatten()
    return weight, bias


RULES_LINEAR = []
RULES_SIMPLE = []


def replicates(target_list, replicated_func, **kwargs):
    '''Decorator to indicate a replication of a function for testing.'''
    def wrapper(func):
        '''Append to ``RULES_LINEAR`` as partial, given ``kwargs``.'''
        target_list.append(
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


def with_grad(func):
    '''Decorator to wrap function such that the gradient is computed and passed to the function instead of module.'''
    @wraps(func)
    def wrapped(module, input, output, **kwargs):
        '''Get gradient and pass along input, output and keyword arguments to func.'''
        gradient, = torch.autograd.grad(module(input), input, output)
        return func(
            gradient,
            input,
            output,
            **kwargs
        )
    return wrapped


def zero_bias(zero_params, bias):
    '''Return a tensor with zeros like ``bias`` if zero_params is equal to or contains the string ``'bias'``, otherwise
    return the unmodified tensor ``bias``.'''
    if zero_params is None:
        zero_params = []
    if bias is not None and (zero_params == 'bias' or 'bias' in zero_params):
        return torch.zeros_like(bias)
    return bias


@replicates(RULES_LINEAR, Epsilon, epsilon=1e-6)
@replicates(RULES_LINEAR, Epsilon, epsilon=1e-6, zero_params='bias')
@replicates(RULES_LINEAR, Epsilon, epsilon=1.0)
@replicates(RULES_LINEAR, Epsilon, epsilon=1.0, zero_params='bias')
@replicates(RULES_LINEAR, Norm)
@matrix_form
def rule_epsilon(weight, bias, input, relevance, epsilon=1e-6, zero_params=None):
    '''Replicates the Epsilon rule.'''
    bias = zero_bias(zero_params, bias)
    return input * ((relevance / stabilize(input @ weight.t() + bias, epsilon)) @ weight)


@replicates(RULES_LINEAR, ZPlus)
@replicates(RULES_LINEAR, ZPlus, zero_params='bias')
@matrix_form
def rule_zplus(weight, bias, input, relevance, zero_params=None):
    '''Replicates the ZPlus rule.'''
    bias = zero_bias(zero_params, bias)
    wplus = weight.clamp(min=0)
    wminus = weight.clamp(max=0)
    xplus = input.clamp(min=0)
    xminus = input.clamp(max=0)
    zval = xplus @ wplus.t() + xminus @ wminus.t() + bias.clamp(min=0)
    rfac = relevance / stabilize(zval)
    return xplus * (rfac @ wplus) + xminus * (rfac @ wminus)


@replicates(RULES_LINEAR, Gamma, gamma=0.25)
@replicates(RULES_LINEAR, Gamma, gamma=0.25, zero_params='bias')
@replicates(RULES_LINEAR, Gamma, gamma=0.5)
@replicates(RULES_LINEAR, Gamma, gamma=0.5, zero_params='bias')
@matrix_form
def rule_gamma(weight, bias, input, relevance, gamma, zero_params=None):
    '''Replicates the Gamma rule.'''
    output = input @ weight.t() + bias
    bias = zero_bias(zero_params, bias)
    pinput = input.clamp(min=0)
    ninput = input.clamp(max=0)
    pwgamma = weight + weight.clamp(min=0) * gamma
    nwgamma = weight + weight.clamp(max=0) * gamma
    pbgamma = bias + bias.clamp(min=0) * gamma
    nbgamma = bias + bias.clamp(max=0) * gamma

    pgrad_out = (relevance / stabilize(pinput @ pwgamma.t() + ninput @ nwgamma.t() + pbgamma)) * (output > 0.)
    positive = pinput * (pgrad_out @ pwgamma) + ninput * (pgrad_out @ nwgamma)

    ngrad_out = (relevance / stabilize(pinput @ nwgamma.t() + ninput @ pwgamma.t() + nbgamma)) * (output < 0.)
    negative = pinput * (ngrad_out @ nwgamma) + ninput * (ngrad_out @ pwgamma)

    return positive + negative


@replicates(RULES_LINEAR, AlphaBeta, alpha=2.0, beta=1.0)
@replicates(RULES_LINEAR, AlphaBeta, alpha=1.0, beta=0.0, zero_params='bias')
@replicates(RULES_LINEAR, AlphaBeta, alpha=2.0, beta=1.0)
@replicates(RULES_LINEAR, AlphaBeta, alpha=1.0, beta=0.0, zero_params='bias')
@matrix_form
def rule_alpha_beta(weight, bias, input, relevance, alpha, beta, zero_params=None):
    '''Replicates the AlphaBeta rule.'''
    bias = zero_bias(zero_params, bias)
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


@replicates(RULES_LINEAR, ZBox, low=-3.0, high=3.0)
@replicates(RULES_LINEAR, ZBox, low=-3.0, high=3.0, zero_params='bias')
@matrix_form
def rule_zbox(weight, bias, input, relevance, low, high, zero_params=None):
    '''Replicates the ZBox rule.'''
    wplus = weight.clamp(min=0)
    wminus = weight.clamp(max=0)
    low = torch.tensor(low).expand_as(input).to(input)
    high = torch.tensor(high).expand_as(input).to(input)
    zval = input @ weight.t() - low @ wplus.t() - high @ wminus.t()
    rfac = relevance / stabilize(zval)
    return input * (rfac @ weight) - low * (rfac @ wplus) - high * (rfac @ wminus)


@replicates(RULES_LINEAR, WSquare)
@replicates(RULES_LINEAR, WSquare, zero_params='bias')
@matrix_form
def rule_wsquare(weight, bias, input, relevance, zero_params=None):
    '''Replicates the WSquare rule.'''
    bias = zero_bias(zero_params, bias)
    wsquare = weight ** 2
    zval = torch.ones_like(input) @ wsquare.t() + bias ** 2
    rfac = relevance / stabilize(zval)
    return rfac @ wsquare


@replicates(RULES_LINEAR, Flat)
@flat_module_params
@matrix_form
def rule_flat(wflat, bias, input, relevance):
    '''Replicates the Flat rule.'''
    zval = torch.ones_like(input) @ wflat.t()
    rfac = relevance / stabilize(zval)
    return rfac @ wflat


@replicates(RULES_SIMPLE, Pass)
def rule_pass(module, input, relevance):
    '''Replicates the Pass rule.'''
    return relevance


@replicates(RULES_SIMPLE, ReLUDeconvNet)
def rule_relu_deconvnet(module, input, relevance):
    '''Replicates the ReLUDeconvNet rule.'''
    return relevance.clamp(min=0)


@replicates(RULES_SIMPLE, ReLUGuidedBackprop)
@with_grad
def rule_relu_guidedbackprop(gradient, input, relevance):
    '''Replicates the ReLUGuidedBackprop rule.'''
    return gradient * (relevance > 0.)


@pytest.fixture(scope='session', params=RULES_LINEAR)
def rule_pair_linear(request):
    '''Fixture to supply ``RULES_LINEAR``.'''
    return request.param


@pytest.fixture(scope='session', params=RULES_SIMPLE)
def rule_pair_simple(request):
    '''Fixture to supply ``RULES_SIMPLE``.'''
    return request.param


def compare_rule_pair(module, data, rule_pair):
    '''Compare rules with their replicated versions.'''
    rule_hook, rule_replicated = rule_pair

    input = data.clone().requires_grad_()
    handle = rule_hook().register(module)
    try:
        output = module(input)
        relevance_hook, = torch.autograd.grad(output, input, grad_outputs=output)
    finally:
        handle.remove()

    relevance_replicated = rule_replicated(module, input, output)

    assert torch.allclose(relevance_hook, relevance_replicated, atol=1e-5)


def test_linear_rule(module_linear, data_linear, rule_pair_linear):
    '''Test whether replicated and original implementations of rules for linear layers agree.'''
    compare_rule_pair(module_linear, data_linear, rule_pair_linear)


def test_simple_rule(module_simple, data_simple, rule_pair_simple):
    '''Test whether replicated and original implementations of rules for simple layers agree.'''
    compare_rule_pair(module_simple, data_simple, rule_pair_simple)


def test_alpha_beta_invalid_values():
    '''Test whether AlphaBeta raises ValueErrors for negative alpha/beta or when alpha - beta is not equal to 1.'''
    with pytest.raises(ValueError):
        AlphaBeta(alpha=-1.)
    with pytest.raises(ValueError):
        AlphaBeta(beta=-1.)
    with pytest.raises(ValueError):
        AlphaBeta(alpha=1., beta=1.)


@pytest.mark.parametrize('params', [None, 'weight', ['weight'], 'bias', ['bias'], ['weight', 'bias']])
def test_zero_bias(params):
    '''Test whether zero_bias correctly appends 'bias' to the zero_params list/str used for ParamMod.'''
    result = name_zero_bias(params)
    assert isinstance(result, list)
    assert 'bias' in result
