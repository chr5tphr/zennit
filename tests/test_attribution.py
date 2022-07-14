'''Tests for Attributors.'''
from functools import partial
from itertools import product

import pytest
import torch

from zennit.attribution import Gradient, IntegratedGradients, SmoothGrad, Occlusion, occlude_independent


class IdentityLogger(torch.nn.Module):
    '''Helper-Module to log input tensors.'''
    def __init__(self):
        super().__init__()
        self.tensors = []

    def forward(self, input):
        '''Clone input, append to self.tensors and return the cloned tensor.'''
        self.tensors.append(input.clone())
        return self.tensors[-1]


def test_gradient_attributor_composite(
    data_linear, model_simple, model_simple_output, any_composite, grad_outputs_func
):
    '''Test whether composite context and attributor match for Gradient.'''
    with any_composite.context(model_simple) as module:
        data = data_linear.detach().requires_grad_()
        output_context = module(data)
        grad_outputs = grad_outputs_func(output_context)
        grad_context, = torch.autograd.grad(output_context, data, grad_outputs)

    with Gradient(model=model_simple, composite=any_composite, attr_output=grad_outputs_func) as attributor:
        output_attributor, grad_attributor = attributor(data_linear)

    assert torch.allclose(output_context, output_attributor)
    assert torch.allclose(grad_context, grad_attributor)
    assert torch.allclose(model_simple_output, output_attributor)


@pytest.mark.parametrize('use_const,use_call,use_init', product(*[[True, False]] * 3))
def test_gradient_attributor_output_fn(data_simple, grad_outputs_func, use_const, use_call, use_init):
    '''Test whether attributors' attr_output supports functions, constants and None in any of supplied or not supplied
    for each the attributor initialization and the call.
    '''
    model = IdentityLogger()

    attr_output = grad_outputs_func(data_simple) if use_const else grad_outputs_func
    init_attr_output = attr_output if use_init else None
    call_attr_output = attr_output if use_call else None

    with Gradient(model=model, attr_output=init_attr_output) as attributor:
        _, grad = attributor(data_simple, attr_output=call_attr_output)

    if (use_call or use_init):
        expected_grad = grad_outputs_func(data_simple)
    else:
        # the identity is the default attr_output
        expected_grad = data_simple

    assert torch.allclose(expected_grad, grad), 'Attributor output function gradient mismatch!'


def test_gradient_attributor_output_fn_precedence(data_simple):
    '''Test whether the gradient attributor attr_output at call is prefered when it is supplied at both initialization
    and call.
    '''
    model = IdentityLogger()

    init_attr_output = torch.ones_like
    call_attr_output = torch.zeros_like

    with Gradient(model=model, attr_output=init_attr_output) as attributor:
        _, grad = attributor(data_simple, attr_output=call_attr_output)

    expected_grad = call_attr_output(data_simple)
    assert torch.allclose(expected_grad, grad), 'Attributor output function precedence mismatch!'


def test_smooth_grad_single(data_linear, model_simple, model_simple_output, model_simple_grad):
    '''Test whether SmoothGrad with a single iteration is equal to the gradient.'''
    with SmoothGrad(model=model_simple, noise_level=0.1, n_iter=1) as attributor:
        output, grad = attributor(data_linear)

    assert torch.allclose(model_simple_grad, grad)
    assert torch.allclose(model_simple_output, output)


@pytest.mark.parametrize('noise_level', [0.0, 0.1, 0.3, 0.5])
def test_smooth_grad_distribution(data_simple, noise_level):
    '''Test whether the SmoothGrad sampled distribution matches.'''
    model = IdentityLogger()

    dims = tuple(range(1, data_simple.ndim))
    noise_var = (noise_level * (data_simple.amax(dims) - data_simple.amin(dims))) ** 2
    n_iter = 100

    with SmoothGrad(model=model, noise_level=noise_level, n_iter=n_iter, attr_output=torch.ones_like) as attributor:
        _, grad = attributor(data_simple)

    assert len(model.tensors) == n_iter, 'SmootGrad iterations did not match n_iter!'

    sample_mean = sum(model.tensors) / len(model.tensors)
    sample_var = ((sum((tensor - sample_mean) ** 2 for tensor in model.tensors) / len(model.tensors))).mean(dims)

    assert torch.allclose(sample_var, noise_var, rtol=0.2), 'SmoothGrad sample variance is too high!'
    assert torch.allclose(grad, torch.ones_like(data_simple)), 'SmoothGrad of identity is wrong!'


@pytest.mark.parametrize('baseline_fn', [None, torch.zeros_like, torch.ones_like])
def test_integrated_gradients_single(data_linear, model_simple, model_simple_output, model_simple_grad, baseline_fn):
    '''Test whether IntegratedGradients with a single iteration is equal to the expected output given multiple
    baselines.
    '''
    with IntegratedGradients(model=model_simple, n_iter=1, baseline_fn=baseline_fn) as attributor:
        output, grad = attributor(data_linear)

    if baseline_fn is None:
        baseline_fn = torch.zeros_like
    expected_grad = model_simple_grad * (data_linear - baseline_fn(data_linear))

    assert torch.allclose(expected_grad, grad), 'Gradient mismatch for IntegratedGradients!'
    assert torch.allclose(model_simple_output, output), 'Output mismatch for IntegratedGradients!'


def test_integrated_gradients_path(data_simple):
    '''Test whether IntegratedGradients with a single iteration and a zero-baseline is equal to the input times the
    gradient.
    '''
    model = IdentityLogger()

    dims = tuple(range(1, data_simple.ndim))
    n_iter = 100
    with IntegratedGradients(model=model, n_iter=n_iter, attr_output=torch.ones_like) as attributor:
        _, grad = attributor(data_simple)

    assert len(model.tensors) == n_iter, 'IntegratedGradients iterations did not match n_iter!'

    data_simple_norm = data_simple / (data_simple ** 2).sum(dim=dims, keepdim=True) ** .5
    assert all(
        torch.allclose(step / (step ** 2).sum(dim=dims, keepdim=True) ** .5, data_simple_norm)
        for step in model.tensors
    ), 'IntegratedGradients segments do not lie on path!'
    assert torch.allclose(data_simple, grad), 'IntegratedGradients of identity is wrong!'


@pytest.mark.parametrize('window,stride', zip([1, 2, 4, (1,), (2,), (4,)], [1, 2, 4, (1,), (2,), (4,)]))
def test_occlusion_disjunct(data_simple, window, stride):
    '''Function to test whether the inputs used for disjunct occlusion windows are correct.'''
    model = IdentityLogger()

    # delete everything except the window
    occlusion_fn = partial(occlude_independent, fill_fn=torch.zeros_like, invert=False)

    with Occlusion(model=model, window=window, stride=stride, occlusion_fn=occlusion_fn) as attributor:
        attributor(data_simple)

    # omit final pass for full output
    reconstruct = sum(model.tensors[:-1])
    assert torch.allclose(data_simple, reconstruct), 'Disjunct occlusion does not sum to original input!'


@pytest.mark.parametrize(
    'fill_fn,invert', [
        (None, False),
        (torch.zeros_like, False),
        (torch.zeros_like, True),
        (torch.ones_like, True),
    ]
)
def test_occlusion_single(data_linear, model_simple, model_simple_output, grad_outputs_func, fill_fn, invert):
    '''Function to test whether the inputs used for a full occlusion window are correct.'''
    window, stride = [data_linear.shape] * 2
    if fill_fn is None:
        # setting when no occlusion_fn is supplied
        occlusion_fn = None
        fill_fn = torch.zeros_like
    else:
        occlusion_fn = partial(occlude_independent, fill_fn=fill_fn, invert=invert)

    identity_logger = IdentityLogger()
    model = torch.nn.Sequential(identity_logger, model_simple)

    with Occlusion(
        model=model,
        window=window,
        stride=stride,
        attr_output=grad_outputs_func,
        occlusion_fn=occlusion_fn,
    ) as attributor:
        output, score = attributor(data_linear)

    expected_occluded = fill_fn(data_linear) if invert else data_linear
    expected_output = model_simple(expected_occluded)
    expected_score = grad_outputs_func(expected_output).sum(
        tuple(range(1, expected_output.ndim))
    )[(slice(None),) + (None,) * (data_linear.ndim - 1)].expand_as(data_linear)

    assert len(identity_logger.tensors) == 2, 'Incorrect number of forward passes for Occlusion!'
    assert torch.allclose(identity_logger.tensors[0], expected_occluded), 'Occluded input mismatch!'
    assert torch.allclose(model_simple_output, output), 'Output mismatch for Occlusion!'
    assert torch.allclose(expected_score, score), 'Scores are incorrect for Occlusion!'


@pytest.mark.parametrize('argument,container', product(
    ['window', 'stride'],
    ['monkey', {3}, ('you', 'are', 'breathtaking'), range(3), [3]]
))
def test_occlusion_stride_window_typecheck(argument, container):
    '''Test whether Occlusion raises a TypeError on incorrect types for window and stride.'''
    with pytest.raises(TypeError):
        Occlusion(model=None, **{argument: container})
