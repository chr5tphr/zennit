'''Tests for core functionality in zennit.core.'''
from itertools import product

import torch
import pytest
from helpers import nograd, prodict

from zennit.core import stabilize, expand, ParamMod, collect_leaves, Stabilizer
from zennit.core import Identity, Hook, BasicHook, RemovableHandle, RemovableHandleList, Composite


@pytest.mark.parametrize('kwargs,input,expected', [
    (
        {},
        [0., -0., 1., -1.],
        [1e-6, 1e-6, 1. + 1e-6, -1. - 1e-6]
    ), (
        {'epsilon': 1e-3, 'clip': False, 'norm_scale': False},
        [0., -0., 1., -1.],
        [1e-3, 1e-3, 1. + 1e-3, -1. - 1e-3]
    ), (
        {'epsilon': 1., 'clip': False, 'norm_scale': False},
        [0., -0., 1., -1.],
        [1., 1., 2., -2.]
    ), (
        {'epsilon': 1e-6, 'clip': True, 'norm_scale': False},
        [0., -0., 1., -1.],
        [1e-6, 1e-6, 1., -1.]
    ), (
        {'epsilon': 1e-3, 'clip': True, 'norm_scale': False},
        [0., -0., 1., -1.],
        [1e-3, 1e-3, 1., -1.]
    ), (
        {'epsilon': 1., 'clip': True, 'norm_scale': False},
        [0., -0., 1., -1.],
        [1., 1., 1., -1.]
    ), (
        {'epsilon': 1e-6, 'clip': False, 'norm_scale': True},
        [0., -0., 2., -2.],
        [1.4142e-6, 1.4142e-6, 2. + 1.4142e-6, -2. - 1.4142e-6]
    ), (
        {'epsilon': 1e-3, 'clip': False, 'norm_scale': True},
        [0., -0., 2., -2.],
        [1.4142e-3, 1.4142e-3, 2. + 1.4142e-3, -2. - 1.4142e-3]
    ), (
        {'epsilon': 1., 'clip': False, 'norm_scale': True},
        [0., -0., 2., -2.],
        [1.4142, 1.4142, 3.4142, -3.4142]
    ), (
        {'epsilon': 1e-6, 'clip': True, 'norm_scale': True},
        [0., -0., 2., -2.],
        [1.4142e-6, 1.4142e-6, 2., -2.]
    ), (
        {'epsilon': 1e-3, 'clip': True, 'norm_scale': True},
        [0., -0., 2., -2.],
        [1.4142e-3, 1.4142e-3, 2., -2.]
    ), (
        {'epsilon': 1., 'clip': True, 'norm_scale': True},
        [0., -0., 2., -2.],
        [1.4142, 1.4142, 2., -2.]
    ),
])
def test_stabilize(kwargs, input, expected):
    '''Test whether stabilize produces the expected outputs given some inputs.'''
    input_tensor = torch.tensor(input, dtype=torch.float64)
    output = stabilize(input_tensor, dim=0, **kwargs)
    expected_tensor = torch.tensor(expected, dtype=torch.float64)
    assert torch.allclose(expected_tensor, output)


@pytest.mark.parametrize('kwargs', prodict(
    epsilon=[1e-6, 1e-3, 1.],
    clip=[True, False],
    norm_scale=[True, False],
    dim=[None, (0,), (1,), (0, 1)]
))
def test_stabilizer_match(kwargs, data_simple):
    '''Test whether stabilize and Stabilizer produce the same output.'''
    stabilizer = Stabilizer(**kwargs)
    stabilizer_out = stabilizer(data_simple)
    stabilize_out = stabilize(data_simple, **kwargs)
    assert torch.allclose(stabilizer_out, stabilize_out)


@pytest.mark.parametrize('value', [1., 1, Stabilizer(epsilon=1.), lambda x: x])
def test_stabilizer_ensure(value):
    '''Test whether Stabilizer.ensure produces a stabilizer with the correct epsilon, or returns callables as-is.'''
    ensured = Stabilizer.ensure(value)
    assert not isinstance(value, float) or isinstance(ensured, Stabilizer) and ensured.epsilon == value
    assert not callable(value) or value is ensured


@pytest.mark.parametrize('value', [None, 'wow'])
def test_stabilizer_ensure_fail(value):
    '''Test whether Stabilizer.ensure fails on unsupported types.'''
    with pytest.raises(TypeError):
        Stabilizer.ensure(value)


@pytest.mark.parametrize('input_shape,target_shape,cut_batch_dim', [
    ((), (), False),
    ((), (2,), False),
    ((2,), (2, 2, 2), False),
    ((1, 2), (2, 2, 2), False),
    ((2, 1, 2), (2, 2, 2), None),
    ((2, 1, 2), (2, 2, 2), False),
    ((3, 1, 2), (2, 2, 2), True),
])
def test_expand_shapes(input_shape, target_shape, cut_batch_dim):
    '''Test whether expand produces correct shapes.'''
    kwargs = {} if cut_batch_dim is None else {'cut_batch_dim': cut_batch_dim}
    input = torch.zeros(input_shape)
    output = expand(input, target_shape, **kwargs)
    assert target_shape == output.shape


def test_expand_non_tensor():
    '''Test whether expand supports non-tensor input.'''
    target_shape = (2, 2, 2)
    input = [0., 1.]
    output = expand(input, target_shape)
    assert target_shape == output.shape


@pytest.mark.parametrize('input_shape,target_shape', [
    ((2,), (3,)),
    ((3,), (2, 2, 2)),
    ((3, 1, 2), (2, 2, 2)),
])
def test_expand_shape_invalid(input_shape, target_shape):
    '''Test whether expand raises RuntimeError on invalid shapes.'''
    input = torch.zeros(input_shape)
    with pytest.raises(RuntimeError):
        expand(input, target_shape)


@pytest.mark.parametrize('param_keys,require_params,zero_params,bias', [
    *product(
        [None, ['weight'], ['bias'], ['weight', 'bias']],
        [True],
        [[], ['weight'], ['bias'], ['weight', 'bias']],
        [False, True]
    ),
    *product([['weight', 'bias', 'beight', 'wias']], [False], [[]], [False, True]),
])
def test_param_mod(param_keys, require_params, zero_params, bias):
    '''Test whether ParamMod correctly changes and restores parameters of a module.'''
    module = nograd(torch.nn.Linear(2, 2, bias=bias))
    mod_param_keys = param_keys
    if param_keys is None:
        param_keys = [name for name, _ in module.named_parameters(recurse=False)]

    for key in param_keys:
        if getattr(module, key, None) is not None:
            getattr(module, key).data = torch.full_like(getattr(module, key), 0.5)

    param_mod = ParamMod(
        lambda x, _: torch.ones_like(x),
        param_keys=mod_param_keys,
        require_params=require_params,
        zero_params=zero_params,
    )

    with param_mod(module) as modified:
        for key in param_keys:
            if getattr(module, key, None) is not None:
                if key not in zero_params:
                    assert torch.all(getattr(modified, key) == 1.0), f'Parameter \'{key}\' was not modified!'
                else:
                    assert torch.all(getattr(modified, key) == 0.0), f'Parameter \'{key}\' was not set to zero!'

    for key in param_keys:
        if getattr(module, key, None) is not None:
            assert torch.all(getattr(module, key) == 0.5), f'Parameter \'{key}\' was not restored!'


def test_param_mod_required_missing():
    '''Test wether ParamMod raises a RuntimeError when it is missing parameters when require_params=True.'''
    module = torch.nn.Module()
    param_keys = ['beight', 'wias']
    param_mod = ParamMod(lambda x, _: torch.ones_like(x), param_keys=param_keys, require_params=True)
    with pytest.raises(RuntimeError):
        with param_mod(module):
            pass


@pytest.mark.parametrize('modifier', [lambda x: x, ParamMod(lambda x: x)])
def test_param_mod_ensure(modifier):
    '''Test wether ParamMod.ensure returns the original object if it is an instance of ParamMod, or a new instance of
    ParamMod if it is not an instance of ParamMod and callable.'''
    result = ParamMod.ensure(modifier)
    assert isinstance(result, ParamMod)
    assert not isinstance(modifier, ParamMod) or result is modifier


def test_param_mod_ensure_unknown_type():
    '''Test wether ParamMod.ensure raises a TypeError when it is called on a non-callable object that is not of type
    ParamMod.'''
    with pytest.raises(TypeError):
        ParamMod.ensure(None)


def test_collect_leaves_dummy():
    '''Test whether collect leaves correctly collects all leaves of an example model.'''
    modules = []

    def add(module):
        modules.append(module)
        return module

    class DummyModel(torch.nn.Module):
        '''Dummy model to produce a hierarchy of modules.'''
        def __init__(self):
            super().__init__()
            self.linear = add(torch.nn.Linear(2, 2))
            self.relu1 = add(torch.nn.ReLU())
            self.more = torch.nn.Sequential(
                add(torch.nn.Linear(2, 2)),
                add(torch.nn.ReLU()),
                add(torch.nn.Linear(2, 2)),
                add(torch.nn.ReLU()),
                add(torch.nn.Linear(2, 2)),
            )

    model = DummyModel()
    leaves = collect_leaves(model)
    assert all(a is b for a, b in zip(modules, leaves))


def test_collect_leaves_no_children():
    '''Test whether collect leaves correctly yields the input if it has no children.'''
    module = torch.nn.Module()
    leaf, = collect_leaves(module)
    assert leaf is module


def test_identity():
    '''Test whether Identity gives the correct output and grad and produces a grad_fn'''
    data = torch.randn(5, requires_grad=True)
    output, = Identity.apply(data)
    assert torch.allclose(data, output)
    assert hasattr(output, 'grad_fn')

    grad, = torch.autograd.grad(output, data, data)
    assert torch.allclose(data, grad)


def test_hook_grad():
    '''Test whether the Hook is correctly called in the forward and gradient passes.'''
    called = set()

    linear = torch.nn.Linear(2, 2)
    data = torch.randn(1, 2, requires_grad=True)
    grad_out = torch.ones(1, 2)

    class DummyHook(Hook):
        '''Dummy subclass of Hook to check whether forward and backward are called correctly.'''
        def forward(self, module, input, output):
            '''Check whether forward is called, and if the arguments are as expected.'''
            called.add('forward')
            assert module is linear
            assert torch.allclose(data, input[0])
            assert data is not input

        def backward(self, module, grad_input, grad_output):
            '''Check whether backward is called, and if the arguments are as expected.'''
            called.add('backward')
            assert grad_output[0] is grad_out

    hook = DummyHook()
    handles = hook.register(linear)
    try:
        out = linear(data)
        assert 'forward' in called
        assert 'backward' not in called
        torch.autograd.grad(out, data, grad_out)
        assert 'backward' in called
        assert 'grad_output' in hook.stored_tensors
        assert hook.stored_tensors['grad_output'][0] is grad_out
    finally:
        handles.remove()


def test_hook_no_grad():
    '''Test whether Hook's backward is never called when no gradient is required.'''
    called = set()

    linear = torch.nn.Linear(2, 2)
    data = torch.randn(1, 2, requires_grad=False)

    class DummyHook(Hook):
        '''Dummy subclass of Hook to check whether forward and backward are called.'''
        def forward(self, module, input, output):
            '''Check whether forward is called.'''
            called.add('forward')

        def backward(self, module, grad_input, grad_output):
            '''Check whether backward is called.'''
            called.add('backward')

    hook = DummyHook()
    handles = hook.register(linear)
    try:
        linear(data)
        assert 'forward' in called
        assert 'backward' not in called
    finally:
        handles.remove()


def test_hook_copy():
    '''Test whether copying a subclass of Hook creates an instance of the appropriate type.'''

    class DummyHook(Hook):
        '''Dummy subclass of Hook to check for the same class when copying.'''

    original = DummyHook()
    copy = original.copy()
    assert original is not copy
    assert isinstance(copy, DummyHook)


def test_hook_tuple():
    '''Test whether pre- and post_forward handle non-tuple inputs and tuple outputs.'''
    hook = Hook()
    module = None
    data = torch.ones(2)

    pre_out = hook.pre_forward(module, data)
    assert pre_out is data
    post_out = hook.post_forward(module, data, (data,))
    assert post_out is data


def test_basic_hook_default():
    '''Check whether BasicHook with default parameters runs as expected.'''
    linear = torch.nn.Linear(2, 2)
    data = torch.randn(1, 2, requires_grad=True)
    grad_out = torch.ones(1, 2)

    hook = BasicHook()
    handles = hook.register(linear)
    try:
        out = linear(data)
        assert 'input' in hook.stored_tensors
        assert hook.stored_tensors['input'][0] is not data
        assert torch.allclose(hook.stored_tensors['input'][0], data)
        torch.autograd.grad(out, data, grad_out)
        assert 'grad_output' in hook.stored_tensors
        assert hook.stored_tensors['grad_output'][0] is grad_out
    finally:
        handles.remove()


def test_basic_hook_custom():
    '''Check whether BasicHook with custom parameters runs as expected.'''
    linear = torch.nn.Linear(2, 2, bias=True)
    data = torch.randn(1, 2, requires_grad=True)
    grad_out = torch.ones(1, 2)

    called = set()

    def assert_params(obj, name=None):
        '''Assert whether param modifier function is called as expected.'''
        called.add('params')
        if name == 'weight':
            assert torch.allclose(obj, linear.weight.data)
        elif name == 'bias':
            assert torch.allclose(obj, linear.bias.data)
        return obj

    def assert_inputs(obj):
        '''Assert whether input modifier function is called as expected.'''
        called.add('input')
        assert torch.allclose(obj, data)
        return obj

    def assert_outputs(obj):
        '''Assert whether output modifier function is called as expected.'''
        called.add('output')
        return obj

    def assert_gradient(out_grad, outputs):
        '''Assert whether gradient mapper function is called as expected.'''
        called.add('output')
        called.add('gradient')
        assert out_grad is grad_out
        return [out_grad for output in outputs]

    def assert_reducer(inputs, gradients):
        '''Assert whether reducer function is called as expected.'''
        called.add('reducer')
        assert len(inputs) == len(gradients)
        assert torch.allclose(inputs[0], data)
        return sum(input * gradient for input, gradient in zip(inputs, gradients))

    hook = BasicHook(
        input_modifiers=[assert_inputs],
        param_modifiers=[
            ParamMod(
                assert_params,
                param_keys=['weight', 'bias'],
                require_params=True,
            )
        ],
        output_modifiers=[assert_outputs],
        gradient_mapper=assert_gradient,
        reducer=assert_reducer,
    )

    call_events = ('input', 'params', 'output', 'gradient', 'reducer')
    handles = hook.register(linear)
    try:
        out = linear(data)
        assert 'input' in hook.stored_tensors
        assert hook.stored_tensors['input'][0] is not data
        assert torch.allclose(hook.stored_tensors['input'][0], data)
        for key in call_events:
            assert key not in called, f'Event {key} was prematurely called in hook!'
        torch.autograd.grad(out, data, grad_out)
        assert 'grad_output' in hook.stored_tensors
        assert hook.stored_tensors['grad_output'][0] is grad_out
        for key in call_events:
            assert key in called, f'Event {key} was not called in hook!'
    finally:
        handles.remove()


def test_basic_hook_copy():
    '''Test whether BasicHook.copy copies the Hook correctly.'''
    hook = BasicHook(
        input_modifiers=[lambda obj, name: obj],
        param_modifiers=[
            ParamMod(
                (lambda obj, name: obj),
                param_keys=['weight', 'bias'],
                require_params=True,
                zero_params=['bias'],
            ),
        ],
        output_modifiers=[lambda obj, name: obj],
        gradient_mapper=(lambda out_grad, outputs: [out_grad for output in outputs]),
        reducer=(lambda inputs, gradients: [input * gradient for input, gradient in zip(inputs, gradients)]),
    )
    copy = hook.copy()

    attributes = (
        'input_modifiers',
        'param_modifiers',
        'output_modifiers',
        'reducer',
        'gradient_mapper',
    )

    assert copy is not hook
    for key in attributes:
        assert getattr(copy, key) is getattr(hook, key)


def test_removable_handle_deleted():
    '''Test whether RemovableHandle does nothing when attempting to call remove on a garbage collected object.'''

    called = set()

    class Dummy:
        '''Dummy object with a remove function which can be used for a RemovableHandle.'''
        @staticmethod
        def remove():
            '''Dummy remove function which tracks whether it has been called.'''
            called.add('remove')

    obj = Dummy()
    handle = RemovableHandle(obj)
    del obj
    handle.remove()
    assert 'remove' not in called


def test_composite_context():
    '''Test whether the composite context correctly uses the module map.'''
    linear = torch.nn.Linear(2, 2)
    called = set()

    class DummyHandle:
        '''Dummy handle to track whether it has been removed.'''
        @staticmethod
        def remove():
            '''Track whether this functions has been called.'''
            called.add('remove')

    handle = DummyHandle()

    class DummyHook:
        '''Dummy hook to track whether it has been registered correctly, and copied.'''
        @staticmethod
        def register(child):
            '''Assert whether child is the expected module and track that it has been called.'''
            assert child is linear
            called.add('register')
            return handle

        def copy(self):
            '''Return self an track that it has been called.'''
            called.add('copy')
            return self

    def module_map(ctx, name, child):
        '''Module map function to check whether all arguments are as expected which returns a DummyHook.'''
        assert child is linear
        assert name == ''
        assert isinstance(ctx, dict)
        return DummyHook()

    composite = Composite(module_map=module_map)

    with composite.context(linear):
        assert isinstance(composite.handles, RemovableHandleList)
        assert handle in composite.handles
        assert 'register' in called
        assert 'copy' in called
        assert 'remove' not in called
    assert 'remove' in called


def test_composite_empty():
    '''Test whether an empty composite with canonizers uses the canonizers as expected.'''
    linear = torch.nn.Linear(2, 2)
    called = set()

    class DummyHandle:
        '''Dummy handle to track whether its remove function has been called.'''
        @staticmethod
        def remove():
            '''Track whether the remove function was called.'''
            called.add('remove')

    handle = DummyHandle()

    class DummyCanonizer:
        '''Dummy canonizer to track whether its apply function has been called.'''
        @staticmethod
        def apply(module):
            '''Track whether the apply function has been called and return a dummy handle.'''
            called.add('apply')
            return [handle]

    canonizer = DummyCanonizer()
    composite = Composite(canonizers=[canonizer])

    with composite.context(linear):
        assert handle in composite.handles
        assert 'apply' in called
        assert 'remove' not in called
    assert 'remove' in called
