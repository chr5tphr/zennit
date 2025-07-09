# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/rules.py
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
'''Rules based on Hooks'''
import torch

from .core import Hook, BasicHook, Stabilizer, expand, ParamMod


def zero_bias(zero_params=None):
    '''Add `'bias'` to `zero_params`, where `zero_params` is a string or a list of strings.

    Parameters
    ----------
    zero_params: str or list of str, optional
        Name or names, to which ``'bias'`` should be added.

    Returns
    -------
    list of str
        Supplied ``zero_params``, with the string ``'bias'`` appended.
    '''
    if zero_params is None:
        return ['bias']
    if isinstance(zero_params, str):
        zero_params = [zero_params]
    if 'bias' in zero_params:
        return zero_params
    return list(zero_params) + ['bias']


class ClampMod(ParamMod):
    '''ParamMod to clamp module parameters.

    Parameters
    ----------
    min: float or None, optional
        Minimum float value for which the parameters should be clamped, or None if no clamping should be done.
    max: float or None, optional
        Maximum float value for which the parameters should be clamped, or None if no clamping should be done.
    **kwargs: dict[str, object]
        Additional keyword arguments used for :py:class:`ParamMod`.
    '''
    def __init__(self, min=None, max=None, **kwargs):
        def modifier(param, name):
            return param.clamp(min=min, max=max)
        super().__init__(modifier, **kwargs)


class GammaMod(ParamMod):
    '''ParamMod to modify module parameters as in the Gamma rule. Adds the scaled, clamped parameters to the parameter
    itself.

    Parameters
    ----------
    gamma: float, optional
        Gamma scaling parameter, by which the clamped parameters are multiplied.
    min: float or None, optional
        Minimum float value for which the parameters should be clamped, or None if no clamping should be done.
    max: float or None, optional
        Maximum float value for which the parameters should be clamped, or None if no clamping should be done.
    **kwargs: dict[str, object]
        Additional keyword arguments used for :py:class:`ParamMod`.
    '''
    def __init__(self, gamma=0.25, min=None, max=None, **kwargs):
        def modifier(param, name):
            return param + gamma * param.clamp(min=min, max=max)
        super().__init__(modifier, **kwargs)


class NoMod(ParamMod):
    '''ParamMod that does not modify the parameters. Allows other modification flags.

    Parameters
    ----------
    **kwargs: dict[str, object]
        Additional keyword arguments used for :py:class:`ParamMod`.
    '''
    def __init__(self, **kwargs):
        super().__init__((lambda param, _: param), **kwargs)


class Epsilon(BasicHook):
    '''LRP Epsilon rule :cite:p:`bach2015pixel`.
    Setting ``(epsilon=0)`` produces the LRP-0 rule :cite:p:`bach2015pixel`.
    LRP Epsilon is most commonly used in middle layers, LRP-0 is most commonly used in upper layers
    :cite:p:`montavon2019layer`.
    Sometimes higher values of ``epsilon`` are used, therefore it is not always only a stabilizer value.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter. If ``epsilon`` is a float, it will be added to the denominator with the same sign as
        each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected, of
        which the output corresponds to the stabilized denominator. Note that this is called ``stabilizer`` for all
        other rules.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    '''
    def __init__(self, epsilon=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(epsilon)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class Gamma(BasicHook):
    '''Generalized LRP Gamma rule :cite:p:`montavon2019layer,andeol2021learning`.
    The gamma parameter scales the added positive/negative parts of the weights.
    The original Gamma rule :cite:p:`montavon2019layer` may only be used with positive inputs.
    The generalized version is equivalent to the original Gamma when there are only positive inputs, but may also be
    used for negative inputs.

    Parameters
    ----------
    gamma: float, optional
        Multiplier for added positive weights.
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    '''
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):
        mod_kwargs = {'zero_params': zero_params}
        mod_kwargs_nobias = {'zero_params': zero_bias(zero_params)}
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input,
            ],
            param_modifiers=[
                GammaMod(gamma, min=0., **mod_kwargs),
                GammaMod(gamma, max=0., **mod_kwargs_nobias),
                GammaMod(gamma, max=0., **mod_kwargs),
                GammaMod(gamma, min=0., **mod_kwargs_nobias),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    output * out_grad / stabilizer_fn(denom)
                    for output, denom in (
                        [(outputs[4] > 0., sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0., sum(outputs[2:4]))] * 2
                    )
                ] + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(input * gradient for input, gradient in zip(inputs[:4], gradients[:4]))
            ),
        )


class ZPlus(BasicHook):
    '''LRP ZPlus rule :cite:p:`bach2015pixel,montavon2017explaining`.
    It is the same as using :py:class:`~zennit.rules.AlphaBeta` with ``(alpha=1, beta=0)``

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.

    Notes
    -----
    Note that the original deep Taylor Decomposition (DTD) specification of the ZPlus Rule
    :cite:p:`montavon2017explaining` only considers positive inputs, as they are used in ReLU Networks.
    This implementation is effectively alpha=1, beta=0, where negative inputs are allowed.
    '''
    def __init__(self, stabilizer=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
            ],
            param_modifiers=[
                ClampMod(min=0., zero_params=zero_params),
                ClampMod(max=0., zero_params=zero_bias(zero_params)),
            ],
            output_modifiers=[lambda output: output] * 2,
            gradient_mapper=(lambda out_grad, outputs: [out_grad / stabilizer_fn(sum(outputs))] * 2),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0] + inputs[1] * gradients[1]),
        )


class AlphaBeta(BasicHook):
    '''LRP AlphaBeta rule :cite:p:`bach2015pixel`.
    The AlphaBeta rule weights positive (alpha) and negative (beta) contributions.
    Most common parameters are ``(alpha=1, beta=0)`` and ``(alpha=2, beta=1)``.
    It is most commonly used for lower layers :cite:p:`montavon2019layer`.

    Parameters
    ----------
    alpha: float, optional
        Multiplier for the positive output term.
    beta: float, optional
        Multiplier for the negative output term.
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.

    Raises
    ------
    ValueError
        If `alpha` and `beta` do not sum to one, or are negative.

    '''
    def __init__(self, alpha=2., beta=1., stabilizer=1e-6, zero_params=None):
        if alpha < 0 or beta < 0:
            raise ValueError("Both alpha and beta parameters must be non-negative!")
        if (alpha - beta) != 1.:
            raise ValueError("The difference of parameters alpha - beta must equal 1!")
        mod_kwargs = {'zero_params': zero_params}
        mod_kwargs_nobias = {'zero_params': zero_bias(zero_params)}
        stabilizer_fn = Stabilizer.ensure(stabilizer)

        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
            ],
            param_modifiers=[
                ClampMod(min=0., **mod_kwargs),
                ClampMod(max=0., **mod_kwargs_nobias),
                ClampMod(max=0., **mod_kwargs),
                ClampMod(min=0., **mod_kwargs_nobias),
            ],
            output_modifiers=[lambda output: output] * 4,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    out_grad / stabilizer_fn(denom)
                    for denom in ([sum(outputs[:2])] * 2 + [sum(outputs[2:])] * 2)
                ]
            ),
            reducer=(
                lambda inputs, gradients: (
                    alpha * (inputs[0] * gradients[0] + inputs[1] * gradients[1])
                    - beta * (inputs[2] * gradients[2] + inputs[3] * gradients[3])
                )
            ),
        )


class ZBox(BasicHook):
    '''LRP ZBox rule :cite:p:`montavon2017explaining`.
    The ZBox rule is intended for "boxed" input pixel space.
    Generally, the lowest and highest *possible* values are used, i.e. ``(low=0., high=1.)`` for raw image data in
    the float data type.
    Neural network inputs are often normalized to match an isotropic gaussian distribution with mean 0 and variance 1,
    which means that the lowest and highest values also need to be adapted.
    For image data, this generally happens per channel, for which case ``low`` and ``high`` can be passed as tensors
    with shape ``(1, 3, 1, 1)``, which will be broadcasted as expected.

    Parameters
    ----------
    low: :py:class:`torch.Tensor` or float
        Lowest pixel values of input. Subject to broadcasting.
    high: :py:class:`torch.Tensor` or float
        Highest pixel values of input. Subject to broadcasting.
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.

    '''
    def __init__(self, low, high, stabilizer=1e-6, zero_params=None):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        mod_kwargs = {'zero_params': zero_params}
        stabilizer_fn = Stabilizer.ensure(stabilizer)

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: expand(low, input.shape, cut_batch_dim=True).to(input),
                lambda input: expand(high, input.shape, cut_batch_dim=True).to(input),
            ],
            param_modifiers=[
                NoMod(**mod_kwargs),
                ClampMod(min=0., **mod_kwargs),
                ClampMod(max=0., **mod_kwargs),
            ],
            output_modifiers=[lambda output: output] * 3,
            gradient_mapper=(lambda out_grad, outputs: (out_grad / stabilizer_fn(sub(*outputs)),) * 3),
            reducer=(lambda inputs, gradients: sub(*(input * gradient for input, gradient in zip(inputs, gradients)))),
        )


class Pass(Hook):
    '''Unmodified pass-through rule.
    If the rule of a layer shall not be any other, is elementwise and shall not be the gradient, the `Pass` rule simply
    passes upper layer relevance through to the lower layer.
    '''
    def backward(self, module, grad_input, grad_output):
        '''Pass through the upper gradient, skipping the one for this layer.

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
        :py:obj:`torch.Tensor`
            The unmodified `grad_output`.
        '''
        return grad_output


class Norm(BasicHook):
    '''Normalize and weight by input contribution.
    This is essentially the same as the LRP :py:class:`~zennit.rules.Epsilon` rule :cite:p:`bach2015pixel` with a fixed
    epsilon only used as a stabilizer, and without the need of the attached layer to have parameters ``weight`` and
    ``bias``.

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    '''
    def __init__(self, stabilizer=1e-6):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(param_keys=[])],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class WSquare(BasicHook):
    '''LRP WSquare rule :cite:p:`montavon2017explaining`.
    It is most commonly used in the first layer when the values are not bounded :cite:p:`montavon2019layer`.

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    '''
    def __init__(self, stabilizer=1e-6, zero_params=None):
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[torch.ones_like],
            param_modifiers=[
                ParamMod((lambda param, _: param ** 2), zero_params=zero_params),
            ],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: gradients[0]),
        )


class Flat(BasicHook):
    '''LRP Flat rule :cite:p:`lapuschkin2019unmasking`.
    It is essentially the same as the LRP :py:class:`~zennit.rules.WSquare` rule, but with all parameters set to ones.

    Parameters
    ----------
    stabilizer: callable or float, optional
        Stabilization parameter. If ``stabilizer`` is a float, it will be added to the denominator with the same sign
        as each respective entry. If it is callable, a function ``(input: torch.Tensor) -> torch.Tensor`` is expected,
        of which the output corresponds to the stabilized denominator.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    '''
    def __init__(self, stabilizer=1e-6, zero_params=None):
        mod_kwargs = {'zero_params': zero_bias(zero_params), 'require_params': False}
        stabilizer_fn = Stabilizer.ensure(stabilizer)
        super().__init__(
            input_modifiers=[torch.ones_like],
            param_modifiers=[
                ParamMod((lambda param, name: torch.ones_like(param)), **mod_kwargs),
            ],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilizer_fn(outputs[0])),
            reducer=(lambda inputs, gradients: gradients[0]),
        )


class ReLUDeconvNet(Hook):
    '''DeconvNet ReLU rule :cite:p:`zeiler2014visualizing`.'''
    def backward(self, module, grad_input, grad_output):
        '''Modify ReLU gradient according to DeconvNet :cite:p:`zeiler2014visualizing`.

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
        :py:obj:`torch.Tensor`
            The modified `grad_output`.
        '''
        return (grad_output[0].clamp(min=0),)


class ReLUGuidedBackprop(Hook):
    '''GuidedBackprop ReLU rule :cite:p:`springenberg2015striving`.'''
    def backward(self, module, grad_input, grad_output):
        '''Modify ReLU gradient according to GuidedBackprop :cite:p:`springenberg2015striving`.

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
        :py:obj:`torch.Tensor`
            The modified `grad_output`.
        '''
        return (grad_input[0] * (grad_output[0] > 0.),)


class ReLUBetaSmooth(Hook):
    '''Modify ReLU gradient to smooth softplus gradient :cite:p:`dombrowski2019explanations`. Used to obtain meaningful
    surrogate gradients to compute higher order gradients with ReLUs. Equivalent to changing the gradient to be the
    (scaled) logistic function (sigmoid).

    Parameters
    ----------
    beta_smooth: float, optional
        The beta parameter for the softplus gradient (i.e. ``sigmoid(beta * input)``). Defaults to ``10``.
    '''
    def __init__(self, beta_smooth=10.):
        super().__init__()
        self.beta_smooth = beta_smooth

    def copy(self):
        '''Return a copy of this hook with the same beta parameter.

        Returns
        -------
        ReLUBetaSmooth
            A copy of this hook.
        '''
        return self.__class__(beta_smooth=self.beta_smooth)

    def forward(self, module, input, output):
        '''Remember the input for the backward pass.

        Parameters
        ----------
        module: :py:obj:`torch.nn.Module`
            The module to which this hook is attached.
        input: :py:obj:`torch.Tensor`
            The input tensor.
        output: :py:obj:`torch.Tensor`
            The output tensor.
        '''
        self.stored_tensors['input'] = input

    def backward(self, module, grad_input, grad_output):
        '''Modify ReLU gradient to the smooth softplus gradient :cite:p:`dombrowski2019explanations`.

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
        :py:obj:`torch.Tensor`
            The modified `grad_output`.
        '''
        return (torch.sigmoid(self.beta_smooth * self.stored_tensors['input'][0]) * grad_output[0],)

class TakesMost(Hook):
    '''LRP rule for max-pooling layers that distributes relevance based on a
    softmax of input contributions within each pooling window.

    Parameters
    ----------
    dimensions : int, optional
        Number of spatial dimensions of the pooling operation (1 for 1D, 2 for 2D).
        Default is 2.
    beta : float, optional
        Scaling factor for the inputs before the softmax calculation. Higher
        values lead to a harder softmax, approximating winner-takes-all.
        Default is 1.0.
    '''
    def __init__(self, dimensions=2, beta=1.0):
        super().__init__()
        self.dimensions = dimensions
        self.beta = beta
        self.pool_fn = getattr(torch.nn.functional, f'max_pool{dimensions}d')
        self.conv_fn = getattr(torch.nn.functional, f'conv{dimensions}d')
        self.conv_transpose_fn = getattr(torch.nn.functional, f'conv_transpose{dimensions}d')

    def _ensure_tuple(self, value, n):
        """Convert int to n-tuple if necessary"""
        return (value,) * n if isinstance(value, int) else value

    def copy(self):
        """Return a copy of this hook with the same beta parameter."""
        return self.__class__(dimensions=self.dimensions, beta=self.beta)

    def forward(self, module, input, output):
        self.stored_tensors['input'] = input

    def _calculate_output_padding(self, input_shape, pooled_shape, stride, kernel_size, padding, dilation):
        """Calculate output padding for transposed convolution"""
        output_padding = []
        for i in range(self.dimensions):
            target_size = input_shape[2 + i]
            pooled_size = pooled_shape[2 + i]
            s, k, p, d = stride[i], kernel_size[i], padding[i], dilation[i]
            op = target_size - ((pooled_size - 1) * s - 2 * p + d * (k - 1) + 1)
            output_padding.append(op)
        return tuple(output_padding)

    def _create_unpool_params(self, input_tensor, pooled_tensor, stride, kernel_size, padding, dilation):
        """Create common parameters for unpooling operations"""
        num_channels = input_tensor.shape[1]
        kernel_shape = (num_channels, 1) + kernel_size
        kernel = torch.ones(kernel_shape, device=input_tensor.device, dtype=input_tensor.dtype)

        output_padding = self._calculate_output_padding(
            input_tensor.shape, pooled_tensor.shape, stride, kernel_size, padding, dilation
        )

        # Create slicing for potential cropping
        slicing = [slice(None), slice(None)]  # For batch and channels
        for i in range(self.dimensions):
            slicing.append(slice(0, input_tensor.shape[2 + i]))

        return kernel, output_padding, tuple(slicing)

    def _unpool(self, pooled, kernel, stride, padding, output_padding, dilation, groups, slicing):
        """Common unpooling function"""
        result = self.conv_transpose_fn(
            pooled, weight=kernel, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation, groups=groups
        )
        return result[slicing]

    def backward(self, module, grad_input, grad_output):
        input_tensor = self.stored_tensors['input'][0]

        # Get module parameters as tuples
        stride = self._ensure_tuple(module.stride, self.dimensions)
        kernel_size = self._ensure_tuple(module.kernel_size, self.dimensions)
        padding = self._ensure_tuple(module.padding, self.dimensions)
        dilation = self._ensure_tuple(module.dilation, self.dimensions)

        # 1. Calculate softmax weights
        scaled_inputs = self.beta * input_tensor

        # Pool to get max values for numerical stability
        max_vals_pooled = self.pool_fn(
            scaled_inputs, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )

        # Create common parameters for unpooling
        unpool_kernel, output_padding, slicing = self._create_unpool_params(
            scaled_inputs, max_vals_pooled, stride, kernel_size, padding, dilation
        )
        num_channels = input_tensor.shape[1]

        # Unpool max values
        max_vals_unpooled = self._unpool(
            max_vals_pooled, unpool_kernel, stride, padding,
            output_padding, dilation, num_channels, slicing
        )

        # Calculate exp(input - max) for numerical stability
        exp_input = torch.exp(scaled_inputs - max_vals_unpooled)

        # Sum exp_input over windows
        summed_exp_pooled = self.conv_fn(
            exp_input, weight=unpool_kernel, stride=stride,
            padding=padding, dilation=dilation, groups=num_channels
        )

        # Unpool summed exp values
        summed_exp_unpooled = self._unpool(
            summed_exp_pooled, unpool_kernel, stride, padding,
            output_padding, dilation, num_channels, slicing
        )

        # Calculate softmax output
        softmax_output = exp_input / (summed_exp_unpooled + 1e-10)

        # 2. Distribute gradients
        expanded_grad = self._unpool(
            grad_output[0], unpool_kernel, stride, padding,
            output_padding, dilation, num_channels, slicing
        )

        # Return weighted gradients
        return (softmax_output * expanded_grad,)
