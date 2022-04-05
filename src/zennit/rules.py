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

from .core import Hook, BasicHook, stabilize, expand


class Epsilon(BasicHook):
    '''LRP Epsilon rule :cite:p:`bach2015pixel`.
    Setting ``(epsilon=0)`` produces the LRP-0 rule :cite:p:`bach2015pixel`.
    LRP Epsilon is most commonly used in middle layers, LRP-0 is most commonly used in upper layers
    :cite:p:`montavon2019layer`.
    Sometimes higher values of ``epsilon`` are used, therefore it is not always only a stabilizer value.

    Parameters
    ----------
    epsilon: float, optional
        Stabilization parameter.
    '''
    def __init__(self, epsilon=1e-6):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param, _: param],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0], epsilon)),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class Gamma(BasicHook):
    '''LRP Gamma rule :cite:p:`montavon2019layer`.

    Parameters
    ----------
    gamma: float, optional
        Multiplier for added positive weights.
    '''
    def __init__(self, gamma=0.25):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param, _: param + gamma * param.clamp(min=0)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class ZPlus(BasicHook):
    '''LRP ZPlus rule :cite:p:`bach2015pixel,montavon2017explaining`.
    It is the same as using :py:class:`~zennit.rules.AlphaBeta` with ``(alpha=1, beta=0)``

    Notes
    -----
    Note that the original deep Taylor Decomposition (DTD) specification of the ZPlus Rule
    :cite:p:`montavon2017explaining` only considers positive inputs, as they are used in ReLU Networks.
    This implementation is effectively alpha=1, beta=0, where negative inputs are allowed.
    '''
    def __init__(self):
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
            ],
            param_modifiers=[
                lambda param, _: param.clamp(min=0),
                lambda param, name: param.clamp(max=0) if name != 'bias' else torch.zeros_like(param),
            ],
            output_modifiers=[lambda output: output] * 2,
            gradient_mapper=(lambda out_grad, outputs: [out_grad / stabilize(sum(outputs))] * 2),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0] + inputs[1] * gradients[1])
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
    '''
    def __init__(self, alpha=2., beta=1.):
        if alpha < 0 or beta < 0:
            raise ValueError("Both alpha and beta parameters must be positive!")
        if (alpha - beta) != 1.:
            raise ValueError("The difference of parameters alpha - beta must equal 1!")

        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
            ],
            param_modifiers=[
                lambda param, _: param.clamp(min=0),
                lambda param, name: param.clamp(max=0) if name != 'bias' else torch.zeros_like(param),
                lambda param, _: param.clamp(max=0),
                lambda param, name: param.clamp(min=0) if name != 'bias' else torch.zeros_like(param),
            ],
            output_modifiers=[lambda output: output] * 4,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    out_grad / stabilize(denom)
                    for output, denom in zip(outputs, [sum(outputs[:2])] * 2 + [sum(outputs[2:])] * 2)
                ]
            ),
            reducer=(
                lambda inputs, gradients: (
                    alpha * (inputs[0] * gradients[0] + inputs[1] * gradients[1])
                    - beta * (inputs[2] * gradients[2] + inputs[3] * gradients[3])
                )
            )
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
    '''
    def __init__(self, low, high):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: expand(low, input.shape, cut_batch_dim=True).to(input),
                lambda input: expand(high, input.shape, cut_batch_dim=True).to(input),
            ],
            param_modifiers=[
                lambda param, _: param,
                lambda param, _: param.clamp(min=0),
                lambda param, _: param.clamp(max=0)
            ],
            output_modifiers=[lambda output: output] * 3,
            gradient_mapper=(lambda out_grad, outputs: (out_grad / stabilize(sub(*outputs)),) * 3),
            reducer=(lambda inputs, gradients: sub(*(input * gradient for input, gradient in zip(inputs, gradients))))
        )


class Pass(Hook):
    '''Unmodified pass-through rule.
    If the rule of a layer shall not be any other, is elementwise and shall not be the gradient, the `Pass` rule simply
    passes upper layer relevance through to the lower layer.
    '''
    def backward(self, module, grad_input, grad_output):
        '''Pass through the upper gradient, skipping the one for this layer.'''
        return grad_output


class Norm(BasicHook):
    '''Normalize and weight by input contribution.
    This is essentially the same as the LRP :py:class:`~zennit.rules.Epsilon` rule :cite:p:`bach2015pixel` with a fixed
    epsilon only used as a stabilizer, and without the need of the attached layer to have parameters ``weight`` and
    ``bias``.
    '''
    def __init__(self):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[None],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
            param_keys=[]
        )


class WSquare(BasicHook):
    '''LRP WSquare rule :cite:p:`montavon2017explaining`.
    It is most commonly used in the first layer when the values are not bounded :cite:p:`montavon2019layer`.
    '''
    def __init__(self):
        super().__init__(
            input_modifiers=[torch.ones_like],
            param_modifiers=[lambda param, _: param ** 2],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: gradients[0])
        )


class Flat(BasicHook):
    '''LRP Flat rule :cite:p:`lapuschkin2019unmasking`.
    It is essentially the same as the LRP :py:class:`~zennit.rules.WSquare` rule, but with all parameters set to ones.
    '''
    def __init__(self):
        super().__init__(
            input_modifiers=[torch.ones_like],
            param_modifiers=[
                lambda param, name: torch.ones_like(param) if name != 'bias' else torch.zeros_like(param)
            ],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: gradients[0]),
            require_params=False
        )


class ReLUDeconvNet(Hook):
    '''DeconvNet ReLU rule :cite:p:`zeiler2014visualizing`.'''
    def backward(self, module, grad_input, grad_output):
        '''Modify ReLU gradient according to DeconvNet :cite:p:`zeiler2014visualizing`.'''
        return (grad_output[0].clamp(min=0),)


class ReLUGuidedBackprop(Hook):
    '''GuidedBackprop ReLU rule :cite:p:`springenberg2015striving`.'''
    def backward(self, module, grad_input, grad_output):
        '''Modify ReLU gradient according to GuidedBackprop :cite:p:`springenberg2015striving`.'''
        return (grad_input[0] * (grad_output[0] > 0.),)
