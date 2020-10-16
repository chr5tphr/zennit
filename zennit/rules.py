'''Rules based on Hooks'''
from .core import LinearHook, stabilize


class Epsilon(LinearHook):
    '''LRP Epsilon Rule

    Parameters
    ----------
    epsilon: float, optional
        Stabilization parameter.
    '''
    def __init__(self, epsilon=1e-6):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param: param],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0], epsilon)),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class Gamma(LinearHook):
    '''LRP Gamma Rule

    Parameters
    ----------
    gamma: float, optional
        Multiplier for added positive weights.
    '''
    def __init__(self, gamma=0.25):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param: param + gamma * param.clamp(min=0)],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class ZPlus(LinearHook):
    '''LRP ZPlus (or alpha=1, beta=1) Rule.'''
    def __init__(self):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param: param.clamp(min=0)],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class AlphaBeta(LinearHook):
    '''AlphaBeta LRP Rule.

    Parameters
    ----------
    alpha: float, optional
        Multiplier for the positive weight part.
    beta: float, optional
        Multiplier for the negative weight part.
    '''
    def __init__(self, alpha=2., beta=1.):
        super().__init__(
            input_modifiers=[lambda input: input] * 2,
            param_modifiers=[
                lambda param: param.clamp(min=0),
                lambda param: param.clamp(max=0)
            ],
            gradient_mapper=(lambda out_grad, outputs: [out_grad / stabilize(output) for output in outputs]),
            reducer=(lambda inputs, gradients: inputs[0] * (alpha * gradients[0] + beta * gradients[1]))
        )


class ZBox(LinearHook):
    '''ZBox LRP Rule for input pixel space.

    Parameters
    ----------
    low: obj:`torch.Tensor`
        Lowest pixel values of input.
    high: obj:`torch.Tensor`
        Highest pixel values of input.
    '''
    def __init__(self, low, high):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: low,
                lambda input: high,
            ],
            param_modifiers=[
                lambda param: param,
                lambda param: param.clamp(min=0),
                lambda param: param.clamp(max=0)
            ],
            gradient_mapper=(lambda out_grad, outputs: [out_grad / stabilize(sub(*outputs))] * 3),
            reducer=(lambda inputs, gradients: sub(*(input * gradient for input, gradient in zip(inputs, gradients))))
        )
