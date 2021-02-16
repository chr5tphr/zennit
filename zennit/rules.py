'''Rules based on Hooks'''
import torch

from .core import Hook, LinearHook, stabilize


class Epsilon(LinearHook):
    '''Epsilon LRP rule.

    Parameters
    ----------
    epsilon: float, optional
        Stabilization parameter.
    '''
    def __init__(self, epsilon=1e-6):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param: param],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0], epsilon)),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class Gamma(LinearHook):
    '''Gamma LRP rule.

    Parameters
    ----------
    gamma: float, optional
        Multiplier for added positive weights.
    '''
    def __init__(self, gamma=0.25):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param: param + gamma * param.clamp(min=0)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class ZPlus(LinearHook):
    '''ZPlus (or alpha=1, beta=0) LRP rule.'''
    def __init__(self):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param: param.clamp(min=0)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class AlphaBeta(LinearHook):
    '''AlphaBeta LRP rule.

    Parameters
    ----------
    alpha: float, optional
        Multiplier for the positive output term.
    beta: float, optional
        Multiplier for the negative output term.
    '''
    def __init__(self, alpha=2., beta=1.): 
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0), 
                lambda input: input.clamp(max=0), 
                lambda input: input.clamp(min=0), 
                lambda input: input.clamp(max=0)  
                ], 

            param_modifiers=[
                lambda param: param.clamp(min=0), 
                lambda param: param.clamp(max=0), 
                lambda param: param.clamp(max=0), 
                lambda param: param.clamp(min=0) 
                ],

            output_modifiers=[lambda output: output] * 4,

            gradient_mapper=(lambda out_grad, outputs: [out_grad / stabilize(output) for output in outputs]),
      
            reducer=(lambda inputs, gradients: alpha * (inputs[0] * gradients[0] + inputs[1] * gradients[1]) 
                - abs(beta) * (inputs[2] * gradients[2] + inputs[3] * gradients[3]))
        )


class ZBox(LinearHook):
    '''ZBox LRP rule for input pixel space.

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
                lambda input: low[:input.shape[0]],
                lambda input: high[:input.shape[0]],
            ],
            param_modifiers=[
                lambda param: param,
                lambda param: param.clamp(min=0),
                lambda param: param.clamp(max=0)
            ],
            output_modifiers=[lambda output: output] * 3,
            gradient_mapper=(lambda out_grad, outputs: (out_grad / stabilize(sub(*outputs)),) * 3),
            reducer=(lambda inputs, gradients: sub(*(input * gradient for input, gradient in zip(inputs, gradients))))
        )


class Pass(Hook):
    '''If the rule of a layer shall not be any other, is elementwise and shall not be the gradient, the `Pass` rule
    simply passes upper layer relevance through to the lower layer.
    '''
    def backward(self, module, grad_input, grad_output):
        '''Pass through the upper gradient, skipping the one for this layer.'''
        return grad_output


class Norm(LinearHook):
    '''Normalize and weigh relevance by input contribution.
    This is essentially the same as the LRP Epsilon Rule with a fixed epsilon only used as a stabilizer, and without
    the need of the attached layer to have parameters `weight` and `bias`.
    '''
    def __init__(self):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[None],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0])
        )


class WSquare(LinearHook):
    '''This is the WSquare LRP rule.'''
    def __init__(self):
        super().__init__(
            input_modifiers=[torch.ones_like],
            param_modifiers=[lambda param: param ** 2],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: gradients[0])
        )


class Flat(LinearHook):
    '''This is the Flat LRP rule. It is essentially the same as the WSquare Rule, but with all parameters set to ones.
    '''
    def __init__(self):
        super().__init__(
            input_modifiers=[torch.ones_like],
            param_modifiers=[torch.ones_like],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: gradients[0])
        )
