# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/attribution.py
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
'''Attributors are convenience objects to compute attributions, optionally using composites.'''
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import product

import torch


def constant(obj):
    '''Wrapper function to create a function which returns a constant object regardless of arguments.

    Parameters
    ----------
    obj : object
        Constant object which the returned wrapper function will return on call.

    Returns
    -------
    wrapped_const : function
        Function which when called with any arguments will return ``obj``.

    '''
    def wrapped_const(*args, **kwargs):
        return obj
    return wrapped_const


def identity(obj):
    '''Identity function.

    Parameters
    ----------
    obj : object
        Any object which will be returned.

    Result
    ------
    obj : object
        The original input argument ``obj``.

    '''
    return obj


def occlude_independent(input, mask, fill_fn=torch.zeros_like, invert=False):
    '''Given a ``mask``, occlude pixels of ``input`` independently given a function ``fill_fn``.

    Parameters
    ----------
    input : :py:obj:`torch.Tensor`
        The input tensor which will be occluded in the pixels where ``mask`` is non-zero, or, if ``invert`` is
        ``True``, where ``mask`` is zero, using function ``fill_fn``.
    mask : :py:obj:`torch.Tensor`
        Boolean mask, at which non-zero or zero (given ``invert``) elements will be occluded in ``input``.
    fill_fn : function, optional
        Function used to occlude pixels with signature ``(input : torch.Tensor) -> torch.Tensor``, where input is the
        same shape as ``input``, and the output shall leave the shape unchanged. Default is ``torch.zeros_like``, which
        will replace occluded pixels with zero.
    invert : bool, optional
        If ``True``, inverts the supplied mask. Default is ``False``, i.e. not to invert.

    Returns
    -------
    :py:obj:`torch.Tensor`
        The occluded tensor.

    '''
    if invert:
        mask = ~mask
    return input * mask + ~mask * fill_fn(input)


class Attributor(metaclass=ABCMeta):
    '''Base Attributor Class.

    Attributors are convenience objects with an optional composite and when called, compute an attribution, e.g., the
    gradient or anything that is the result of computing the gradient when using the provided composite.  Attributors
    also provide a context to be used in a `with` statement, similar to `CompositeContext`s. If the forward function
    (or `self.__call__`) is called and the composite has not been registered (i.e. `composite.handles` is empty), the
    composite will be temporarily registered to the model.

    Parameters
    ----------
    model: obj:`torch.nn.Module`
        The model for which the attribution will be computed. If `composite` is provided, this will also be the model
        to which the composite will be registered within `with` statements, or when calling the `Attributor` instance.
    composite: obj:`zennit.core.Composite`, optional
        The optional composite to, if provided, be registered to the model within `with` statements or when calling the
        `Attributor` instance.
    attr_output: obj:`torch.Tensor` or callable, optional
        The default output attribution to be used when calling the `Attributor` instance, which is either a Tensor
        compatible with any input used, or a function of the model's output. If None (default), the value will be the
        identity function.

    '''
    def __init__(self, model, composite=None, attr_output=None):
        self.model = model
        self.composite = composite

        if attr_output is None:
            self.attr_output_fn = identity
        elif not callable(attr_output):
            self.attr_output_fn = constant(attr_output)
        else:
            self.attr_output_fn = attr_output

    def __enter__(self):
        '''Register the composite, if provided.

        Returns
        -------
        self: obj:`Attributor`
            The `Attributor` instance.

        '''
        if self.composite is not None:
            self.composite.register(self.model)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''Remove the composite, if provided.

        Returns
        -------
        False

        '''
        if self.composite is not None:
            self.composite.remove()
        return False

    def __call__(self, input, attr_output=None):
        '''Compute the attribution of the model wrt. `input`, using `attr_output` as the output attribution if
        provided, or the default output attribution otherwise (if not supplied during instantiation either, this will
        be the full output of the model). If a composite was supplied to the `Attributor` instance, but it was not yet
        registered (either manually, or in a `with` statement), it will be registered to the model temporarily during
        the call.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution, which is either a Tensor compatible `input` (i.e. has the same shape as the output
            of the model), or a function of the model's output. If None (default), the default attribution will be
            used, which if neither supplied, will result in the model output used as the output attribution.

        Returns
        -------
        output: obj:`torch.Tensor`
            Output of the model with argument `input`.
        attribution: obj:`torch.Tensor`
            Attribution of the model wrt. to `input`, with the same shape as `input`.
        '''
        if attr_output is None:
            attr_output_fn = self.attr_output_fn
        elif not callable(attr_output):
            attr_output_fn = constant(attr_output)
        else:
            attr_output_fn = attr_output

        if self.composite is None or self.composite.handles:
            return self.forward(input, attr_output_fn)

        with self:
            return self.forward(input, attr_output_fn)

    @abstractmethod
    def forward(self, input, attr_output_fn):
        '''Abstract method. Compute the attribution of the model wrt. input, by using `attr_output_fn` as the function
        of the model output to provide the output attribution. This function will not register the composite, and is
        wrapped in the `__call__` of `Attributor`.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution function of the model's output.
        '''


class Gradient(Attributor):
    '''The Gradient Attributor. The result is the product of the attribution output and the (possibly modified)
    jacobian. With a composite, i.e. `EpsilonGammaBox`, this will compute the Layerwise Relevance Propagation
    attribution values.
    '''
    def forward(self, input, attr_output_fn):
        '''Compute the gradient of the model wrt. input, by using `attr_output_fn` as the function of the model output
        to provide the vector for the vector jacobian product.
        This function will not register the composite, and is wrapped in the `__call__` of `Attributor`.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution.
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution function of the model's output.

        Returns
        -------
        output: obj:`torch.Tensor`
            Output of the model given `input`.
        attribution: obj:`torch.Tensor`
            Attribution of the model wrt. to `input`, with the same shape as `input`.
        '''
        input = input.detach().requires_grad_(True)
        output = self.model(input)
        gradient, = torch.autograd.grad((output,), (input,), grad_outputs=(attr_output_fn(output.detach()),))
        return output, gradient


class SmoothGrad(Attributor):
    '''This implements SmoothGrad [1]_. The result is the average over the gradient of multiple iterations where some
    normal distributed noise was added to the input. Supplying a composite will result instead in averaging over the
    modified gradient.

    Parameters
    ----------
    model: obj:`torch.nn.Module`
        The model for which the attribution will be computed. If `composite` is provided, this will also be the model
        to which the composite will be registered within `with` statements, or when calling the `Attributor` instance.
    composite: obj:`zennit.core.Composite`, optional
        The optional composite to, if provided, be registered to the model within `with` statements or when calling the
        `Attributor` instance.
    attr_output: obj:`torch.Tensor` or callable, optional
        The default output attribution to be used when calling the `Attributor` instance, which is either a Tensor
        compatible with any input used, or a function of the model's output. If None (default), the value will be the
        identity function.
    noise_level: float, optional
        The noise level, which is :math:`\\frac{\\sigma}{x_{max} - x_{min}}` and defaults to 0.1.
    n_iter: int, optional
        The number of iterations over which to average, defaults to 20.

    References
    ----------
    .. [1] D. Smilkov, N. Thorat, B. Kim, F. B. Viégas, and M. Wattenberg: "SmoothGrad: removing noise by adding
           noise," CoRR, vol. abs/1706.03825, 2017.

    '''
    def __init__(self, model, composite=None, attr_output=None, noise_level=0.1, n_iter=20):
        super().__init__(model=model, composite=composite, attr_output=attr_output)
        self.noise_level = noise_level
        self.n_iter = n_iter

    def forward(self, input, attr_output_fn):
        '''Compute the SmoothGrad of the model wrt. input, by using `attr_output_fn` as the function of the model output
        to provide the vector for the vector jacobian product used to compute the gradient.
        This function will not register the composite, and is wrapped in the `__call__` of `Attributor`.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution.
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution function of the model's output.

        Returns
        -------
        output: obj:`torch.Tensor`
            Output of the model given `input`.
        attribution: obj:`torch.Tensor`
            Attribution of the model wrt. to `input`, with the same shape as `input`.
        '''
        input = input.detach()

        dims = tuple(range(1, input.ndim))
        std = self.noise_level * (input.amax(dims, keepdim=True) - input.amin(dims, keepdim=True))

        result = torch.zeros_like(input)
        for n in range(self.n_iter):
            # the last epsilon is defined as zero to compute the true output,
            # and have SmoothGrad w/ n_iter = 1 === gradient
            if n == self.n_iter - 1:
                epsilon = torch.zeros_like(input)
            else:
                epsilon = torch.randn_like(input) * std
            noisy_input = (input + epsilon).requires_grad_()
            output = self.model(noisy_input)
            gradient, = torch.autograd.grad((output,), (noisy_input,), grad_outputs=(attr_output_fn(output.detach()),))
            result += gradient / self.n_iter

        # output is leaking from the loop for the last epsilon (which is zero)
        return output, result


class IntegratedGradients(Attributor):
    '''This implements Integrated Gradients [2]_. The result is the path integral of the gradients, estimated over
    multiple discrete iterations. Supplying a composite will result instead in the path integral over the modified
    gradient.

    Parameters
    ----------
    model: obj:`torch.nn.Module`
        The model for which the attribution will be computed. If `composite` is provided, this will also be the model
        to which the composite will be registered within `with` statements, or when calling the `Attributor` instance.
    composite: obj:`zennit.core.Composite`, optional
        The optional composite to, if provided, be registered to the model within `with` statements or when calling the
        `Attributor` instance.
    attr_output: obj:`torch.Tensor` or callable, optional
        The default output attribution to be used when calling the `Attributor` instance, which is either a Tensor
        compatible with any input used, or a function of the model's output. If None (default), the value will be the
        identity function.
    baseline_fn: callable, optional
        The baseline for which the model output is zero, supplied as a function of the input. Defaults to
        `torch.zeros_like`.
    n_iter: int, optional
        The number of iterations used to estimate the integral, defaults to 20.

    References
    ----------
    .. [2] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in Proceedings of the 34th
       International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, ser.
       Proceedings of Machine Learning Research, D. Precup and Y. W. Teh, Eds., vol. 70. PMLR, 2017, pp. 3319–3328.

    '''
    def __init__(self, model, composite=None, attr_output=None, baseline_fn=None, n_iter=20):
        super().__init__(model=model, composite=composite, attr_output=attr_output)
        if baseline_fn is None:
            baseline_fn = torch.zeros_like
        self.baseline_fn = baseline_fn
        self.n_iter = n_iter

    def forward(self, input, attr_output_fn):
        '''Compute the Integrated Gradients of the model wrt. input, by using `attr_output_fn` as the function of the
        model output to provide the vector for the vector jacobian product used to compute the gradient.
        This function will not register the composite, and is wrapped in the `__call__` of `Attributor`.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution.
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution function of the model's output.

        Returns
        -------
        output: obj:`torch.Tensor`
            Output of the model given `input`.
        attribution: obj:`torch.Tensor`
            Attribution of the model wrt. to `input`, with the same shape as `input`.
        '''
        input = input.detach()

        baseline = self.baseline_fn(input)

        result = torch.zeros_like(input)
        for alpha in torch.linspace(1. / self.n_iter, 1., self.n_iter):
            path_step = (baseline + alpha * (input - baseline)).requires_grad_()
            output = self.model(path_step)
            gradient, = torch.autograd.grad((output,), (path_step,), grad_outputs=(attr_output_fn(output.detach()),))
            result += gradient / self.n_iter

        result *= (input - baseline)
        # in the last step, path_step is equal to input, thus `output` is the original output
        return output, result


class Occlusion(Attributor):
    '''This implements attribution by occlusion. Supplying a composite will have no effect on the result, as the
    gradient is not used.

    Parameters
    ----------
    model: obj:`torch.nn.Module`
        The model for which the attribution will be computed. If `composite` is provided, this will also be the model
        to which the composite will be registered within `with` statements, or when calling the `Attributor` instance.
    composite: obj:`zennit.core.Composite`, optional
        The optional composite to, if provided, be registered to the model within `with` statements or when calling the
        `Attributor` instance. Note that for Occlusion, this has no effect on the result.
    attr_output: obj:`torch.Tensor` or callable, optional
        The default output attribution to be used when calling the `Attributor` instance, which is either a Tensor
        compatible with any input used, or a function of the model's output. If None (default), the value will be the
        identity function.
    occlusion_fn: callable, optional
        The occluded function, called with `occlusion_fn(input, mask)`, where `mask` is 1 inside the sliding window,
        and 0 outside. Either values inside or outside the sliding window may be occluded for different effects. By
        default, all values except inside the sliding window will be occluded.
    window: int or tuple of ints, optional
        The size of the sliding window to occlude over the input for each dimension. Defaults to 8. If a single integer
        is provided, the sliding window will slide with the same size over all dimensions except the first, which is
        assumed as the batch-dimension. If a tuple is provided, the window will only slide over the n-last dimensions,
        where n is the length of the tuple, e.g., if the data has shape `(3, 32, 32)` and `window=(8, 8)`, the
        resulting mask will have a block of shape `(3, 8, 8)` set to True. `window` must have the same length as
        `stride`.
    stride: int or tuple of ints, optional
        The strides used for the sliding window to occlude over the input for each dimension. Defaults to 8. If a
        single integer is provided, the strides will be the same size for all dimensions. If a tuple is provided,
        the window will only stride over the n-last dimensions, where n is the length of the tuple.
        `stride` must have the same length as `window`.

    '''
    def __init__(self, model, composite=None, attr_output=None, occlusion_fn=None, window=8, stride=8):
        def typecheck(obj):
            return isinstance(obj, int) or isinstance(obj, tuple) and all(isinstance(elem, int) for elem in obj)

        super().__init__(model=model, composite=composite, attr_output=attr_output)

        if not typecheck(window):
            raise TypeError('Occlusion window must either be an int, or a tuple of ints.')
        if not typecheck(stride):
            raise TypeError('Occlusion window must either be an int, or a tuple of ints.')

        if occlusion_fn is None:
            occlusion_fn = partial(occlude_independent, fill_fn=torch.zeros_like, invert=False)
        self.occlusion_fn = occlusion_fn
        self.window = window
        self.stride = stride

    def _resolve_window_stride(self, input):
        window = self.window
        stride = self.stride
        if isinstance(window, int):
            window = tuple(min(window, size) for size in input.shape[1:])
        if isinstance(stride, int):
            stride = tuple(min(stride, size) for size in input.shape[1:])

        if len(window) < input.ndim:
            window = tuple(input.shape)[:input.ndim - len(window)] + window
        if len(stride) < len(window):
            stride = window[:len(window) - len(stride)] + stride

        return window, stride

    def forward(self, input, attr_output_fn):
        '''Compute the occlusion analysis of the model wrt. input, by using `attr_output_fn` as function of the output,
        to return a weighting, which when multiplied again with the output, results in the classification score.
        This function will not register the composite, and is wrapped in the `__call__` of `Attributor`.

        Parameters
        ----------
        input: obj:`torch.Tensor`
            Input for the model, and wrt. compute the attribution.
        attr_output: obj:`torch.Tensor` or callable, optional
            The output attribution function of the model's output.

        Returns
        -------
        output: obj:`torch.Tensor`
            Output of the model given `input`.
        attribution: obj:`torch.Tensor`
            Attribution of the model wrt. to `input`, with the same shape as `input`.
        '''
        window, stride = self._resolve_window_stride(input)

        root_mask = torch.zeros_like(input, dtype=bool)
        root_mask[tuple(slice(0, elem) for elem in window)] = True

        result = torch.zeros_like(input)
        for offset in product(*(range(0, size, local_stride) for size, local_stride in zip(input.shape, stride))):
            mask = root_mask.roll(offset, tuple(range(input.ndim)))
            occluded_input = self.occlusion_fn(input, mask)
            with torch.no_grad():
                output = self.model(occluded_input)
            score = attr_output_fn(output).sum(tuple(range(1, output.ndim)))
            result += mask * score[(slice(None),) + (None,) * (input.ndim - 1)]

        output = self.model(input)
        return output, result
