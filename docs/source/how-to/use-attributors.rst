=================
Using Attributors
=================

**Attributors** are used to both shorten Zennit's common ``composite.context ->
gradient`` approach, as well as provide model-agnostic attribution approaches.
Available **Attributors** can be found in :py:mod:`zennit.attribution`, some of
which are:

* :py:class:`~zennit.attribution.Gradient`, which computes the gradient
* :py:class:`~zennit.attribution.IntegratedGradients`, which computes the
  Integrated Gradients
* :py:class:`~zennit.attribution.SmoothGrad`, which computes SmoothGrad
* :py:class:`~zennit.attribution.Occlusion`, which computes the attribution
  based on the model output activation values when occluding parts of the input
  with a sliding window

Using the basic :py:class:`~zennit.attribution.Gradient`, the unmodified
gradient may be computed with:

.. code-block:: python

    import torch
    from torch.nn import Sequential, Conv2d, ReLU, Linear, Flatten
    from zennit.attribution import Gradient

    # setup the model
    model = Sequential(
        Conv2d(3, 8, 3, padding=1),
        ReLU(),
        Conv2d(8, 16, 3, padding=1),
        ReLU(),
        Flatten(),
        Linear(16 * 32 * 32, 1024),
        ReLU(),
        Linear(1024, 10),
    )
    # some random input data
    input = torch.randn(1, 3, 32, 32, requires_grad=True)

    # compute the gradient and output using the Gradient attributor
    with Gradient(model) as attributor:
        output, relevance = attributor(input)

Computing attributions using a composite can be done with:

.. code-block:: python

    from zennit.composites import EpsilonPlusFlat

    # prepare the composite
    composite = EpsilonPlusFlat()

    # compute the gradient within the composite's context, i.e. the
    # EpsilonPlusFlat LRP relevance
    with Gradient(model, composite) as attributor:
        # torch.eye is used here to get a one-hot encoding of the
        # first (index 0) label
        output, relevance = attributor(input, torch.eye(10)[[0]])

which uses the second argument ``attr_output_fn`` of the call to
:py:class:`~zennit.attribution.Attributor` to specify a constant tensor used for
the *output relevance* (i.e. ``grad_output``), but alternatively, a function
of the output may also be used:

.. code-block:: python

    def one_hot_max(output):
        '''Get the one-hot encoded max at the original indices in dim=1'''
        values, indices = output.max(1)
        return values[:, None] * torch.eye(output.shape[1])[indices]

    with Gradient(model) as attributor:
        output, relevance = attributor(input, one_hot_max)

The constructor of :py:class:`~zennit.attribution.Attributor` also has a third
argument ``attr_output``, which also can either be a constant
:py:class:`~torch.Tensor`, or a function of the model's output and specifies
which *output relevance* (i.e. ``grad_output``) should be used by default. When
not supplying anything, the default will be the *identity*. If the default
should be for example ones for all outputs, one could write:

.. code-block:: python

    # compute the gradient and output using the Gradient attributor, and with
    # a vector of ones as grad_output
    with Gradient(model, attr_output=torch.ones_like) as attributor:
        output, relevance = attributor(input)

Gradient-based **Attributors** like
:py:class:`~zennit.attribution.IntegratedGradients` and
:py:class:`~zennit.attribution.SmoothGrad` may also be used together with
composites to produce *hybrid attributions*:

.. code-block:: python

    from zennit.attribution import SmoothGrad

    # prepare the composite
    composite = EpsilonPlusFlat()

    # do a *smooth* version of EpsilonPlusFlat LRP by using the SmoothGrad
    # attributor in combination with the composite
    with SmoothGrad(model, composite, noise_level=0.1, n_iter=20) as attributor:
         output, relevance = attributor(input, torch.eye(10)[[0]])

which in this case will sample 20 samples in an epsilon-ball (size controlled
with `noise_level`) around the input. Note that for Zennit's implementation of
:py:class:`~zennit.attribution.SmoothGrad`, the first sample will always be the
original input, i.e. ``SmoothGrad(model, n_iter=1)`` will produce the plain
gradient as ``Gradient(model)`` would.

:py:class:`~zennit.attribution.Occlusion` will move a sliding window with
arbitrary size and strides over an input with any dimensionality. In addition to
specifying window-size and strides, a function may be specified, which will be
supplied with the input and a mask. When using the default, everything within
the sliding window will be set to zero. A function
:py:func:`zennit.attribution.occlude_independent` is available to simplify the
process of specifying how to fill the window, and to invert the window if
desired. The following adds some gaussian noise to the area within the sliding
window:

.. code-block:: python

    from functools import partial
    from zennit.attribution import Occlusion, occlude_independent

    input = torch.randn((16, 3, 32, 32))

    attributor = Occlusion(
        model,
        window=8,  # 8x8 overlapping windows
        stride=4,  # with strides 4x4
        occlusion_fn=partial(  # occlusion_fn gets the full input and a mask
            occlude_independent,  # applies fill_fn at provided mask
            fill_fn=lambda x: x * torch.randn_like(x) * 0.2,  # add some noise
            invert=False  # do not invert, i.e. occlude *within* mask
        )
    )
    with attributor:
        # for occlusion, the tracked score is the vector product of the
        # provided *grad_output* and the model's output
        output, relevance = attributor(input, torch.eye(10)[[0]])


Note that while the interface allows to pass a composite for any
:py:class:`~zennit.attribution.Attributor`, using a composite with
:py:class:`~zennit.attribution.Occlusion` does not change the outcome, as it
does not utilize the gradient.

An introduction on how to write custom **Attributors** can be found at
:doc:`/how-to/write-custom-attributors`.
