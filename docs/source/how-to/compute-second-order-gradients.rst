================================
Computing Second Order Gradients
================================

Sometimes, it may be necessary to compute the gradient of the attribution. One
example is to compute the gradient with respect to the input in order to
find adversarial explanations :cite:p:`dombrowski2019explanations`,
or to regularize or transform the attributions of a network
:cite:p:`anders2020fairwashing`.

In Zennit, the attribution is computed using the modified gradient, which means
that in order to compute the gradient of the attribution, the second order
gradient needs to be computed. Pytorch natively supports the computation of
higher order gradients, simply by supplying ``create_graph=True`` with
:py:func:`torch.autograd.grad` to declare that the backward-function needs to
be backward-able itself.


Vanilla Gradient and ReLU
-------------------------

If we simply need the second order gradient of a model, without using Zennit, we can do the following:

.. code-block:: python

    import torch
    from torch.nn import Sequential, Conv2d, ReLU, Linear, Flatten


    # setup the model and data
    model = Sequential(
        Conv2d(3, 10, 3, padding=1),
        ReLU(),
        Flatten(),
        Linear(10 * 32 * 32, 10),
    )
    input = torch.randn(1, 3, 32, 32)

    # make sure the input requires a gradient
    input.requires_grad = True

    output = model(input)
    # a vector for the vector-jacobian-product, i.e. the grad_output
    target = torch.ones_like(output)

    grad, = torch.autograd.grad(output, input, target, create_graph=True)

    # the grad_output for grad
    gradtarget = torch.ones_like(grad)
    # compute the second order gradient
    gradgrad, = torch.autograd.grad(grad, input, gradtarget)

Here, you might notice that ``gradgrad`` is all zeros, regardless of the input
and model parameters. The culprit is ``ReLU``, which has a gradient of zero
everywhere except at zero, where it is undefined. In order to get a meaningful
gradient, we could instead use a *smooth* activation function in our model.
However, ReLU models are quite common, and we may not like to retrain every
model using only smooth activation functions.

:cite:t:`dombrowski2019explanations` proposed to replace the ReLU activations
with its smooth variation, the *Softplus* function:

.. math::

    \text{Softplus}(x;\beta) = \frac{1}{\beta} \log (1 + \exp (\beta x))
    \,\text{.}

With :math:`\beta\rightarrow\infty`, Softplus will be equivalent to ReLU, but in
practice choosing :math:`\beta = 10` is most often sufficient to keep the model
output unchanged but still obtain a meaningful second order gradient.

To temporarily replace the ReLU gradients in-place, we can use the
:py:class:`~zennit.rules.ReLUBetaSmooth` rule:


.. code-block:: python

    from zennit.composites import BetaSmooth

    # LayerMapComposite which assigns the ReLUBetaSmooth hook to ReLUs
    composite = BetaSmooth(beta_smooth=10.)

    with composite.context(model):
        output = model(input)
        target = torch.ones_like(output)
        grad, = torch.autograd.grad(output, input, target, create_graph=True)

    gradtarget = torch.ones_like(grad)
    gradgrad, = torch.autograd.grad(grad, input, gradtarget)

Notice here that we computed the second order gradient **outside** of the
composite context. A property of the Pytorch gradients hooks is that they are
also called when the *second* order gradient with respect to a tensor is
computed.
Due to this, computing the second order gradient *while rules are still
registered* will lead to incorrect results.

Temporarily Disabling Hooks
---------------------------

In order compute the second order gradient *without* removing the hooks (i.e. to
compute multiple values in a loop), we can temporarily deactivate them using
:py:meth:`zennit.core.Composite.inactive`:

.. code-block:: python

    with composite.context(model):
        output = model(input)
        target = torch.ones_like(output)
        grad, = torch.autograd.grad(output, input, target, create_graph=True)

        # temporarily disable all hooks registered by composite
        with composite.inactive():
            gradtarget = torch.ones_like(grad)
            gradgrad, = torch.autograd.grad(grad, input, gradtarget)

All Attributors support the computation of gradients. For gradient-based
attributors like :py:class:`~zennit.attribution.Gradient` or
:py:class:`~zennit.attribution.SmoothGrad`, the ``create_graph=True`` parameter
can be supplied to the class constructor:

.. code-block:: python

    from zennit.attribution import Gradient
    from zennit.composites import EpsilonGammaBox

    # any composites support second order gradients
    composite = EpsilonGammaBox(low=-3., high=3.)

    with Gradient(model, composite, create_graph=True) as attributor:
        output, grad = attributor(input, torch.ones_like)

        # temporarily disable all hooks registered by the attributor's composite
        with attributor.inactive():
            gradtarget = torch.ones_like(grad)
            gradgrad, = torch.autograd.grad(grad, input, gradtarget)

Here, we also used a different composite, which results in the gradient
computation of the modified gradient. Since the ReLU gradient is ignored (using
the :py:class:`~zennit.rules.Pass` rule) for Layer-wise Relevance
Propagation-specific composites, we do not need to use the
:py:class:`~zennit.rules.ReLUBetaSmooth` rule. However, if this behaviour
should be overwritten, :ref:`cooperative-layermapcomposites` can be used.

Using Hooks Only
----------------

Under the hood, :py:class:`~zennit.core.Hook` has an attribute ``active``,
which, when set to ``False``, will not execute the associated backward function.
A minimal example without using composites would look like the following:

.. code-block:: python

    from zennit.rules import Epsilon

    conv = Conv2d(3, 10, 3, padding=1)

    # create and register the hook
    epsilon = Epsilon()
    handles = epsilon.register(conv)

    output = conv(input)
    target = torch.ones_like(output)
    grad, = torch.autograd.grad(output, input, target, create_graph=True)

    # during this block, epsilon will be inactive
    epsilon.active = False
    grad_target = torch.ones_like(grad)
    gradgrad, = torch.autograd.grad(grad, input, grad_target)
    epsilon.active = True

    # after calling handles.remove, epsilon will also be inactive
    handles.remove()

The same can here also be achieved by simply removing the handles before calling
``torch.autograd.grad`` on ``grad``, although the hooks would then need to be
re-registered in order to compute the epsilon-modified gradient again.
