==========================
Writing Custom Attributors
==========================

**Attributors** provide an additional layer of abstraction over the context of
**Composites**, and are used to directly produce *attributions*, which may or
may not be computed with modified gradients, if they are used, from
**Composites**.
More information on **Attributors**, examples and their use can be found in
:doc:`/how-to/use-attributors`.

**Attributors** can be used to implement non-layer-wise or only partly
layer-wise attribution methods.
For this, it is enough to define a subclass of
:py:class:`zennit.attribution.Attributor` and implement its
:py:meth:`~zennit.attribution.Attributor.forward` and optionally its
:py:meth:`~zennit.attribution.Attributor.__init__` methods.

:py:meth:`~zennit.attribution.Attributor.forward` takes 2 arguments, the tensor
with respect to which the attribution shall be computed ``input``, and
``attr_output_fn``, which is a function that, given the output of the
attributed model, computes the *gradient output* for the gradient computation,
which is, for example, a one-hot encoding of the target label of the attributed
input.
When calling an :py:class:`~zennit.attribution.Attributor`, the ``__call__``
function will ensure ``forward`` receives a valid function to transform the
output of the analyzed model to a tensor which can be used for the
``grad_output`` argument of :py:func:`torch.autograd.grad`.
A constant tensor or function is provided by the user either to ``__init__`` or
to ``__call__``.
It is expected that :py:meth:`~zennit.attribution.Attributor.forward` will
return a tuple containing, in order, the model output and the attribution.

As an example, we can implement *gradient times input* in the following way:

.. code-block:: python

    import torch
    from torchvision.models import vgg11

    from zennit.attribution import Attributor


    class GradientTimesInput(Attributor):
        '''Model-agnostic gradient times input.'''
        def forward(self, input, attr_output_fn):
            '''Compute gradient times input.'''
            input_detached = input.detach().requires_grad_(True)
            output = self.model(input_detached)
            gradient, = torch.autograd.grad(
                (output,), (input_detached,), (attr_output_fn(output.detach()),)
            )
            relevance = gradient * input
            return output, relevance

    model = vgg11()
    data = torch.randn((1, 3, 224, 224))

    with GradientTimesInput(model) as attributor:
        output, relevance = attributor(data)

:py:class:`~zennit.attribution.Attributor` accepts an optional
:py:class:`~zennit.core.Composite`, which, if supplied, will always be used to
create a context in ``__call__`` around ``forward``.
For the ``GradientTimesInput`` class above, using a **Composite** will probably
not produce anything useful, although more involved combinations of custom
**Rules** and a custom **Attributor** can be used to implement complex
attribution methods with both model-agnostic and layer-wise parts.

The following shows an example of *sensitivity analysis*, which is the absolute
value, with a custom ``__init__()`` where we can pass the argument
``sum_channels`` to specify whether the **Attributor** should sum over the
channel dimension:

.. code-block:: python

    import torch
    from torchvision.models import vgg11

    from zennit.attribution import Attributor


    class SensitivityAnalysis(Attributor):
        '''Model-agnostic sensitivity analysis which optionally sums over color
        channels.
        '''
        def __init__(
            self, model, sum_channels=False, composite=None, attr_output=None
        ):
            super().__init__(
                model, composite=composite, attr_output=attr_output
            )

            self.sum_channels = sum_channels


        def forward(self, input, attr_output_fn):
            '''Compute the absolute gradient (or the sensitivity) and
            optionally sum over the color channels.
            '''
            input_detached = input.detach().requires_grad_(True)
            output = self.model(input_detached)
            gradient, = torch.autograd.grad(
                (output,), (input_detached,), (attr_output_fn(output.detach()),)
            )
            relevance = gradient.abs()
            if self.sum_channels:
                relevance = relevance.sum(1)
            return output, relevance

    model = vgg11()
    data = torch.randn((1, 3, 224, 224))

    with SensitivityAnalysis(model, sum_channels=True) as attributor:
        output, relevance = attributor(data)
