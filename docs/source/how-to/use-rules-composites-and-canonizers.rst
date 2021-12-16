=======================================
Using Rules, Composites, and Canonizers
=======================================


Zennit implements propagation-based attribution methods by overwriting the
gradient of PyTorch modules within PyTorch's auto-differentiation engine.
There are three building blocks in Zennit to achieve attributions:
:ref:`use-rules`, :ref:`use-composites` and :ref:`use-canonizers`.
In short, **Rules** specify how to overwrite the gradient, **Composites** map
rules to modules, and **Canonizers** transform some module types and configurations
to a canonical form, necessary in some cases.

.. _use-rules:

Rules
-----

**Rules** are used to overwrite the gradient of individual modules by adding
forward hooks to modules, which track the input and output
:py:class:`torch.Tensors` and add `gradient hooks` to the tensors to intercept
normal gradient computation. All available built-in rules can be found in
:py:mod:`zennit.rules`, some of which are:

* :py:class:`~zennit.rules.Epsilon`, the most basic LRP rule
* :py:class:`~zennit.rules.AlphaBeta`, an LRP rule that splits positive and
  negative contributions and weights them
* :py:class:`~zennit.rules.ZPlus`, an LRP rule that only takes positive
  contributions into account
* :py:class:`~zennit.rules.Gamma`, an LRP rule that amplifies the positive
  weights
* :py:class:`~zennit.rules.ZBox`, an LRP rule for bounded inputs, like pixel
  space, commonly used in the first layer of a network
* :py:class:`~zennit.rules.ReLUGuidedBackprop`, a rule for ReLU activations to
  only propagate positive gradients back
* :py:class:`~zennit.rules.Norm`, a rule which distributes the gradient weighted
  by each output's fraction of the full output
* :py:class:`~zennit.rules.Pass`, a rule which passes the incoming gradient on
  without changing it
* :py:class:`~zennit.rules.Flat`, a rule meant for linear (dense, convolutional)
  layers to equally distribute relevance as if inputs and weights were constant

Rules can be instantiated, after which they may be used directly:

.. code-block:: python

    import torch
    from torch.nn import Conv2d
    from zennit.rules import Epsilon

    # instantiate the rule
    rule = Epsilon(epsilon=1e-5)
    conv_layer = Conv2d(3, 10, 3, padding=1)

    # registering a rule adds hooks to the module which temporarily overwrites
    # its gradient computation; handles are returned to remove the hooks to undo
    # the modification
    handles = rule.register(conv_layer)

    # to compute the gradient (i.e. the attribution), requires_grad must be True
    input = torch.randn(1, 3, 32, 32, requires_grad=True)
    output = conv_layer(input)

    # torch.autograd.grad returns a tuple, the comma after `attribution`
    # unpacks the single element in the tuple; the `grad_outputs` are necessary
    # for non-scalar outputs, and can be used to target which output should be
    # attributed for; `ones_like` targets all outputs
    attribution, = torch.autograd.grad(
        output, input, grad_outputs=torch.ones_like(output)
    )

    # remove the hooks
    handles.remove()

See :doc:`/how-to/write-custom-rules` for further technical detail on how to
write custom rules.

Note that some rules, in particular the ones that modify parameters (e.g.
:py:class:`~zennit.rules.ZPlus`, :py:class:`~zennit.rules.AlphaBeta`, ...)
are not thread-safe in the backward-phase, because they modify the model
parameters for a brief moment. For most users, this is unlikely to cause any
problems, and may be avoided by using locks in appropriate locations.


.. _use-composites:

Composites
----------

For a model with multiple layers, it may be inconvenient to register
each rule individually. Therefore, **Composites** are used to map rules to
layers given various criterions. **Composites** also take care of registering
all models, and removing their handles after use.
All available **Composites** can be found in :py:mod:`zennit.composites`.

Some built-in composites implement rule-mappings needed for some common
attribution methods, some of which are

* :py:class:`~zennit.composites.EpsilonPlus`, which uses
  :py:class:`~zennit.rules.ZPlus` for convolutional layers and
  :py:class:`~zennit.rules.Epsilon` for densely connected linear layers
* :py:class:`~zennit.composites.EpsilonAlpha2Beta1`, which uses
  :py:class:`~zennit.rules.AlphaBeta`\ ``(alpha=2, beta=1)`` for convolutional and
  :py:class:`~zennit.rules.Epsilon` for densely connected linear layers
* :py:class:`~zennit.composites.EpsilonPlusFlat` and
  :py:class:`~zennit.composites.EpsilonAlpha2Beta1Flat`, which, extending the
  previous two composites respectively, use the :py:class:`~zennit.rules.Flat`
  rule for the first linear (convolution or dense) layer
* :py:class:`~zennit.composites.EpsilonGammaBox`, which uses
  :py:class:`~zennit.rules.Gamma`\ ``(gamma=0.25)`` for convolutional layers,
  :py:class:`~zennit.rules.Epsilon` for dense linear layers, and
  :py:class:`~zennit.rules.ZBox` for the first linear (dense, convolutional)
  layer
* :py:class:`~zennit.composites.GuidedBackprop`, which implements Guided
  Backpropagation by using the :py:class:`~zennit.rules.GuidedBackprop` rule
  for all ReLUs
* :py:class:`~zennit.composites.ExcitationBackprop`, which implements Excitation
  Backpropagation, by using :py:class:`~zennit.rules.ZPlus` for linear (dense
  or convolutional) layers

Additionally, the :py:class:`~zennit.rules.Norm` rule, which normalizes the
gradient by output fraction, is used for :py:class:`~zennit.layer.Sum` and
:py:class:`~zennit.types.AvgPool` layers in all of the listed **Composites**
except for :py:class:`~zennit.composites.GuidedBackprop`.

Since the gradient is only *overwritten* by **Rules**, the gradient will be
unchanged for layers without applicable rules. If layers should only pass their
received gradient/relevance on, the :py:class:`~zennit.rules.Pass` rule should
be used (which is done for all activations in all LRP **Composites**, but not
in :py:class:`~zennit.composites.GuidedBackprop` or
:py:class:`~zennit.composites.ExcitationBackprop`).

Note on **MaxPool**: For LRP, the gradient of MaxPool assigns values only to the
*largest* inputs (winner-takes-all), which is already the expected behaviour for
LRP rules.

Composites may require arguments, e.g.
:py:class:`~zennit.composites.EpsilonGammaBox` requires keyword arguments
``high`` and ``low`` to specify the bounds of the first layer's
:py:class:`~zennit.rules.ZBox`.

.. code-block:: python

    import torch
    from torch.nn import Sequential, Conv2d, ReLU, Linear, Flatten
    from zennit.composites import EpsilonGammaBox

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
    # sigma of normal distribution, just for visual purposes
    sigma = 1.
    # some random input data, still requires grad
    input = torch.randn(1, 3, 32, 32, requires_grad=True) * sigma

    # low and high values for ZBox need to be Tensors in the shape of the input
    # the batch-dimension may be chosen larger, to support different sizes
    composite = EpsilonGammaBox(
        low=torch.full_like(input, -3 * sigma),
        high=torch.full_like(input, 3 * sigma)
    )

There are two basic ways using only the **Composite** to register the modules,
either using :py:func:`~zennit.core.Composite.register`:

.. code-block:: python

    # register hooks for rules to all modules that apply
    composite.register(model)
    # execute the hooked/modified model
    output = model(input)
    # compute the attribution via the gradient
    attribution, = torch.autograd.grad(
        output, input, grad_outputs=torch.ones_like(output)
    )
    # remove all hooks, undoing the modification
    composite.remove()

and using :py:func:`~zennit.core.Composite.context`:

.. code-block:: python

    # register hooks for rules to all modules that apply within the context
    # note that model and modified_model are the same model, the context
    # variable is purely visual
    # hooks are removed when the context is exited
    with composite.context(model) as modified_model:
        # execute the hooked/modified model
        output = modified_model(input)
        # compute the attribution via the gradient
        attribution, = torch.autograd.grad(
            output, input, grad_outputs=torch.ones_like(output)
        )

There is a third option using :py:class:`zennit.attribution.Attributor`, which is
explained in :doc:`/how-to/use-attributors`.

Finally, there are abstract **Composites** which may be used to specify custom
**Composites**:

* :py:class:`~zennit.composites.LayerMapComposite`, which maps module types to
  rules
* :py:class:`~zennit.composites.SpecialFirstLayerMapComposite`, which also maps
  module types to rules, with a special mapping for the first layer
* :py:class:`~zennit.composites.NameMapComposite`, which maps module names to
  rules

For example, the built-in :py:class:`~zennit.composites.EpsilonPlus` composite
may be written like the following:

.. code-block:: python

    from zennit.composites import LayerMapComposite
    from zennit.rules import Epsilon, ZPlus, Norm, Pass
    from zennit.types import Convolution, Activation, AvgPool

    # the layer map is a list of tuples, where the first element is the target
    # layer type, and the second is the rule template
    layer_map = [
        (Activation, Pass()),  # ignore activations
        (AvgPool, Norm()),  # normalize relevance for any AvgPool
        (Convolution, ZPlus()),  # any convolutional layer
        (Linear, Epsilon(epsilon=1e-6))  # this is the dense Linear, not any
    ]
    composite = LayerMapComposite(layer_map=layer_map)

Note that rules used in composites are only used as templates and copied for
each layer they apply to using :py:func:`zennit.core.Hook.copy`.
If we want to map the :py:class:`~zennit.rules.ZBox` rule to the first
convolutional layer, we can use
:py:class:`~zennit.composites.SpecialFirstLayerMapComposite` instead:

.. code-block:: python

    from zennit.composites import SpecialFirstLayerMapComposite
    from zennit.rules import ZBox
    # abstract base class to describe convolutions + dense linear layers
    from zennit.types import Linear as AnyLinear

    # shape of our data
    shape = (1, 3, 32, 32)
    low = torch.full(shape, -3)
    high = torch.full(shape, 3)
    # the first map is only used once, to the first module which applies to the
    # map, i.e. here the first layer of type AnyLinear
    first_map = [
        (AnyLinear, ZBox(low, high))
    ]
    # layer_map is used from the previous example
    composite = SpecialFirstLayerMapComposite(
        layer_map=layer_map, first_map=first_map
    )

If a composite is made to apply for a single model, a
:py:class:`~zennit.composites.NameMapComposite` can provide a transparent
mapping from module name to rule:

.. code-block:: python

    from collections import OrderedDict
    from zennit.composites import NameMapComposite

    # setup the model, explicitly naming them
    model = Sequential(OrderedDict([
        ('conv0', Conv2d(3, 8, 3, padding=1)),
        ('relu0', ReLU()),
        ('conv1', Conv2d(8, 16, 3, padding=1)),
        ('relu1', ReLU()),
        ('flatten', Flatten()),
        ('linear0', Linear(16 * 32 * 32, 1024)),
        ('relu2', ReLU()),
        ('linear1', Linear(1024, 10)),
    ]))

    # look at the available modules
    print(list(model.named_modules()))

    # manually write a rule mapping:
    composite = NameMapComposite([
        (['conv0'], ZBox(low, high)),
        (['conv1'], ZPlus()),
        (['linear0', 'linear1'], Epsilon()),
    ])

Modules built using :py:class:`torch.nn.Sequential` without explicit names will have a
number string as their name. Explicitly assigning a module to a parent module as
an attribute will assign the attribute as the child module's name. Nested
modules will have their names split by a dot ``.``.

To create custom composites following more complex patterns, see
:doc:`/how-to/write-custom-compositors`.


.. _use-canonizers:

Canonizers
----------

Layerwise relevance propagation (LRP) is not implementation invariant.
A good example for this is that for some rules, two consecutive linear layers do
not produce the same attribution as a single linear layer with its weight
parameter chosen as the product of the two linear layers.
The most common case this happens is when models use
:py:class:`~zennit.types.BatchNorm`, which is commonly used directly after, or
sometimes before a linear (dense, convolutional) layer.
**Canonizers** are used to avoid this by temporarily
enforcing a canonical form of the model. They differ from **Rules** in that the
model is actively changed while the :py:class:`~zennit.canonizers.Canonizer` is
registered, as opposed to using hooks to modify the gradient during runtime.

All available **Canonizers** can be found in :py:mod:`zennit.canonizers`.
Some of the available basic ones are:

* :py:class:`~zennit.canonizers.SequentialMergeBatchNorm`, which traverses the
  tree of submodules in-order to create a sequence of leave modules, which is
  then used to detect adjacent linear (dense, convolutional) and BatchNorm modules
* :py:class:`~zennit.canonizers.NamedMergeBatchNorm`, which is used to specify
  explicitly by module name which linear (dense, convolutional) and BatchNorm
  modules should be merged
* :py:class:`~zennit.canonizers.AttributeCanonizer`, which expects a function
  mapping from module name and type to a :py:class:`dict` of attribute names
  and values which should be changed for applicable modules
* :py:class:`~zennit.canonizers.CompositeCanonizer`, which expects a list of
  canonizers which are then combined to a single canonizer

:py:class:`~zennit.canonizers.SequentialMergeBatchNorm` traverses the module
tree in-order leaves-only using :py:func:`zennit.core.collect_leaves`, and
iterates the resulting list to detect adjacent linear (dense, convolutional) and
batch-norm modules. The batch-norm's scale and shift are merged into the
adjacent linear layer's weights and bias.

While not recommended, **Canonizers** can be used on their own:

.. code-block:: python

    import torch
    from torch.nn import Sequential, Conv2d, ReLU, Linear, Flatten, BatchNorm2d
    from zennit.canonizers import SequentialMergeBatchNorm

    # setup the model
    model = Sequential(
        Conv2d(3, 8, 3, padding=1),
        ReLU(),
        Conv2d(8, 16, 3, padding=1),
        BatchNorm2d(16),
        ReLU(),
        Flatten(),
        Linear(16 * 32 * 32, 1024),
        ReLU(),
        Linear(1024, 10),
    )

    # create the canonizer
    canonizer = SequentialMergeBatchNorm()

    # apply the canonizer to the model, which creates multiple canonizer
    # instances, one per applicable case
    instances = canonizer.apply(model)

    # do something with the model
    input = torch.randn(1, 3, 32, 32)
    output = model(input)

    # remove the canonizer instances to revert the model to its original state
    for instance in instances:
        instance.remove()

However, the recommended way is to use them with **Composites**, which will
apply and remove canonizer instances automatically while the **Composite** is
active:

.. code-block:: python

    from zennit.composites import EpsilonPlusFlat

    # create the canonizer
    canonizer = SequentialMergeBatchNorm()
    # create the composite, with the canonizer as an argument
    composite = EpsilonPlusFlat(canonizers=[canonizer])
    # create some input data
    input = torch.randn(1, 3, 32, 32, requires_grad=True)
    # register the composite within the context, which also applies the
    # canonizer
    with composite.context(model) as modified_model:
        output = modified_model(input)
        # compute the attribution
        attribution, = torch.autograd.grad(output, input, torch.eye(10)[[0]])

    # print the absolute sum of the attribution
    print(attribution.abs().sum().item())

Be careful not to accidentally save a model's parameters (e.g. using
``model.state_dict()``) while **Canonizers** are applied, as this will store the
modified state of the model.

Some models implemented in :py:mod:`torchvision.models` have their own specific
**Canonizer** implemented in :py:mod:`zennit.torchvision`, which currently
are:

* :py:class:`~zennit.torchvision.VGGCanonizer`, which applies to
  :py:mod:`torchvision`'s implementation of VGG networks and currently is an alias for
  :py:class:`~zennit.canonizers.SequentialMergeBatchNorm`
* :py:class:`~zennit.torchvision.ResNetCanonizer`, which applies to
  :py:mod:`torchvision`'s implementation of ResNet networks, which merges BatchNorms
  and replaces residual connections with the explicit
  :py:class:`~zennit.layer.Sum` module, which makes it possible to assign a rule
  to the residual connection.

More technical detail to implement custom **Canonizers** may be found in
:doc:`/how-to/write-custom-canonizers`.
