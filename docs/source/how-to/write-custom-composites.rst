=========================
Writing Custom Composites
=========================

Zennit provides a number of commonly used **Composites**.
While these are often enough for feed-forward-type neural networks, one primary goal of Zennit is to provide the tools to easily customize the computation of rule-based attribution methods.
This is especially useful to analyze novel architectures, for which no attribution-approach has been designed before.

For most use-cases, using the abstract **Composites** :py:class:`~zennit.composites.LayerMapComposite`, :py:class:`~zennit.composites.SpecialFirstLayerMapComposite`, and :py:class:`~zennit.composites.NameMapComposite` already provides enough freedom to customize which Layer should receive which rule. See :ref:`use-composites` for an introduction.
Depending on the setup, it may however be more convenient to either directly use or implement a new **Composite** by creating a Subclass from :py:class:`zennit.core.Composite`.
In either case, the :py:class:`~zennit.core.Composite` requires an argument ``module_map``, which is a function with the signature ``(ctx: dict, name: str, module: torch.nn.Module) -> Hook or None``, which, given a context dict, the name of a single module and the module itself, either returns an instance of :py:class:`~zennit.core.Hook` which should be copied and registered to the module, or ``None`` if no ``Hook`` should be applied.
The context dict ``ctx`` can be used to track subsequent calls to the ``module_map`` function, e.g. to count the number of processed modules, or to verify if some condition has been met before, e.g. a linear layer has been seen before.
The ``module_map`` is used in :py:meth:`zennit.core.Composite.register`, where the context dict is initialized to an empty dict ``{}`` before iterating over all the sub-modules of the root-module to which the composite will be registered.
The iteration is done using :py:meth:`torch.nn.Module.named_modules`, which will therefore dictate the order modules are visited, which is depth-first in the order sub-modules were assigned.

A simple **Composite**, which only provides rules for linear layers that are leaves and bases the rule on how many leaf modules were visited before could be implemented like the following:


.. code-block:: python

    import torch
    from torchvision.models import vgg16
    from zennit.rules import Epsilon, AlphaBeta
    from zennit.types import Linear
    from zennit.core import Composite
    from zennit.attribution import Gradient


    def module_map(ctx, name, module):
        # check whether there is at least one child, i.e. the module is not a leaf
        try:
            next(module.children())
        except StopIteration:
            # StopIteration is raised if the iterator has no more elements,
            # which means in this case there are no children and module is a leaf
            pass
        else:
            # if StopIteration is not raised on the first element, module is not a leaf
            return None

        # if the module is not Linear, we do not want to assign a hook
        if not isinstance(module, Linear):
            return None

        # count the number of the leaves processed yet in 'leafnum'
        if 'leafnum' not in ctx:
            ctx['leafnum'] = 0
        else:
            ctx['leafnum'] += 1

        # the first 10 leaf-modules which are of type Linear should be assigned
        # the Alpha2Beta1 rule
        if ctx['leafnum'] < 10:
            return AlphaBeta(alpha=2, beta=1)
        # all other rules should be assigned Epsilon
        return Epsilon(epsilon=1e-3)


    # we can then create a composite by passing the module_map function
    # canonizers may also be passed as with all composites
    composite = Composite(module_map=module_map)

    # try out the composite
    model = vgg16()
    with Gradient(model, composite) as attributor:
        out, grad = attributor(torch.randn(1, 3, 224, 224))


A more general **Composite**, where we can specify which layer number and which type should be assigned which rule, can be implemented by creating a class:

.. code-block:: python

    from itertools import islice

    import torch
    from torchvision.models import vgg16
    from zennit.rules import Epsilon, ZBox, Gamma, Pass, Norm
    from zennit.types import Linear, Convolution, Activation, AvgPool
    from zennit.core import Composite
    from zennit.attribution import Gradient


    class LeafNumberTypeComposite(Composite):
        def __init__(self, leafnum_map):
            # pass the class method self.mapping as the module_map
            super().__init__(module_map=self.mapping)
            # set the instance attribute so we can use it in self.mapping
            self.leafnum_map = leafnum_map

        def mapping(self, ctx, name, module):
            # check whether there is at least one child, i.e. the module is not a leaf
            # but this time shorter using itertools.islice to get at most one child
            if list(islice(module.children(), 1)):
                return None

            # count the number of the leaves processed yet in 'leafnum'
            # this time in a single line with get and all layers count, e.g. ReLU
            ctx['leafnum'] = ctx.get('leafnum', -1) + 1

            # loop over the leafnum_map and use the first template for which
            # the module type matches and the current ctx['leafnum'] falls into
            # the bounds
            for (low, high), dtype, template in self.leafnum_map:
                if isinstance(module, dtype) and low <= ctx['leafnum'] < high:
                    return template
            # if none of the leafnum_map apply this means there is no rule
            # matching the current layer
            return None


    # this can be compared with int and will always be larger
    inf = float('inf')

    # we create an example leafnum-map, note that Linear is here
    # zennit.types.Linear and not torch.nn.Linear
    # the first two entries are for demonstration only and would
    # in practice most likely be a single "Linear" with appropriate low/high
    leafnum_map = [
        [(0, 1), Convolution, ZBox(low=-3.0, high=3.0)],
        [(0, 1), torch.nn.Linear, ZBox(low=0.0, high=1.0)],
        [(1, 17), Linear, Gamma(gamma=0.25)],
        [(17, 31), Linear, Epsilon(epsilon=0.5)],
        [(31, inf), Linear, Epsilon(epsilon=1e-9)],
        # catch all activations
        [(0, inf), Activation, Pass()],
        # explicit None is possible e.g. to (ab-)use precedence
        [(0, 17), torch.nn.MaxPool2d, None],
        # catch all AvgPool/MaxPool2d, isinstance also accepts tuples of types
        [(0, inf), (AvgPool, torch.nn.MaxPool2d), Norm()],
    ]

    # finally, create the composite using the leafnum_map
    composite = LeafNumberTypeComposite(leafnum_map)

    # try out the composite
    model = vgg16()
    with Gradient(model, composite) as attributor:
        out, grad = attributor(torch.randn(1, 3, 224, 224))

In practice, however, we do not recommend to use the index of the layer when designing **Composites**, because most of the time, when such a configuration is chosen, it is done to shape the **Composite** for an explicit model.
For these kinds of **Composites**, a :py:class:`~zennit.composites.NameMapComposite` will directly map the name of a sub-module to a Hook, which is a more explicit and transparent way to create a special **Composite** for a single neural network.
