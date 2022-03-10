=========================
Writing Custom Canonizers
=========================

**Canonizers** are used to temporarily transform models into a canonical form to
mitigate the lack of implementation invariance of methods Layerwise Relevance
Propagation (LRP). A general introduction to **Canonizers** can be found here:
:ref:`use-canonizers`.

As both **Canonizers** and **Composites** (via **Rules**) change the outcome of
the attribution, it can be a little bit confusing in the beginning when
challenged with the question whether a novel network architectures needs a new
set of **Rules** and **Composites**, or if it should be adapted to the existing
framework using **Canonizers**. While ultimately it depends on the design
preference of the developer, our suggestion is to go through the following steps
in order:

1. Check whether a custom **Composite** is enough to correctly attribute the
   model, i.e. the new layer-type is only a composition of existing layer types
   without any unaccounted intermediate steps or incapabilities with existing
   rules.
2. If some of the rules which should be used are incompatible without changes
   (e.g. subsequent linear layers), or some parts of a module has intermediate
   computations that are not implemented with sub-modules, it should be checked
   whether a **Canonizer** can be implemented to fix these issues. If you are in
   control of the module in question, check whether rewriting the module with
   sub-modules is easier than implementing a **Canonizer**.
3. If the module consists of computations which cannot be separated into
   existing modules with compatible rules, or would result in an overly complex
   architecture, a custom **Rule** may be the choice to go with.

**Rules** and **Composites** are not designed to change the forward computation
of a model. While **Canonizers** can change the outcome of the forward pass,
this should be used with care, since a modified function output means that the
function itself has been modified, which will therefore result in an attribution
of the modified function instead.

To implement a custom **Canonizer**, a class inheriting from
:py:class:`zennit.canonizers.Canonizer` needs to implement the following four
methods:

* :py:meth:`~zennit.canonizers.Canonizer.apply`, which finds the sub-modules
  that should be modified by the **Canonizer** and passes their information to ...
* :py:meth:`~zennit.canonizers.Canonizer.register`, which copies the current
  instance using :py:meth:`~zennit.canonizers.Canonizer.copy`, applies the
  changes that should be introduced by the **Canonizer**, and makes sure they
  can be reverted later, using ...
* :py:meth:`~zennit.canonizers.Canonizer.remove`, which reverts the changes
  introduced by the **Canonizer**, by i.e. loading the original parameters which
  were temporarily stored, and
* :py:meth:`~zennit.canonizers.Canonizer.copy`, which copies the current
  instance, to create an individual instance for each applicable module with the
  same parameters.

Suppose we have a ReLU model (e.g. VGG11) for which we want to compute the
second-order derivative, e.g. to find an adversarial explanation (see
:cite:p:`dombrowski2019explanations`). The ReLU is not differentiable at 0, and
its second order derivative is zero everywhere except at 0, where it is
undefined. :cite:t:`dombrowski2019explanations` replace the ReLU activations in
a model with *Softplus* activations, which when running *beta* towards infinity
will be identical to the ReLU activation. For the numerical estimate, it is
enough to set *beta* to a relatively large value, e.g. to 10. The following is
an implementation of the **SoftplusCanonizer**, which will temporarily replace
the ReLU activations in a model with Softplus activations:

.. code-block:: python

    import torch

    from zennit.canonizers import Canonizer


    class SoftplusCanonizer(Canonizer):
        '''Replaces ReLUs with Softplus units.'''
        def __init__(self, beta=10.):
            self.beta = beta
            self.module = None
            self.relu_children = None

        def apply(self, root_module):
            '''Iterate all modules under root_module and register the Canonizer
            if they have immediate ReLU sub-modules.
            '''
            # track the SoftplusCanonizer instances to remove them later
            instances = []
            # iterate recursively over all modules
            for module in root_module.modules():
                # get all the direct sub-module instances of torch.nn.ReLU
                relu_children = [
                    (name, child)
                    for name, child in module.named_children()
                    if isinstance(child, torch.nn.ReLU)
                ]
                # if there is at least on direct ReLU sub-module
                if relu_children:
                    # create a copy (with the same beta parameter)
                    instance = self.copy()
                    # register the module
                    instance.register(module, relu_children)
                    # add the copy to the instance list
                    instances.append(instance)
            return instances

        def register(self, module, relu_children):
            '''Store the module and the immediate ReLU-sub-modules, and then
            overwrite the attributes that corresponds to each ReLU-sub-modules
            with a new instance of ``torch.nn.Softplus``.
            '''
            self.module = module
            self.relu_children = relu_children
            for name, _ in relu_children:
                # set each of the attributes corresponding to the ReLU to a new
                # instance of toch.nn.Softplus
                setattr(module, name, torch.nn.Softplus(beta=self.beta))

        def remove(self):
            '''Undo the changes introduces by this Canonizer, by setting the
            appropriate attributes of the stored module back to the original
            ReLU sub-module instance.
            '''
            for name, child in self.relu_children:
                setattr(self.module, name, child)

        def copy(self):
            '''Create a copy of this instance. Each module requires its own
            instance to call ``.register``.
            '''
            return SoftplusCanonizer(beta=self.beta)


Note that we can only replace modules by changing their immediate parent. This
means that if ``root_module`` was a ``torch.nn.ReLU`` itself, it would be
impossible to replace it with a ``torch.nn.Softplus`` without replacing the
``root_module`` itself.

For demonstration purposes, we can compute the gradient w.r.t. a loss which uses
the gradient of the modified model in the following way:

.. code-block:: python

    import torch
    from torchvision.models import vgg11

    from zennit.core import Composite
    from zennit.image import imgify


    # create a VGG11 model with random parameters
    model = vgg11()
    # use the Canonizer with an "empty" Composite (without specifying
    # module_map), which will not attach rules to any sub-module, thus resulting
    # in a plain gradient computation, but with a Canonizer applied
    composite = Composite(
        canonizers=[SoftplusCanonizer()]
    )

    input = torch.randn(1, 3, 224, 224, requires_grad=True)
    target = torch.eye(1000)[[0]]
    with composite.context(model) as modified_model:
        out = modified_model(input)
        relevance, = torch.autograd.grad(out, input, target, create_graph=True)
        # find adversarial example such that input and its respective
        # attribution are close
        loss = ((relevance - input.detach()) ** 2).mean()
        # compute the gradient of input w.r.t. loss, using the second order
        # derivative w.r.t. input; note that this currently does not work when
        # using BasicHook, which detaches the gradient to avoid wrong values
        adv_grad, = torch.autograd.grad(loss, input)

    # visualize adv_grad
    imgify(adv_grad[0].abs().sum(0), cmap='hot').show()

