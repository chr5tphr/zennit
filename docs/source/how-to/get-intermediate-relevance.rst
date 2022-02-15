==============================
Getting Intermediate Relevance
==============================

In some cases, intermediate gradients or relevances of a model may be needed.
Since Zennit uses Pytorch's autograd engine, intermediate relevances can be
retained simply as the intermediate gradients of accessible non-leaf tensors
in the tensor's ``.grad`` attribute by calling ``tensor.retain_grad()`` before
the gradient computation.

In most cases when using ``torch.nn.Module``-based models, the intermediate
outputs are not easily accessible, which we can solve by using forward-hooks.

We create following setting with some random input data and a simple, randomly
initialized model, for which we want to compute the LRP EpsilonPlus relevance:

.. code-block:: python

    import torch
    from torch.nn import Sequential, Conv2d, ReLU, Linear, Flatten

    from zennit.attribution import Gradient
    from zennit.composites import EpsilonPlusFlat

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

    # create a composite instance
    composite = EpsilonPlusFlat()

    # create a gradient attributor
    attributor = Gradient(model, composite)

Now we create a function ``store_hook`` which we register as a forward hook to
all modules. The function sets the module's attribute ``.output`` to its output
tensor, and ensures the gradient is stored in the tensor's ``.grad`` attribute
even if it is not a leaf-tensor by using ``.retain_grad()``.

.. code-block:: python

    # create a hook to keep track of intermediate outputs
    def store_hook(module, input, output):
        # set the current module's attribute 'output' to the its tensor
        module.output = output
        # keep the output tensor gradient, even if it is not a leaf-tensor
        output.retain_grad()

    # enter the attributor's context to register the rule-hooks
    with attributor:
        # register the store_hook AFTER the rule-hooks have been registered (by
        # entering the context) so we get the last output before the next module
        handles = [
            module.register_forward_hook(store_hook) for module in model.modules()
        ]
        # compute the relevance wrt. output/class 1
        output, relevance = attributor(input, torch.eye(10)[[1]])

    # remove the hooks using store_hook
    for handle in handles:
        handle.remove()

    # print the gradient tensors for demonstration
    for name, module in model.named_modules():
        print(f'{name}: {module.output.grad}')

The hooks are registered within the attributor's with-context, such that they
are applied after the rule hooks. Once we are finished, we can remove the
store-hooks by calling ``.remove()`` on all handles returned when registering the
hooks.

Be aware that storing the intermediate outputs and their gradients may require
significantly more memory, depending on the model. In practice, it may be better
to register the store-hook only to modules for which the relevance is needed.
