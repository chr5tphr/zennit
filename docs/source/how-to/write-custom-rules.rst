====================
Writing Custom Rules
====================

**Rules** overwrite the gradient of specific modules. An introduction to their
usage may be found under :ref:`use-rules`.
A number of rules, specifically for Layer-wise Relevance Propagation
:cite:p:`bach2015pixel` for common layers and networks, are available in Zennit.
For novel or custom architectures and layer-types, as well as for custom
layer-wise attribution methods or rules, it may become necessary to implement a
custom **Rule**.
For this case, Zennit provides the :py:class:`~zennit.core.Hook` class, with a
straight-forward interface to change or create side-effects during the forward
and backward passes.


General Rules
-------------

In most cases, if simply the gradient should be overwritten, it is enough
inherit from :py:class:`~zennit.core.Hook` and implement the
:py:meth:`~zennit.core.Hook.backward` method.
The :py:meth:`~zennit.core.Hook.backward` method has three arguments:

* ``module``, which is the current module to which the hook has been registered,
* ``grad_input``, which is the gradient of the output of the full gradient chain
  with respect to the module's input (which is the gradient of the module wrt.
  its input multiplied by ``grad_output``), and
* ``grad_output``, which is the gradient of the output of the full gradient
  chain with respect to the module's output.

If we define ``module`` as :math:`f:\mathbb{R}^{d_\text{in}} \rightarrow
\mathbb{R}^{d_\text{out}}` and the function after ``module`` :math:`g:\mathbb{R}^{d_\text{out}}
\rightarrow \mathbb{R}^{d_\text{after}}` and the input :math:`x\in\mathbb{R}^{d_\text{in}}` and
output :math:`y = g(f(x))`, with the chain rule we get

.. math::

    \frac{\partial g(f(x))}{\partial x} =
    \frac{\partial f(x)}{\partial x}
    \frac{\partial g(f(x))}{\partial f(x)}`

where ``grad_input`` is :math:`\frac{\partial g(f(x))}{\partial x}` and
``grad_output`` is :math:`\frac{\partial g(f(x))}{\partial f(x)}``.

Returning a value in the implementation of
:py:meth:`~zennit.core.Hook.backward` overwrites the full gradient
:math:`\frac{\partial g(f(x))}{\partial x}`` within the chain, which will
become ``grad_output`` for modules before the current one.
Usually, the current ``grad_output`` is multiplied with a modified
gradient of ``module``, thus using the values after the current module (in forward
perspective) in the chain, keeping the graph connected.
:py:meth:`~zennit.core.Hook.backward` is always called *after* the gradient of
the module with respect to its input has been computed, thus making
``grad_input`` available.

We can, for example, implement a rule which ignores the gradient of the
current module:

.. code-block:: python

    import torch
    from zennit.core import Hook


    class Ignore(Hook):
        '''Ignore the module's gradient and pass through the output gradient.'''
        def backward(self, module, grad_input, grad_output):
            '''Directly return grad_output.'''
            return grad_output


    ignore_hook = Ignore()

    input = torch.randn(1, 4, requires_grad=True)
    module = torch.nn.ReLU()

    handles = ignore_hook.register(module)
    output = module(input)
    grad, = torch.autograd.grad(output, input, torch.ones_like(output))
    handles.remove()

This particular rule is already included as :py:class:`zennit.rules.Pass`, and,
for the layer-wise relevance propagation (LRP)-based **Composites**, used for
all activations.
:py:class:`~zennit.core.Hook` has a dictionary attribute ``stored_tensors``,
which is used to store the output gradient as ``stored_tensors['grad_output']``.
:py:meth:`~zennit.core.Hook.forward` has 3 arguments:

* ``module``, which is the current module the hook has been registered to,
* ``input``, which is the module's input tensor, and
* ``output``, which is the module's output tensor.

:py:meth:`~zennit.core.Hook.forward` is always called *after* the forward has
been called, thus making ``output`` available.
Using the notation above, ``input`` is :math:`x` and ``output`` is :math:`f(x)`.

A layer-wise *gradient times input* can be implemented by storing the input
tensor in the forward pass and directly using ``grad_input`` in the backward
pass:

.. code-block:: python

    import torch
    from zennit.core import Hook


    class GradTimesInput(Hook):
        '''Hook for layer-wise gradient times input.'''
        def forward(self, module, input, output):
            '''Remember the input for the backward pass.'''
            self.stored_tensors['input'] = input

        def backward(self, module, grad_input, grad_output):
            '''Modify the gradient by element-wise multiplying the input.'''
            return (self.stored_tensors['input'][0] * grad_input[0],)


    grad_times_input_hook = GradTimesInput()

    input = torch.randn(1, 4, requires_grad=True)
    module = torch.nn.Linear(4, 4)

    handles = grad_times_input_hook.register(module)
    output = module(input)
    grad, = torch.autograd.grad(output, input, torch.ones_like(output))
    handles.remove()

The elements of ``stored_tensors`` will be removed once
:py:meth:`~zennit.core.Hook.remove` is called, e.g. when the context of the
**Rule**'s **Composite** is left.
Returning ``None`` in :py:meth:`~zennit.core.Hook.forward` (like implicitly
above) will not modify the output.
This is also the case for :py:meth:`~zennit.core.Hook.backward` and the
gradient.

When Hooks are not directly registered, which is the usual case, they will be
used as templates and copied by **Composites** using
:py:meth:`zennit.core.Hook.copy`. The default ``copy()`` function will
instantiate a new instance of the **Hook**'s direct ``type()`` without any arguments.
If a **Hook** subtype implements a custom ``__init__()`` or otherwise has
parameters that need to be copied, a ``copy()`` function needs to be
implemented.

As an example, if we implement a *gradient times input* where the negative part
of the input is scaled by some parameter:

.. code-block:: python

    import torch
    from zennit.core import Hook


    class GradTimesScaledNegativeInput(Hook):
        '''Gradient times input, where the negative part of the input is
        scaled.
        '''
        def __init__(self, scale=0.2):
            super().__init__()
            self.scale = scale

        def forward(self, module, input, output):
            '''Remember the input for the backward pass.'''
            self.stored_tensors['input'] = input

        def backward(self, module, grad_input, grad_output):
            '''Modify the gradient by element-wise multiplication of the input,
            but with the negative part of the input scaled.
            '''
            return (
                grad_input[0] * (
                    self.stored_tensors['input'][0].clip(min=0.0)
                    + self.stored_tensors['input'][0].clip(max=0.0) * self.scale
                )
            ,)

        def copy(self):
            return self.__class__(scale=self.scale)


    scaled_negative_hook = GradTimesScaledNegativeInput(scale=0.23)
    hook_copy = scaled_negative_hook.copy()
    assert scaled_negative_hook.scale == hook_copy.scale

    input = torch.randn(1, 4, requires_grad=True)
    module = torch.nn.Linear(4, 4)

    handles = scaled_negative_hook.register(module)
    output = module(input)
    grad, = torch.autograd.grad(output, input, torch.ones_like(output))
    handles.remove()

Here, ``self.__class__`` returns the direct class of ``self``, which is
``GradTimesScaledNegativeInput`` unless a subtype of our class was created, and
is called with the scale keyword argument to create a proper copy of our hook.
An alternative is to use :py:func:`copy.deepcopy`, however, in *Zennit*'s
implementation of **Hooks** we decided in favor of a transparent per-hook copy
method instead.

LRP-based Rules
---------------

While it introduces a little more complexity, :py:class:`~zennit.core.BasicHook`
abstracts over the components of all LRP-based **Rules**.
The components are split into 3 :math:`K`-tuples of functions, and 2 *single*
functions:

* ``input_modifiers``, which is a tuple of :math:`K` functions, each with a
  single argument to modify the input tensor,
* ``param_modifiers``, which is a tuple of :math:`K` functions or
  :py:class:`~zennit.core.ParamMod` instances, each with two arguments, the
  parameter tensor ``obj`` and its name ``name`` (e.g. ``weight`` or ``bias``),
  to modify the parameter,
* ``output_modifiers``, which is a tuple of :math:`K` functions, each with a
  single argument to modify the output tensor, each produced by applying the
  module with a modified input and its respective modified parameters,
* ``gradient_mapper``, which is a single function which produces a tuple of
  :math:`K` tensors with two arguments: the gradient with respect to the
  module's gradient ``grad_output`` and a :math:`K`-tuple of the modified
  outputs ``outputs``, and
* ``reducer``, which is a single function with two arguments, a :math:`K`-tuple
  of the modified inputs, and a :math:`K`-tuple of the vector-Jacobian product
  of each element of the output of ``gradient_mapper`` and the Jacobian of each
  modified output with respect to its modified input.

Formally, :py:meth:`~zennit.core.Hook.backward` computes the modified gradient
:math:`R_\text{in}\in\mathbb{R}^{d_\text{out}}` as

.. math::
   :nowrap:

    \begin{align}
    x^{(k)} &= h^{(k)}_\text{in}(x)
        &x^{(k)}\in\mathbb{R}^{d_\text{in}} \\
    y^{(k)} &= h^{(k)}_\text{out}\big( f(x^{(k)};h^{(k)}_\text{param}(\theta)) \big)
        &y^{(k)}\in\mathbb{R}^{d_\text{out}} \\
    \gamma^{(k)} &= \Big[ h_\text{gmap}\big( R_\text{out}; (y^{(1)}, ..., y^{(K)}) \big) \Big]^{(k)}
        &\gamma^{(k)}\in\mathbb{R}^{d_\text{out}} \\
    v^{(k)} &= \Big( \frac{\partial y^{(k)}}{\partial x^{(k)}} \Big)^\top \gamma^{(k)}
        &v^{(k)}\in\mathbb{R}^{d_\text{in}} \\
    R_\text{in} &= h_\text{reduce}\Big[
        (x^{(1)}, v^{(1)}); ...; (x^{(K)}, v^{(K)})
    \Big]
    \end{align}

where input :math:`x\in\mathbb{R}^{d_\text{in}}` with input dimensionality
:math:`d_\text{in}`,
``module`` function :math:`f: \mathbb{R}^{d_\text{in}} \times
\mathbb{R}^{d_\text{params}} \rightarrow \mathbb{R}^{d_\text{out}}` with
parameters :math:`\theta \in \mathbb{R}^{d_\text{params}}`,
``grad_output`` :math:`R_\text{out}\in\mathbb{R}^{d_\text{out}}`,
:math:`\big[\cdot\big]^{(k)}` denotes the element at index :math:`k` of the
tuple within brackets,
:math:`\frac{\partial y^{(k)}}{\partial x^{(k)}} \in
\mathbb{R}^{d_\text{out}\times d_\text{in}}` is the Jacobian,
:math:`K`-tuple functions with :math:`k\in\{1,...,K\}`:

* input modifiers :math:`h^{(k)}_\text{in}: \mathbb{R}^{d_\text{in}}
  \rightarrow \mathbb{R}^{d_\text{in}}`,
* output modifiers :math:`h^{(k)}_\text{out}: \mathbb{R}^{d_\text{out}}
  \rightarrow \mathbb{R}^{d_\text{out}}`, and
* parameter modifiers :math:`h^{(k)}_\text{param}: \mathbb{R}^{d_\text{params}}
  \rightarrow \mathbb{R}^{d_\text{params}}`,

and single functions

* output gradient map :math:`h_\text{gmap}: \mathbb{R}^{d_\text{out}}
  \times(\mathbb{R}^{d_\text{out}})^K \rightarrow
  (\mathbb{R}^{d_\text{out}})^K`, and
* combined input and gradient reduce function :math:`h_\text{reduce}:
  (\mathbb{R}^{d_\text{in}} \times \mathbb{R}^{d_\text{in}})^K \rightarrow
  \mathbb{R}^{d_\text{in}}`.

With this abstraction, the basic, unstabilized LRP-0 Rule can be implemented
using

.. code-block:: python

    import torch
    from zennit.core import BasicHook


    lrp_zero_hook = BasicHook(
         input_modifiers=[lambda input: input],
         param_modifiers=[lambda param, _: param],
         output_modifiers=[lambda output: output],
         gradient_mapper=(lambda out_grad, outputs: [out_grad / outputs[0]]),
         reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
    )

    input = torch.randn(1, 4, requires_grad=True)
    module = torch.nn.Linear(4, 4)

    handles = lrp_zero_hook.register(module)
    output = module(input)
    grad, = torch.autograd.grad(output, input, torch.ones_like(output))
    handles.remove()

This creates a single, usable hook, which can be copied with
:py:meth:`zennit.core.BasicHook.copy`. The number of modifiers here is only 1,
thus the modifiers are each a list of a single function, and the function for
``gradient_mapper`` only returns a list with a single element (here, it would
also be valid to return a single element).
The reducer has to return a single tensor in the end, which means that when
there is more than 1 modifier each, ``inputs`` and ``gradients`` need to be
reduced e.g. by summation.
The default parameters for each modifier will be the identity, and specifying
only one modifier with more than one function will automatically add more
identity functions for the unspecified modifiers.
The default gradient mapper is the ``tuple(out_grad / stabilize(output) for
output in outputs)``, and the default reducer is ``sum(input * gradient for
input, gradient in zip(inputs, gradients))``.
This means that creating a :py:class:`~zennit.core.BasicHook` only with default
arguments will result in the :py:class:`~zennit.rules.Epsilon` -Rule with a
default epsilon-value which cannot be specified.

.. code-block:: python

    import torch
    from zennit.core import BasicHook


    epsilon_hook = BasicHook()

    input = torch.randn(1, 4, requires_grad=True)
    module = torch.nn.Linear(4, 4)

    handles = epsilon_hook.register(module)
    output = module(input)
    grad, = torch.autograd.grad(output, input, torch.ones_like(output))
    handles.remove()

As another example, the :py:class:`~zennit.rules.ZPlus` -Rule for ReLU-networks
with strictly positive inputs can be implemented as

.. code-block:: python

    import torch
    from zennit.core import BasicHook


    class ZPlusReLU(BasicHook):
        '''LRP-ZPlus Rule for strictly negative inputs. All parameters not
        listed in names will be set to zero.
        '''
        def __init__(self, names=None):
            self.names = [] if names is None else names
            super().__init__(
                 param_modifiers=[self._param_modifier],
            )

        def _param_modifier(self, param, name):
            '''Only take the positive part of parameters specified in
            self.names. Other parameters are set to zero.'''
            if name in self.names:
                return param.clip(min=0.0)
            return torch.zeros_like(param)


    hook = ZPlusReLU(['weight'])

    input = torch.randn(1, 4, requires_grad=True)
    module = torch.nn.Linear(4, 4)

    handles = hook.register(module)
    output = module(input)
    grad, = torch.autograd.grad(output, input, torch.ones_like(output))
    handles.remove()

Here, we first implemented the new rule hook as a class by inheriting from
:py:class:`~zennit.core.BasicHook` and calling ``super().__init__()``.
We also added the argument ``names`` to the ``__init__`` function, and
implemented the single ``_param_modifier`` as a method, such that we can use
``self.names`` inside the modifier function.
This ``_param_modifier`` allows us to choose which parameters of the module
should be used and clipped, by specifying their name in the constructor.
The rest will be set to zero (to not use the bias, for example).
The internal implementation of :py:class:`~zennit.rules.ZPlus` uses two
modifiers in order to take negative input values into account.
We recommend taking a look at the implementation of each rule in
:py:mod:`zennit.rules` for more examples.

For more control over the parameter modification,
:py:class:`~zennit.core.ParamMod` instances may be used in ``param_modifiers``.
A common use-case for this is to specify a number of parameter names which
should be set to zero instead of applying the modification:

.. code-block:: python

    import torch
    from zennit.core import BasicHook, ParamMod


    lrp_zplus_hook = BasicHook(
         param_modifiers=[ParamMod(lambda x, _: x.clip(min=0.), zero_params='bias')],
    )

    input = torch.randn(1, 4, requires_grad=True)
    module = torch.nn.Linear(4, 4)

    handles = lrp_zplus_hook.register(module)
    output = module(input)
    grad, = torch.autograd.grad(output, input, torch.ones_like(output))
    handles.remove()

This is used in all built-in rules based on :py:class:`~zennit.core.BasicHook`,
where the argument ``zero_params`` is passed to all applicable
:py:class:`~zennit.core.ParamMod` arguments.

There are two more arguments to :py:class:`~zennit.core.ParamMod`:

* ``param_keys``, an optional list of parameter names that should be modified,
  which when ``None`` (default), will modify all parameters, and
* ``require_params``, an optional flag to indicate whether the specified
  ``param_keys`` are mandatory (``True``, default). A missing parameter with
  ``param_keys=True`` will cause a ``RuntimeError`` during the backward pass.

During the backward pass inside :py:class:`~zennit.core.BasicHook`, functions
will be internally converted to :py:class:`~zennit.core.ParamMod` with default
parameters.

The built-in rules furthermore introduce subclasses of
:py:class:`~zennit.core.ParamMod` for the common modifiers
:py:class:`~zennit.rules.ClampMod`, :py:class:`~zennit.rules.GammaMod`, and
:py:class:`~zennit.rules.NoMod`.
