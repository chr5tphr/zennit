================
 Getting started
================


Install
-------

Zennit can be installed directly from PyPI:

.. code-block:: console

   $ pip install zennit

For the current development version, or to try out examples, Zennit may be
alternatively cloned and installed with

.. code-block:: console

   $ git clone https://github.com/chr5tphr/zennit.git
   $ pip install ./zennit

Basic Usage
-----------

Zennit implements propagation-based attribution methods by overwriting the
gradient of PyTorch modules in PyTorch's auto-differentiation engine. This means
that Zennit will only work on models which are strictly implemented using
PyTorch modules, including activation functions. The following demonstrates a
setup to compute Layerwise Relevance Propagation (LRP) relevance for a simple
model and random data.

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

The most important high-level structures in Zennit are ``Composites``,
``Attributors`` and ``Canonizers``.


Composites
^^^^^^^^^^

Composites map ``Rules`` to modules based on their properties and context to
modify their gradient. The most common composites for LRP are implemented in
:py:mod:`zennit.composites`.

The following computes LRP relevance using the ``EpsilonPlusFlat`` composite:

.. code-block:: python

    from zennit.composites import EpsilonPlusFlat


    # create a composite instance
    composite = EpsilonPlusFlat()

    # make sure the input requires a gradient
    input.requires_grad = True

    # compute the output and gradient within the composite's context
    with composite.context(model) as modified_model:
        output = modified_model(input)
        # gradient/ relevance wrt. class/output 0
        output.backward(gradient=torch.eye(10)[[0]])
        # relevance is not accumulated in .grad if using torch.autograd.grad
        # relevance, = torch.autograd.grad(output, input, torch.eye(10)[[0])

    # gradient is accumulated in input.grad
    print('Backward:', input.grad)


The context created by :py:func:`zennit.core.Composite.context` registers the
composite, which means that all rules are applied according to the composite's
mapping. See :doc:`/how-to/use-rules-composites-and-canonizers` for information on
using composites, :py:mod:`zennit.composites` for an API reference and
:doc:`/how-to/write-custom-compositors` for writing new compositors. Available
``Rules`` can be found in :py:mod:`zennit.rules`, their use is described in
:doc:`/how-to/use-rules-composites-and-canonizers` and how to add new ones is described in
:doc:`/how-to/write-custom-rules`.

Attributors
^^^^^^^^^^^

Alternatively, *attributors* may be used instead of ``composite.context``.

.. code-block:: python

   from zennit.attribution import Gradient


   attributor = Gradient(model, composite)

   with attributor:
        # gradient/ relevance wrt. output/class 1
        output, relevance = attributor(input, torch.eye(10)[[1]])

   print('EpsilonPlusFlat:', relevance)

Attribution methods which are not propagation-based, like
:py:class:`zennit.attribution.SmoothGrad` are implemented as attributors, and
may be combined with propagation-based (composite) approaches.

.. code-block:: python

   from zennit.attribution import SmoothGrad


   # we do not need a composite to compute vanilla SmoothGrad
   with SmoothGrad(model, noise_level=0.1, n_iter=10) as attributor:
        # gradient/ relevance wrt. output/class 7
        output, relevance = attributor(input, torch.eye(10)[[7]])

    print('SmoothGrad:', relevance)

More information on attributors can be found in :doc:`/how-to/use-attributors`
and :doc:`/how-to/write-custom-attributors`.

Canonizers
^^^^^^^^^^

For some modules and operations, Layerwise Relevance Propagation (LRP) is not
implementation-invariant, eg. ``BatchNorm -> Dense -> ReLU`` will be attributed
differently than ``Dense -> BatchNorm -> ReLU``. Therefore, LRP needs a
canonical form of the model, which is implemented in ``Canonizers``. These may
be simply supplied when instantiating a composite:

.. code-block:: python

   from torchvision.models import vgg16
   from zennit.composites import EpsilonGammaBox
   from zennit.torchvision import VGGCanonizer


   # instatiate the model
   model = vgg16()
   # create the canonizers
   canonizers = [VGGCanonizer()]
   # EpsilonGammaBox needs keyword arguments 'low' and 'high'
   high = torch.full_like(input, 4)
   composite = EpsilonGammaBox(low=-high, high=high, canonizers=canonizers)

   with Gradient(model, composite) as attributor:
        # gradient/ relevance wrt. output/class 0
        # torchvision.vgg16 has 1000 output classes by default
        output, relevance = attributor(input, torch.eye(1000)[[0]])

   print('EpsilonGammaBox:', relevance)

Some pre-defined canonizers for models from ``torchvision`` can be found in
:py:mod:`zennit.torchvision`. The :py:class:`zennit.torchvision.VGGCanonizer`
specifically is simply :py:class:`zennit.canonizers.SequentialMergeBatchNorm`,
which may be used when ``BatchNorm`` is used in sequential models. Note that for
``SequentialMergeBatchNorm`` to work, all functions (linear layers, activations,
...) must be modules and assigned to their parent module in the order they are
visited (see :py:class:`zennit.canonizers.SequentialMergeBatchNorm`). For more
information on canonizers see :doc:`/how-to/use-rules-composites-and-canonizers` and
:doc:`/how-to/write-custom-canonizers`.


Visualizing Results
^^^^^^^^^^^^^^^^^^^

While attribution approaches are not limited to the domain of images, they are
predominantly used on image models and produce heat maps of relevance. For
this reason, Zennit implements methods to visualize relevance heat maps.

.. code-block:: python

   from zennit.image import imsave


   # sum over the color channels
   heatmap = relevance.sum(1)
   # get the absolute maximum, to center the heat map around 0
   amax = heatmap.abs().numpy().max((1, 2))

   # save heat map with color map 'coldnhot'
   imsave(
       'heatmap.png',
       heatmap[0],
       vmin=-amax,
       vmax=amax,
       cmap='coldnhot',
       level=1.0,
       grid=False
   )

Information on ``imsave`` can be found at :py:func:`zennit.image.imsave`.
Saving an image with 3 color channels will result in the image being saved
without a color map but with the channels assumed as RGB. The keyword argument
``grid`` will create a grid of multiple images over the batch dimension if
``True``. Custom color maps may be created with
:py:class:`zennit.cmap.ColorMap`, eg. to save the previous image with a color
map ranging from blue to yellow to red:

.. code-block:: python

   from zennit.cmap import ColorMap


   # 00f is blue, ff0 is yellow, f00 is red, 0x80 is the center of the range
   cmap = ColorMap('00f,80:ff0,f00')

   imsave(
       'heatmap.png',
       heatmap,
       vmin=-amax,
       vmax=amax,
       cmap=cmap,
       level=1.0,
       grid=True
   )

More details to visualize heat maps and color maps can be found in
:doc:`/how-to/visualize-results`. The ColorMap specification language is
described in :py:class:`zennit.cmap.ColorMap` and built-in color maps are
implemented in :py:obj:`zennit.image.CMAPS`.

Example Script
--------------

A ready-to use example to analyze a few ImageNet models provided by torchvision
can be found at :repo:`share/example/feed_forward.py`.

The following setup requires bash, cURL and (magic-)file.

Create a virtual environment, install Zennit and download the example scripts:

.. code-block:: console

   $ mkdir zennit-example
   $ cd zennit-example
   $ python -m venv .venv
   $ .venv/bin/pip install zennit
   $ curl -o feed_forward.py \
       'https://raw.githubusercontent.com/chr5tphr/zennit/master/share/example/feed_forward.py'
   $ curl -o download-lighthouses.sh \
       'https://raw.githubusercontent.com/chr5tphr/zennit/master/share/scripts/download-lighthouses.sh'

Prepare the data required for the example:

.. code-block:: console

   $ mkdir params data results
   $ bash download-lighthouses.sh --output data/lighthouses
   $ curl -o params/vgg16-397923af.pth 'https://download.pytorch.org/models/vgg16-397923af.pth'

This creates the needed directories and downloads the pre-trained vgg16
parameters and 8 images of light houses from wikimedia commons into the
required label-directory structure for the imagenet dataset in PyTorch.

The ``feed_forward.py`` example can then be run using:

.. code-block:: console

   $ .venv/bin/python feed_forward.py \
       data/lighthouses \
       'results/vgg16_epsilon_gamma_box_{sample:02d}.png' \
       --inputs 'results/vgg16_input_{sample:02d}.png' \
       --parameters params/vgg16-397923af.pth \
       --model vgg16 \
       --composite epsilon_gamma_box \
       --relevance-norm symmetric \
       --cmap coldnhot

which computes the lrp heatmaps according to the ``epsilon_gamma_box`` rule and
stores them in results, along with the respective input images. Other possible
composites that can be passed to ``--composites`` are, e.g., ``epsilon_plus``,
``epsilon_alpha2_beta1_flat``, ``guided_backprop``, ``excitation_backprop``.


..
    The resulting heatmaps may look like the following:

    .. image:: /img/beacon_vgg16_epsilon_gamma_box.png
       :alt: Lighthouses with Attributions

Alternatively, heatmaps for SmoothGrad with absolute relevances may be computed
by omitting ``--composite`` and supplying ``--attributor``:

.. code-block:: console

   $ .venv/bin/python feed_forward.py \
        data/lighthouses \
        'results/vgg16_smoothgrad_{sample:02d}.png' \
        --inputs 'results/vgg16_input_{sample:02d}.png' \
        --parameters params/vgg16-397923af.pth \
        --model vgg16 \
        --attributor smoothgrad \
        --relevance-norm absolute \
        --cmap hot

For Integrated Gradients, ``--attributor integrads`` may be provided.

Heatmaps for Occlusion Analysis with unaligned relevances may be computed by
executing:

.. code-block:: console

   $ .venv/bin/python feed_forward.py \
        data/lighthouses \
        'results/vgg16_occlusion_{sample:02d}.png' \
        --inputs 'results/vgg16_input_{sample:02d}.png' \
        --parameters params/vgg16-397923af.pth \
        --model vgg16 \
        --attributor occlusion \
        --relevance-norm unaligned \
        --cmap hot

