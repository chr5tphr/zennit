====================
Zennit Documentation
====================

Zennit (Zennit Explains Neural Networks in Torch) is a python framework using PyTorch to compute local attributions in the sense of eXplainable AI (XAI) with a focus on Layerwise Relevance Progagation.
It works by defining *rules* which are used to overwrite the gradient of PyTorch modules in PyTorch's auto-differentiation engine.
Rules are mapped to layers with *composites*, which contain directions to compute the attributions of a full model, which maps rules to modules based on their properties and context.

Zennit is available on PyPI and may be installed using:

.. code-block:: console

   $ pip install zennit

Contents
--------

.. toctree::
    :maxdepth: 2

    getting-started
    how-to/index
    tutorial/index
    reference/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Citing
------

If you find Zennit useful, why not cite our related paper:

.. code-block:: bibtex

    @article{anders2021software,
          author  = {Anders, Christopher J. and
                     Neumann, David and
                     Samek, Wojciech and
                     Müller, Klaus-Robert and
                     Lapuschkin, Sebastian},
          title   = {Software for Dataset-wide XAI: From Local Explanations to Global Insights with {Zennit}, {CoRelAy}, and {ViRelAy}},
          journal = {CoRR},
          volume  = {abs/2106.13200},
          year    = {2021},
    }

