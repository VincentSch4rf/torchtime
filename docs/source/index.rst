Torchtime Documentation
========================

.. image:: _static/img/logo-dark.svg

Torchtime is a library for time series processing with PyTorch.
It provides I/O, signal and data processing functions, datasets,
model implementations and application components.

.. toctree::
   :maxdepth: 2
   :caption: Torchtime Documentation
   :hidden:

   references

.. toctree::
   :maxdepth: 2
   :caption: Installation
   :hidden:

   installation
   get-started

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/classification_tutorial

.. toctree::
   :maxdepth: 1
   :caption: API Tutorials
   :hidden:


.. toctree::
   :maxdepth: 1
   :caption: Training Recipes
   :hidden:


.. toctree::
   :maxdepth: 4
   :caption: Python API Reference
   :hidden:

   torchtime
   io
   datasets
   transforms
   models

.. toctree::
   :maxdepth: 1
   :caption: PyTorch Libraries
   :hidden:

   PyTorch <https://pytorch.org/docs>
   torchaudio <https://pytorch.org/audio>
   torchtext <https://pytorch.org/text>
   torchvision <https://pytorch.org/vision>
   TorchElastic <https://pytorch.org/elastic/>
   TorchServe <https://pytorch.org/serve>
   PyTorch on XLA Devices <http://pytorch.org/xla/>


Getting started
---------------
.. code-block:: bash

    pip install pytorchtime


Citing torchtime
----------------

If you find torchtime useful, please cite the following paper:

- Scharf, Vincent, Ploeger, Paul Gerhard (2022).
  PyTorch Time: Bringing Deep Learning to Time Series Classification.


In BibTeX format:

.. code-block:: bibtex

    @article{scharf2022torchtime,
      title={PyTorch Time: Bringing Deep Learning to Time Series Classification},
      author={Vincent Scharf and Paul Gerhard Ploeger},
      url={https://github.com/VincentSch4rf/torchtime},
      year={2022}
    }
