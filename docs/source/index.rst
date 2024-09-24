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

- Scharf, V. (2022). torchtime (0.1.2). Zenodo. https://doi.org/10.5281/zenodo.13832395


In BibTeX format:

.. code-block:: bibtex

    @software{scharf2024torchtime,
      author       = {Vincent Scharf},
      title        = {torchtime},
      year         = 2022,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.13832394},
      url          = {https://github.com/VincentSch4rf/torchtime}
    }
