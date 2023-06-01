.. py:module:: torchtime.datasets


torchtime.datasets
==================

Torchtime provides many built-in datasets in the ``torchtime.datasets``
module, as well as utility classes for building your own datasets.

Built-in datasets
-----------------

All datasets are subclasses of :class:`torch.utils.data.Dataset`
i.e, they have ``__getitem__`` and ``__len__`` methods implemented.
Hence, they can all be passed to a :class:`torch.utils.data.DataLoader`
which can load multiple samples in parallel using ``torch.multiprocessing`` workers.
For example: ::

    ucr_data = torchtime.datasets.UCR('path/to/ucr_root/', name='AbnormalHeartbeat')
    data_loader = torch.utils.data.DataLoader(ucr_data,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=args.nThreads)

.. currentmodule:: torchtime.datasets

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.
You can also create your own datasets using the provided :ref:`base classes <base_classes_datasets>`.

Time Series Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: dataset_class.rst

   UCR

.. _base_classes_datasets:

Base classes for custom datasets
--------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: dataset_class.rst

   TimeSeriesDataset
   PandasDataset

.. currentmodule:: torchtime.datasets
