.. _io:

torchtime.io
============
.. py:module:: torchtime.io

The :mod:`torchtime.io` module provides implementations for time series I/O functionalities, such as, reading and
writing common file formats.

.. note::
   Currently, some data formats are yet only available through the functional interface. This is supposed to change in
   the near future.

.. currentmodule:: torchtime.io

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: class.rst

   TSFileLoader

Functional
----------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: function.rst

   load_from_arff_to_dataframe
