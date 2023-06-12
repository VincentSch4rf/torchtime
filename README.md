torchtime: time series library for PyTorch
==========================================
[![Downloads](https://static.pepy.tech/badge/pytorchtime)](https://pypi.org/project/pytorchtime)
[![Documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorchtime%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorchtime.com/docs/stable)

![Torchtime Logo](docs/source/_static/img/logo.png)

--------------------------------------------------------------------------------

The aim of torchtime is to apply [PyTorch](https://github.com/pytorch/pytorch) to
the time series domain. By supporting PyTorch, torchtime follows the same philosophy
of providing strong GPU acceleration, having a focus on trainable features through
the autograd system, and having consistent style (tensor names and dimension names).
Therefore, it is primarily a machine learning library and not a general signal
processing library. The benefits of PyTorch can be seen in torchtime through
having all the computations be through PyTorch operations which makes it easy
to use and feel like a natural extension.

- [Support time series I/O (Load files, Save files)](http://pytorchtime.com/docs/stable/)
  - Load a variety of time series formats, such as `ts`, `arff`, `dvi`, `dxd`, into a torch Tensor
- [Dataloaders for common time series datasets](http://pytorchtime.com/docs/stable/datasets.html)
- Common time series transforms
    - [Nan2Value, Normalization, Padding, Resample](http://pytorchtime.com/docs/stable/transforms.html)

Installation
------------

Please refer to https://pytorchtime.com/docs/stable/installation.html for installation and build process of torchtime.

API Reference
-------------

API Reference is located here: http://pytorchtime.com/docs/stable/

Contributing Guidelines
-----------------------

Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md)

Citation
--------

If you find this package useful, please cite as:

```bibtex
@article{scharf2022torchtime,
  title={PyTorch Time: Bringing Deep Learning to Time Series Classification},
  author={Vincent Scharf and Paul Gerhard Ploeger},
  url={https://github.com/VincentSch4rf/torchtime},
  year={2022}
}
```

Disclaimer on Datasets
----------------------

This is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the ML community!