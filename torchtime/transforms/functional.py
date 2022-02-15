import warnings
from typing import Any, List, Sequence, Tuple, Optional, Union, Set

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from ..exceptions import DataConversionWarning
from ..utils import _check_unknown


@torch.jit.unused
def _is_numpy(ts: Any) -> bool:
    return isinstance(ts, np.ndarray)


@torch.jit.unused
def _is_numpy_timeseries(ts: Any) -> bool:
    return _is_numpy(ts) and ts.ndim in {1, 2}


def pad(series: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    if not isinstance(padding, (tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2]:
        raise ValueError("Padding must be an int or a 1 or 2 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if padding_mode not in ["constant", "replicate", "reflect"]:
        raise ValueError("Padding mode should be either constant, replicate or reflect")

    out_dtype = series.dtype
    need_cast = False
    if (padding_mode != "constant") and series.dtype not in (torch.float32, torch.float64):
        # Temporary cast input tensor to float until pytorch issue is resolved :
        # https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        series = series.to(torch.float32)

    series = F.pad(series, padding, mode=padding_mode, value=float(fill))

    if need_cast:
        series = series.to(out_dtype)

    return series


def normalize(tensor: Tensor, mean: Sequence[float], std: Sequence[float], inplace: bool = False) -> Tensor:
    """Normalize a float tensor time series with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.

    See :class:`~torchtime.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor time series of size (C, L) or (B, C, L) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor time series.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 2:
        raise ValueError('Expected tensor to be a tensor time series of size (..., C, L). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def column_or_1d(y, *, warn=False) -> np.ndarray:
    """Ravel column or 1d numpy array, else raises an error.
    Parameters
    ----------
    y : array-like
    warn : bool, default=False
       To control display of warnings.
    Returns
    -------
    y : ndarray
    """
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples, ), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        return np.ravel(y)

    raise ValueError(
        "y should be a 1d array, got an array of shape {} instead.".format(shape)
    )


def encode_labels(targets: List[Any], classes: Optional[List[Any]] = None) -> Tuple[List[Any], Tensor]:
    if classes is None:
        classes = set(targets)
    diff = _check_unknown(targets, classes)
    if diff:
        raise ValueError(f"y contains previously unseen labels: {str(diff)}")
    table = {val: i for i, val in enumerate(classes)}
    return classes, torch.as_tensor([table[v] for v in targets])
    # r = np.searchsorted(classes, targets)
    # return classes, torch.as_tensor(r)
