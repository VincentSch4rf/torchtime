from typing import Any, Union, Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional
from torch import Tensor

from . import functional as F
from .functional import _is_numpy, _is_numpy_timeseries, column_or_1d


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Nan2Value(),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.Nan2Value(),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions, ``numpy`` or ``pandas``.

    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, ts: Any):
        for t in self.transforms:
            ts = t(ts)
        return ts

    def add_transform(self, transforms: Union[List[Callable], Callable]):
        if isinstance(transforms, list):
            self.transforms.extend(transforms)
        else:
            self.transforms.append(transforms)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor:
    """
    Convert a numpy time series to tensor and scale the values accordingly.

    This transform does not support torchscript.
    """
    def __call__(self, ts: np.ndarray):
        if not _is_numpy(ts):
            raise TypeError('ts should be a numpy array. Got {}'.format(type(ts)))

        if _is_numpy(ts) and not _is_numpy_timeseries(ts):
            raise ValueError('ts should be 1/2 dimensional. Got {} dimensions.'.format(ts.ndim))

        default_float_type = torch.get_default_dtype()

        if ts.ndim == 1:
            ts = ts[:, None]

        ts = torch.from_numpy(ts.transpose((0, 1))).contiguous()
        return ts

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Nan2Value:
    """Replace NaN values in a time series by some value or its median.

    Args:
        value (int, float): The value NaNs should be replaced with.
        median (bool): If True, NaN values are replaced by the median of time series.
        by_channel (bool): If True, NaN values are replaced with the channel-wise median.
    """
    def __init__(self, value: Union[int, float] = 0, median: bool = False, by_channel: bool = False):
        self.median = median
        self.value = value
        self.by_channel = by_channel

    def __call__(self, ts: torch.Tensor):
        """Replaces NaN values in the input time series with either some value provided by the user, or the median of
        the given time series.

        Args:
            ts (Tensor): A time series as a 1 or 2-dimensional Tensor.

        Returns:
            Tensor: The time series with the same dimensions, containing no NaN values.
        """
        mask = torch.isnan(ts)
        if mask.any():
            if self.median:
                if self.by_channel:
                    median = torch.nanmedian(ts, dim=1, keepdim=True)[0].repeat(1, ts.shape[-1])
                    ts[mask] = median[mask]
                else:
                    ts = torch.nan_to_num(ts, torch.nanmedian(ts).item())
            else:
                ts = torch.nan_to_num(ts, self.value)
        return ts

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Pad(torch.nn.Module):
    """Pads a time series to a given target length

    Args:
        series_length (int): the target length of the time series
        fill (float, int): fill value for `'constant'` padding. Default `0`.
        padding_mode (str): the padding mode that should be used. One of `'constant'`, `'reflect'` or `'replicate'`.
            Default `'constant'`.
    """

    def __init__(self, series_length: int, fill: Union[float, int] = 0, padding_mode="constant"):
        super(Pad, self).__init__()
        if not isinstance(series_length, int):
            raise TypeError("Got inappropriate padding arg")
        if not isinstance(fill, (float, int)):
            raise TypeError("Got inappropriate fill arg")
        if padding_mode not in ["constant", "reflect", "replicate"]:
            raise ValueError("Padding mode should be either constant, replicate or reflect")

        self.series_length = series_length
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, series: Tensor) -> Tensor:
        """
        Args:
            series (Tensor): Time series to be padded.

        Returns:
            Tensor: Padded time series.
        """
        padding = [0, self.series_length - series.size(dim=-1)]
        return F.pad(series, padding, fill=self.fill, padding_mode=self.padding_mode)

    def extra_repr(self) -> str:
        return 'series_length={}, mode={}, value={}'.format(self.series_length, self.padding_mode, self.value)


class Normalize(torch.nn.Module):
    """Normalize a tensor time series with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (tuple): Sequence of means for each channel.
        std (tuple): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean: Tuple[float], std: Tuple[float], inplace: bool = False):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor time series to be normalized.

        Returns:
            Tensor: Normalized Tensor series.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resample(torch.nn.Module):
    """Down/up samples the given time series to either the given :attr:`size` or by the given
    :attr:`scale_factor`.

    The algorithm used for interpolation is determined by :attr:`mode`.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x width`.

    The modes available for resizing are: `linear`, `bilinear`, `area`, `nearest-exact`.

    Args:
        sz (int):
            output temporal size.
        scale_factor (float): multiplier for temporal size.
        mode (str): algorithm used for upsampling:
            ``'linear'`` | ``'bilinear'`` | ``'nearest-exact'``. Default: ``'linear'``
    """

    def __init__(self, sz: Optional[int], scale_factor: Optional[float], mode: str = 'linear'):
        super(Resample, self).__init__()
        self.sz = sz
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, ts: Tensor) -> Tensor:
        """
        Args:
            ts (Tensor): Tensor time series to resample.

        Returns:
            Tensor: Resampled time series.
        """
        return torch.nn.functional.interpolate(ts, self.sz, self.scale_factor, self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '(size={},scale_factor={},mode={})'.format(
            self.sz, self.scale_factor, self.mode)


class LabelEncoder:
    """Transformer to encode labels into [0, n_uniques - 1]. Uses pure python method for object dtype, and numpy method
    for all other dtypes.
    This transform does not support torchscript.

    Args:
        targets (numpy ndarray): numpy array containing all possible labels. Since the uniques are computed intuitively
        this can simply be the targets of the training set.
    """
    def __init__(self, targets: npt.ArrayLike):
        targets = column_or_1d(targets)
        self.classes = np.unique(targets)
        self.table = {val: i for i, val in enumerate(self.classes)}

    def __call__(self, target: Any) -> int:
        """Encode a target using the defined set of label encodings.

        Args:
            target: Some label that should be encoded

        Returns:
            int: Encoded label.
        """
        if isinstance(target, np.ndarray):
            diff = [d for d in np.setdiff1d(np.unique(target), self.classes) if d not in self.classes]
            if diff:
                raise ValueError(f"y contains previously unseen labels: {str(diff)}")
            return np.searchsorted(self.classes, target)
        else:
            try:
                return self.table[target]
            except KeyError as e:
                raise ValueError(f"y contains previously unseen labels: {str(e)}")

    def __repr__(self) -> str:
        encoding_table = '\n\tEncoding Table: {\n'
        for k, v in self.table.items():
            encoding_table += '\t\t' + str(k) + ': ' + str(v) + '\n'
        encoding_table += '\t}\n'
        return self.__class__.__name__ + '(' + encoding_table + ')'
