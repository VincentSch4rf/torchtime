import math
import numbers
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch


def stack_pad(ts, padding_value=float('nan')):
    """Converts an iterable into a tensor array using padding if necessary

    Args:
        ts:
        padding_value:

    Returns:

    """
    row_length = len(max(ts, key=len))
    result = torch.full((len(ts), row_length), padding_value)
    for i, row in enumerate(ts):
        result[i, :len(row)] = torch.as_tensor(row)
    return result


def match_seq_len(*arrays: np.ndarray):
    """

    Args:
        *arrays:

    Returns:

    """
    max_len = np.stack([x.shape[-1] for x in arrays]).max()
    return [np.pad(x, pad_width=((0, 0), (0, 0), (max_len - x.shape[-1], 0)), mode='constant', constant_values=0) for x
            in arrays]


def is_scalar_nan(x):
    """Tests if x is NaN.
    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not float('nan').
    Parameters
    ----------
    x : any type
    Returns
    -------
    boolean
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils import is_scalar_nan
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    return isinstance(x, numbers.Real) and math.isnan(x)


class MissingValues(NamedTuple):
    """Data class for missing data information"""

    nan: bool
    none: bool

    def to_list(self):
        """Convert tuple to a list where None is always first."""
        output = []
        if self.none:
            output.append(None)
        if self.nan:
            output.append(np.nan)
        return output


def _extract_missing(values):
    """Extract missing values from `values`.
    Parameters
    ----------
    values: set
        Set of values to extract missing from.
    Returns
    -------
    output: set
        Set with missing values extracted.
    missing_values: MissingValues
        Object with missing value information.
    """
    missing_values_set = {
        value for value in values if value is None or is_scalar_nan(value)
    }

    if not missing_values_set:
        return values, MissingValues(nan=False, none=False)

    if None in missing_values_set:
        if len(missing_values_set) == 1:
            output_missing_values = MissingValues(nan=False, none=True)
        else:
            # If there is more than one missing value, then it has to be
            # float('nan') or np.nan
            output_missing_values = MissingValues(nan=True, none=True)
    else:
        output_missing_values = MissingValues(nan=True, none=False)

    # create set without the missing values
    output = values - missing_values_set
    return output, output_missing_values


def _check_unknown(values: npt.ArrayLike, known_values: npt.ArrayLike, return_mask=False):
    """
    Helper function to check for unknowns in values to be encoded.
    Uses pure python method for object dtype, and numpy method for
    all other dtypes.
    Parameters
    ----------
    values : array
        Values to check for unknowns.
    known_values : array
        Known values. Must be unique.
    return_mask : bool, default=False
        If True, return a mask of the same shape as `values` indicating
        the valid values.
    Returns
    -------
    diff : list
        The unique values present in `values` and not in `know_values`.
    valid_mask : boolean array
        Additionally returned if ``return_mask=True``.
    """
    valid_mask = None
    values = np.asarray(values)

    if values.dtype.kind in "OUS":
        values_set = set(values)
        values_set, missing_in_values = _extract_missing(values_set)

        uniques_set = set(known_values)
        uniques_set, missing_in_uniques = _extract_missing(uniques_set)
        diff = values_set - uniques_set

        nan_in_diff = missing_in_values.nan and not missing_in_uniques.nan
        none_in_diff = missing_in_values.none and not missing_in_uniques.none

        def is_valid(value):
            return (
                value in uniques_set
                or missing_in_uniques.none
                and value is None
                or missing_in_uniques.nan
                and is_scalar_nan(value)
            )

        if return_mask:
            if diff or nan_in_diff or none_in_diff:
                valid_mask = np.array([is_valid(value) for value in values])
            else:
                valid_mask = np.ones(len(values), dtype=bool)

        diff = list(diff)
        if none_in_diff:
            diff.append(None)
        if nan_in_diff:
            diff.append(np.nan)
    else:
        unique_values = np.unique(values)
        diff = np.setdiff1d(unique_values, known_values, assume_unique=True)
        if return_mask:
            if diff.size:
                valid_mask = np.in1d(values, known_values)
            else:
                valid_mask = np.ones(len(values), dtype=bool)

        # check for nans in the known_values
        if np.isnan(known_values).any():
            diff_is_nan = np.isnan(diff)
            if diff_is_nan.any():
                # removes nan from valid_mask
                if diff.size and return_mask:
                    is_nan = np.isnan(values)
                    valid_mask[is_nan] = 1

                # remove nan from diff
                diff = diff[~diff_is_nan]
        diff = list(diff)

    if return_mask:
        return diff, valid_mask
    return diff
