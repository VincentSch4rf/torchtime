import os.path
from typing import Optional, Callable, Tuple, Any, List, Union

import numpy as np
import pandas as pd

from .timeseries import TimeSeriesDataset


class PandasDataset(TimeSeriesDataset):
    """
    Base class for creating datasets which are compatible with torchtime from ``pandas.Dataframe``.
    
    Args:
        path (str): The path to the pickle file containing the ``pandas.Dataframe`` to load.
        dataframe (pd.Dataframe): The ``pandas.Dataframe`` to load.
        dimensions (list, optional): The columns of the ``pandas.Dataframe`` which contain the individual dimensions of
            contained time series. If ``None`` is given, all columns are considered to hold the timer series data.
        labels (str, optional): The column of the `pandas.Dataframe` which contains the labels. If ``None`` is given,
            the data is assumed to have no labels.

    .. note::

        :attr:`path` and :attr:`dataframe` are mutually exclusive.
    """

    @property
    def dim(self):
        if self.__has_dimensions:
            return len(self.dimensions)
        return self.data.shape[1]

    @property
    def classes(self):
        if self.__has_labels:
            return self.data[self.labels].unique()
        return None

    def __init__(self,
                 path: Optional[str] = None,
                 dataframe: Optional[pd.DataFrame] = None,
                 dimensions: Optional[List[str]] = None,
                 labels: Optional[str] = None,
                 **kwargs
                 ):
        if path is not None:
            if dataframe is not None:
                raise ValueError(
                    "path and dataframe are exclusive and must not be given at the same time"
                )
            self.path = path
            self.name = os.path.basename(path)
            self.data = pd.read_pickle(self.path)
        else:
            self.name = "In Memory Dataframe"
            self.data = dataframe
        self.__has_labels = labels is not None
        if self.__has_labels:
            if not isinstance(labels, str):
                raise TypeError(
                    "The columns containing the labels needs to be provided as a string"
                )
        self.labels = labels
        self.__has_dimensions = dimensions is not None
        if self.__has_dimensions:
            if not isinstance(dimensions, list):
                raise TypeError(
                    "The columns containing the time series dimensions needs to be provided as a list of strings"
                    "containing valid column identifier"
                )
        self.dimensions = dimensions

        super(PandasDataset, self).__init__(self.name, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: Union[int, slice]) -> Tuple[Any, Any]:
        """
        Args:
            index: The index of the sample to return.

        Returns:
            tuple: If :attr:`__has_labels` is ``True``, returns (series, target), where target is the index of the
            target class. Else, returns (series, ``torch.empty``), where the empty tensor has the required shape.
        """
        if self.__has_dimensions:
            columns = self.dimensions
        else:
            columns = self.data.columns
        rows = self.data.iloc[index]
        if isinstance(index, slice):
            data = np.swapaxes(np.stack([
                np.stack(rows[columns].iloc[:, i]) for i in range(len(columns))
            ]), 0, 1).astype(np.float32)
            targets = rows[self.labels].to_numpy()
        elif isinstance(index, int):
            data = np.stack(rows[columns]).astype(np.float32)
            targets = rows[self.labels]
        else:
            raise ValueError("index must be int or slice of ints, got {}".format(type(index)))
        if self.transforms is not None:
            return self.transforms(data, targets)
        else:
            return data, targets
