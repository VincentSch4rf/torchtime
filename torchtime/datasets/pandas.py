import os.path
from typing import Optional, Callable, Tuple, Any, List, Union

import numpy as np
import pandas as pd
import torch

from .timeseries import TimeSeriesDataset


class PandasDataset(TimeSeriesDataset):

    @property
    def dim(self):
        if self.__has_dimensions:
            return len(self.dimensions)
        return self.df.shape[1]

    @property
    def classes(self):
        if self.__has_labels:
            return self.df[self.labels].unique()
        return None

    def __init__(self,
                 path: Optional[str] = None,
                 dataframe: Optional[pd.DataFrame] = None,
                 dimensions: Optional[List[str]] = None,
                 labels: Optional[str] = None,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        if path is not None:
            if dataframe is not None:
                raise ValueError(
                    "path and dataframe are exclusive and must not be given at the same time"
                )
            self.path = path
            self.name = os.path.basename(path)
            self.df = pd.read_pickle(self.path)
        else:
            self.name = "In Memory Dataframe"
            self.df = dataframe
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

        super(PandasDataset, self).__init__(self.name, transforms, transform, target_transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: Union[int, slice]) -> Tuple[Any, Any]:
        if self.__has_dimensions:
            columns = self.dimensions
        else:
            columns = self.df.columns
        rows = self.df.iloc[index]
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
