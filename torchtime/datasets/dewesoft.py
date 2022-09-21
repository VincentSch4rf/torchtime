
from typing import Optional, Callable

from . import TimeSeriesDataset


class DEWESoftDataset(TimeSeriesDataset):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
                 ) -> None:
        super(DEWESoftDataset, self).__init__(root, transforms, transform, target_transform)
