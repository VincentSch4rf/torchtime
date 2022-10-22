import csv
import os
from abc import abstractmethod
from typing import Tuple, Any, Callable, List, Optional, Union

import torch
from torch.utils.data import Dataset

from ..transforms import Compose


class TimeSeriesDataset(Dataset):
    """
    Base Class For making datasets which are compatible with torchtime.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.

    Args:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    .. note::

        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """
    _repr_indent = 4

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def classes(self):
        pass

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        torch._C._log_api_usage_once(f"torchtime.datasets.{self.__class__.__name__}")
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument!")

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (series, target) where target is index of the target class.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""


class TsvDataset(TimeSeriesDataset):
    """Create a Dataset for `.tsv` data.

    Args:
        root (str or Path): Path to the directory where the dataset is located.
             (Where the ``tsv`` file is present.)
        tsv (str, optional):
            The name of the tsv file used to construct the metadata, such as
            ``"train.tsv"``, ``"test.tsv"``, ``"dev.tsv"``, ``"invalidated.tsv"``,
            ``"validated.tsv"`` and ``"other.tsv"``. (default: ``"train.tsv"``)
    """

    def __init__(self,
                 root: str,
                 tsv: str = "train.tsv",
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
                 ) -> None:
        super(TsvDataset, self).__init__(root, transforms, transform, target_transform)

        self._tsv = os.path.join(self.root, tsv)

        with open(self._tsv, "r") as tsv_:
            walker = csv.reader(tsv_, delimiter="\t")
            self._header = next(walker)
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, Dict[str, str]): ``(waveform, sample_rate, dictionary)``,  where dictionary
            is built from the TSV file with the following keys: ``client_id``, ``path``, ``sentence``,
            ``up_votes``, ``down_votes``, ``age``, ``gender`` and ``accent``.
        """
        line = self._walker[n]
        sample = torch.as_tensor(line[:-1])
        target = line[-1]
        return sample, target

    def __len__(self) -> int:
        return len(self._walker)


class StandardTransform(object):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, data: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def _add_transform(self, transform: Callable, target: bool = False):
        transform_ = self.target_transform if target else self.transform
        if transform_ is not None:
            if isinstance(transform_, Compose):
                transform_.add_transform(transform)
            else:
                if isinstance(transform, list):
                    transform_ = Compose([transform_] + transform)
                else:
                    transform_ = Compose([transform_, transform])
        else:
            transform_ = transform

        if target:
            self.target_transform = transform_
        else:
            self.transform = transform_

    def add_transform(self, transform: Optional[Union[Callable, List[Callable]]] = None,
                      target_transform: Optional[Union[Callable, List[Callable]]] = None):
        if transform is not None:
            self._add_transform(transform, target=False)

        if target_transform is not None:
            self._add_transform(target_transform, target=True)

    @staticmethod
    def _format_transform_repr(transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)
