import os
from typing import Optional, Callable, Any, Tuple, List, Union

import torch

from . import TimeSeriesDataset
from ..io import dwsoft

dwsoft.loadDLL()


class PatchedDEWESoftDataset(TimeSeriesDataset):

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

    series_dir_name = "series"
    patches_dir_name = "patches"
    labels_dir_name = "labels"

    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = False,
                 dimensions: Optional[List[str]] = None,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
                 ) -> None:
        super(PatchedDEWESoftDataset, self).__init__(root, transforms, transform, target_transform)

        self.train = train
        self._location = "train" if self.train else "test"
        self.series_files = []
        self.patch_files = []
        self.target_files = []

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        series_dir = os.path.join(self.root, self._location, self.series_dir_name)
        patches_dir = os.path.join(self.root, self._location, self.patches_dir_name)
        if self.train:
            labels_dir = os.path.join(self.root, self._location, self.labels_dir_name)
        for dxd_file in os.listdir(series_dir):
            self.series_files.append(os.path.join(series_dir, dxd_file))
            self.patch_files.append(os.path.join(patches_dir, f"{dxd_file.split('.')[0]}.txt"))
            if self.train:
                self.target_files.append(os.path.join(labels_dir, f"{dxd_file.split('.')[0]}.txt"))

        self.dimensions = dimensions
        self.__has_dimensions = dimensions is not None

        self.patches = self._build_patches()
        if self.train:
            self.targets = self._build_targets()

    def _build_patches(self) -> List[Tuple[int, int, int]]:
        patches = []
        for i, patch_file in enumerate(self.patch_files):
            with open(patch_file) as f:
                for i, line in enumerate(f.readlines()):
                    start_idx, stop_idx = map(int, line.split(","))
                    patches.append((i, start_idx, stop_idx))
        return patches

    def _build_targets(self) -> List[int]:
        targets = []
        for idx in range(len(self.target_files)):
            with open(self.target_files[idx]) as target_file:
                for i, line in enumerate(target_file.readlines()):
                    label = int(line)
                    targets.append(label)
        return targets

    def _check_exists(self):
        return True

    def __len__(self):
        return len(self.series)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, int):
            series_idx, start_idx, stop_idx = self.patches[index]
            dxd = dwsoft.open(self.series_files[series_idx])
            data = dxd.tensor(channels=self.dimensions)[:, start_idx:stop_idx]
        else:
            raise ValueError("index must be int, got {}".format(type(index)))
        targets = torch.tensor(self.targets[index])
        if self.transforms is not None:
            return self.transforms(data, targets)
        return data, targets
