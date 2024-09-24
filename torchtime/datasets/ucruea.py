import os
import shutil
import warnings
import zipfile
from typing import Optional, Callable, Tuple, Any, List
from urllib.error import URLError

import torch
from torch import Tensor

from . import TimeSeriesDataset
from .utils import download_and_extract_archive
from ..exceptions import DataConversionWarning
from ..io.arff import load_from_arff_to_dataframe
from ..io.ts import TSFileLoader
from ..transforms import functional as F
from ..utils import stack_pad


class UCR(TimeSeriesDataset):
    """*UEA & UCR Time Series Classification Repository* :cite:`Dau2019UCR`.

    Args:
        root (str): Root directory of dataset where ``UCR/<dataset_name>`` exist.
        name (str): The name of the dataset to load from the UCR archive.
        train (bool, optional): If True, creates dataset from ``TRAIN.ts`` or ``TRAIN.arff``, otherwise from
            ``TEST.ts`` or ``TEST.arff``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        classes (list, optional): A list containing the class labels, where the index in the list indicates the numeric
            value it should be mapped to.
        transform (callable, optional): A function/transform that  takes in a uni- or multivariate time series and
            returns a transformed version. E.g, ``transforms.NaN2Value``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _repr_indent = 4

    mirrors = [
        'http://timeseriesclassification.com/aeon-toolkit/'
    ]

    univariate = [
        'AbnormalHeartbeat', 'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
        'AllGestureWiimoteZ', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken',
        'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration',
        'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
        'CricketZ', 'Crop', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame',
        'DodgerLoopWeekend', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
        'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal',
        'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
        'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain',
        'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
        'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham',
        'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
        'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
        'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
        'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
        'MoteStrain', 'NonInvasiveFetalECGThorax1',
        'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
        'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ',
        'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane',
        'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
        'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
        'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
        'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
        'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
        'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
        'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
        'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
        'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
    ]

    multivariate = [
        'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
        'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
        'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
        'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
        'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
        'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports',
        'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits',
        'StandWalkJump', 'UWaveGestureLibrary'
    ]

    @property
    def data_dir(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, self.name)

    @property
    def training_file(self) -> str:
        return self.name + "_TRAIN"

    @property
    def test_file(self) -> str:
        return self.name + "_TEST"

    archive_format = ".zip"

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    def __init__(
            self,
            name: str,
            root: str,
            train: bool = True,
            download: bool = False,
            classes: Optional[List[Any]] = None,
            **kwargs
    ) -> None:

        super(UCR, self).__init__(root, **kwargs)

        self.name = name
        self.root = root
        self.train = train  # training set or test set
        if classes is not None and len(set(classes)) != len(classes):
            raise ValueError("Class mapping contains duplicate labels!")
        self.classes = classes

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets, self.classes = self._load_data()

    def _load_data(self) -> Tuple[Tensor, Tensor, List[Any]]:
        file_name = (self.training_file if self.train else self.test_file) + ".ts"
        abspath = os.path.join(self.data_dir, file_name)
        if os.path.exists(abspath):
            file_loader = TSFileLoader(abspath)
            data, targets = file_loader.as_tensor(return_targets=True)
            classes = file_loader.get_classes()
            self.dim = (file_loader.dim, file_loader.series_length)
        else:
            file_name = (self.training_file if self.train else self.test_file) + ".arff"
            abspath = os.path.join(self.data_dir, file_name)
            data, targets, classes = load_from_arff_to_dataframe(abspath, return_class_labels=True)
            data_ = []
            for i in range(data.shape[-1]):
                # to_time_series_dataset(data[f'dim_{i}'])
                data_.append(stack_pad(data[f'dim_{i}']))  # stack arrays even if they have different lengths
            data = torch.permute(torch.stack(data_, dim=-1), (0, 2, 1))
            self.dim = data.size()[1:]
            # data = match_seq_len(data)

        if self.classes is not None and classes is not None:
            if len(self.classes) != len(classes):
                raise ValueError("Number of classes present in {} is different from the provided class mapping."
                                 .format(abspath))
            if self.classes != classes:
                warnings.warn("The classes provided do not match the classes from the dataset!")

        try:
            # targets = list(map(int, targets))
            targets = torch.as_tensor(targets)
            if classes is None:
                classes = torch.unique(targets).tolist()
        except (ValueError, TypeError):
            if self.classes is not None:
                _, targets = F.encode_labels(targets, self.classes)
            else:
                if classes is not None:
                    warnings.warn("Labels are not numeric and no explicit class mapping is provided. "
                                  "Please pass an explicit class mapping to the constructor. "
                                  "Using class labels parsed from dataset.",
                                  DataConversionWarning,
                                  stacklevel=2)
                    _, targets = F.encode_labels(targets, classes)
                else:
                    warnings.warn("Dataset did not include explicit class labels. Inferring class labels from targets.",
                                  DataConversionWarning,
                                  stacklevel=2)
                    classes, targets = F.encode_labels(targets)
                    self.classes = classes
        return data, targets, classes

    def _ensure_structure(self) -> None:
        """Remove nested data directory, which seems to exist for some UCR/UEA time series datasets"""

        with os.scandir(self.data_dir) as d:
            for entry in d:
                if entry.is_dir() and entry.name == self.name:
                    with os.scandir(os.path.join(self.data_dir, entry.name)) as nested_dir:
                        for f in nested_dir:
                            shutil.move(f.path, self.data_dir)
                    shutil.rmtree(os.path.join(self.data_dir, entry.name))

    def _check_exists(self) -> bool:
        # TODO: maybe get rid of multiple archive extractions and check ts/arff file checksum instead
        return os.path.exists(self.data_dir) and os.path.isdir(self.data_dir)
        # contents = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]

    def download(self) -> None:
        """Download the UCR/UEA data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        # download files
        filename = self.name + self.archive_format
        for mirror in self.mirrors:
            url = "{}{}".format(mirror, filename)
            try:
                print("Downloading {}".format(url))
                download_and_extract_archive(
                    url, download_root=self.data_dir,
                    filename=filename,
                    md5=None
                )
                self._ensure_structure()
            except URLError as error:
                print(
                    "Failed to download (trying next):\n{}".format(error)
                )
                continue
            except zipfile.BadZipFile as e:
                os.remove(os.path.join(self.data_dir, filename))
                if os.path.exists(self.data_dir):
                    shutil.rmtree(self.data_dir)
                raise ValueError(
                    "Invalid dataset name! ",
                    self.name,
                    "Please make sure the dataset "
                    + "is available on http://timeseriesclassification.com/.",
                ) from e
            finally:
                print()
            break

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): The index of the sample to return.

        Returns:
            tuple: (series, target) where target is index of the target class.
        """
        series, target = self.data[index], self.targets[index]

        if self.transforms is not None:
            series, target = self.transforms(series, target)

        return series, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Dataset: {}\n".format(self.name) + \
               "Split: {}".format("Train" if self.train is True else "Test")
