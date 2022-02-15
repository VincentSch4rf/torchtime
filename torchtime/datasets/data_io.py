from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from typing import List
from urllib.request import urlretrieve

import pandas as pd

__all__ = [
    "load_UCR_UEA_dataset"
]

__author__ = [
    "VincentSch4rf"
]

from torchtime.io.arff import load_from_arff_to_dataframe
from torchtime.io.ts import load_from_tsfile_to_dataframe

DIRNAME = "data"
MODULE = os.path.dirname(__file__)


# time series classification data sets
def _download_and_extract(url, extract_path=None):
    """
    Download and unzip datasets (helper function).

    This code was modified from
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L28

    Parameters
    ----------
    url : string
        Url pointing to file to download
    extract_path : string, optional (default: None)
        path to extract downloaded zip to, None defaults
        to sktime/datasets/data

    Returns
    -------
    extract_path : string or None
        if successful, string containing the path of the extracted file, None
        if it wasn't succesful

    """
    file_name = os.path.basename(url)
    dl_dir = tempfile.mkdtemp()
    zip_file_name = os.path.join(dl_dir, file_name)
    urlretrieve(url, zip_file_name)

    if extract_path is None:
        extract_path = os.path.join(MODULE, "data/%s/" % file_name.split(".")[0])
    else:
        extract_path = os.path.join(extract_path, "%s/" % file_name.split(".")[0])

    try:
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        zipfile.ZipFile(zip_file_name, "r").extractall(extract_path)
        shutil.rmtree(dl_dir)
        _ensure_structure(extract_path)
        return extract_path
    except zipfile.BadZipFile as e:
        shutil.rmtree(dl_dir)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        raise ValueError(
            "Invalid dataset name. ",
            extract_path,
            "Please make sure the dataset "
            + "is available on http://timeseriesclassification.com/.",
        ) from e


def _ensure_structure(data_dir: str) -> None:
    """Remove nested data directory, which seems to exist for some UCR/UEA time series datasets

    :param data_dir: Path to the downloaded and extracted dataset directory
    :return: None
    """
    with os.scandir(data_dir) as dir:
        for entry in dir:
            if entry.is_dir() and entry.name == os.path.basename(os.path.dirname(data_dir)):
                with os.scandir(os.path.join(data_dir, entry.name)) as nested_dir:
                    for f in nested_dir:
                        shutil.move(f.path, data_dir)
                shutil.rmtree(os.path.join(data_dir, entry.name))


def _list_downloaded_datasets(extract_path):
    """Return a list of all the currently downloaded datasets.

    Modified version of
    https://github.com/tslearn-team/tslearn/blob
    /775daddb476b4ab02268a6751da417b8f0711140/tslearn/datasets.py#L250

    Returns
    -------
    datasets : List
        List of the names of datasets downloaded

    """
    if extract_path is None:
        data_dir = os.path.join(MODULE, DIRNAME)
    else:
        data_dir = extract_path
    datasets = [
        path
        for path in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, path))
    ]
    return datasets


def load_UCR_UEA_dataset(name, split=None, return_X_y=False, extract_path=None):
    """Load dataset from UCR UEA time series archive.

    Downloads and extracts dataset if not already downloaded. Data is assumed to be
    in the standard .ts format: each row is a (possibly multivariate) time series.
    Each dimension is separated by a colon, each value in a series is comma
    separated. ArrowHead is an example of a univariate equal length problem, BasicMotions
    an equal length multivariate problem.

    Parameters
    ----------
    name : str
        Name of data set. If a dataset that is listed in tsc_dataset_names is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from www.timeseriesclassification.com, saving it to
        the extract_path.
    split : None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    return_X_y : bool, optional (default=False)
        it returns two objects, if False, it appends the class labels to the dataframe.
    extract_path : str, optional (default=None)
        the path to look for the data. If no path is provided, the function
        looks in `sktime/datasets/data/`.

    Returns
    -------
    X: pandas DataFrame
        The time series data for the problem with n_cases rows and either
        n_dimensions or n_dimensions+1 columns. Columns 1 to n_dimensions are the
        series associated with each case. If return_X_y is False, column
        n_dimensions+1 contains the class labels/target variable.
    y: numpy array, optional
        The class labels for each case in X, returned separately if return_X_y is
        True, or appended to X if False
    """
    return _load_dataset(name, split, return_X_y, extract_path)


def _load_dataset(name: str, split: str, return_X_y, extract_path: str = None):
    """Load time series classification datasets (helper function)."""
    # Allow user to have non standard extract path
    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
        local_dirname = extract_path
    else:
        local_module = MODULE
        local_dirname = DIRNAME

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))
    if name not in _list_downloaded_datasets(extract_path):
        url = "http://timeseriesclassification.com/Downloads/%s.zip" % name
        # This also tests the validitiy of the URL, can't rely on the html
        # status code as it always returns 200
        _download_and_extract(
            url,
            extract_path=extract_path,
        )
    if isinstance(split, str):
        split = split.upper()

    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        abspath = os.path.join(local_module, local_dirname, name, fname)
        if os.path.exists(abspath):
            X, y = load_from_tsfile_to_dataframe(abspath)
        else:
            fname = name + "_" + split + ".arff"
            abspath = os.path.join(local_module, local_dirname, name, fname)
            X, y = load_from_arff_to_dataframe(abspath)
    # if split is None, load both train and test set
    elif split is None:
        X = pd.DataFrame(dtype="object")
        y = pd.Series(dtype="object")
        for split in ("TRAIN", "TEST"):
            fname = name + "_" + split + ".ts"
            abspath = os.path.join(local_module, local_dirname, name, fname)
            if os.path.exists(abspath):
                result = load_from_tsfile_to_dataframe(abspath)
            else:
                fname = name + "_" + split + ".arff"
                abspath = os.path.join(local_module, local_dirname, name, fname)
                result = load_from_arff_to_dataframe(abspath)
            X = pd.concat([X, pd.DataFrame(result[0])])
            y = pd.concat([y, pd.Series(result[1])])
    else:
        raise ValueError("Invalid `split` value =", split)

    # Return appropriately
    if return_X_y:
        return X, y
    else:
        X["class_val"] = pd.Series(y)
        return X