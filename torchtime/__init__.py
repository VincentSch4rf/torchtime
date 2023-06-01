import importlib.metadata
try:
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from torchtime import utils
from torchtime import exceptions
from torchtime import io
from torchtime import datasets
from torchtime import transforms
from torchtime import models
