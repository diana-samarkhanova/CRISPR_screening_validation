"""CRISPR-EvidenceRank public package interface."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from .labels import LabelCode, model_target

try:
    __version__ = package_version("crispr-evidencerank")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["LabelCode", "__version__", "model_target"]
