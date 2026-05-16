"""scAPAview: visualizing alternative polyadenylation from single-cell data."""

__version__ = "0.1.0"
__author__ = "Martín E. García Solá"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("scapaview")
except PackageNotFoundError:
    pass
