import sys

from pycaret.utils._show_versions import show_versions

version_ = "3.4.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]

# Pycaret only supports Python 3.9, 3.10, 3.11, 3.12, 3.13.
# This code is to avoid issues with Python 3.7 or other unsupported versions.
# Example (see package versions): https://github.com/pycaret/pycaret/issues/3746.

if sys.version_info < (3, 9) or sys.version_info >= (3, 14):
    raise RuntimeError(f"Pycaret only supports Python 3.9, 3.10, 3.11, 3.12, 3.13. "
                       f"Your actual Python version is {sys.version_info}.")
