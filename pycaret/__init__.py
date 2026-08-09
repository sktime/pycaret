from skbase.utils.dependencies import _check_python_version

from pycaret.utils._show_versions import show_versions

version_ = "3.4.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]

msg = "Pycaret only supports python 3.10 to 3.14."
        
_check_python_version(">=3.10,<3.14", msg=msg)
