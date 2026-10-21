from skbase.utils.dependencies import _check_python_version

from pycaret.utils._show_versions import show_versions

version_ = "3.5.0"

__version__ = version_

__all__ = ["show_versions", "__version__"]


class _Pycaret:
    @classmethod
    def get_class_tag(cls, tag, tag_value_default=None):
        if tag == "python_version":
            return ">=3.10,<3.15"
        return tag_value_default


msg = "Pycaret only supports python 3.10 to 3.14."

_check_python_version(_Pycaret, msg=msg)
