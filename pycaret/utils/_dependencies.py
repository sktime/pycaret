# This module now only exports compatibility helpers for version checking.
# The main _check_soft_dependencies has been migrated to use scikit-base directly.

from skbase.utils.dependencies._dependencies import _get_installed_packages


def get_module_version_str(modname: str) -> str:
    """Get module version string for compatibility with existing code.

    Parameters
    ----------
    modname : str
        Module name to check

    Returns
    -------
    str
        Version string or "Not installed"
    """
    versions = _get_installed_packages()
    return versions.get(modname, "Not installed")
