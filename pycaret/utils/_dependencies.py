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


def _install_pycaret_extra_msg(pkg_name: str, extra: str) -> str:
    """Generate installation message for PyCaret soft dependencies.

    This function generates a custom message for scikit-base's _check_soft_dependencies.
    Since scikit-base appends its own installation instruction, this message only provides
    PyCaret-specific context and the extra installation option.

    Parameters
    ----------
    pkg_name : str
        The Python distribution/package name (e.g., "xgboost", "scikit-optimize")
    extra : str
        The PyCaret extra that includes this package (e.g., "models", "analysis", "tuners")

    Returns
    -------
    str
        Installation message for the soft dependency (without the direct pip install)
    """
    if extra:
        return (
            f"{pkg_name} is a soft dependency and not included in the pycaret "
            f"installation. Alternatively, you can install {pkg_name} by running "
            f"`pip install pycaret-core[{extra}]` "
        )
    else:
        return f"{pkg_name} is a soft dependency and not included in the pycaret installation. "
