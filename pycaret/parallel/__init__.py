from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies(
    "fugue",
    severity="error",
    msg="fugue is a soft dependency and not included in the pycaret installation. Please run: `pip install 'fugue'` to install. Alternately, you can install fugue by running `pip install pycaret-core[parallel]`",
):
    from .fugue_backend import FugueBackend
