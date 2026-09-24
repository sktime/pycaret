from skbase.utils.dependencies import _check_soft_dependencies

from pycaret.utils._dependencies import _install_pycaret_extra_msg

if _check_soft_dependencies(
    "fugue",
    severity="error",
    msg=_install_pycaret_extra_msg("fugue", "parallel"),
):
    from .fugue_backend import FugueBackend
