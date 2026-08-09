import matplotlib.cm
import matplotlib.pyplot

# matplotlib 3.11 removed matplotlib.cm.get_cmap, which yellowbrick (unmaintained)
# still calls in yellowbrick.style.colors and yellowbrick.features.base.
# Restore it until yellowbrick is replaced, see #49.

if not hasattr(matplotlib.cm, "get_cmap"):
    matplotlib.cm.get_cmap = matplotlib.pyplot.get_cmap

from yellowbrick.utils.helpers import get_model_name as get_model_name_original

from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator


def is_estimator(model):
    try:
        return callable(getattr(model, "fit"))
    except Exception:
        return False


def get_model_name(model):
    return get_model_name_original(get_estimator_from_meta_estimator(model))
