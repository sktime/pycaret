"""Temporary compatibility fixes for sktime.

The ETS interval regression was introduced by sktime/sktime#10972 and is
reproduced on sktime 1.2.0. Track the upstream correction through
https://github.com/sktime/pycaret/pull/155.

Once an upstream fix is released, narrow the affected version range below to
exclude that release. Remove the wrapper and restore the native ETS container
when PyCaret's supported sktime range no longer includes affected releases.
Keep the public API regression test when removing the workaround.
"""

import warnings
from importlib.metadata import version

import pandas as pd
from sktime.forecasting.ets import AutoETS
from verlib2 import Version


def _needs_ets_prediction_interval_workaround():
    """Limit the workaround to the sktime release series verified affected."""
    return Version("1.2.0") <= Version(version("sktime")) < Version("1.3.0")


def warn_if_native_autoets_is_affected(model):
    """Explain the workaround when a custom native AutoETS bypasses it."""
    if type(model) is AutoETS and _needs_ets_prediction_interval_workaround():
        warnings.warn(
            "Native sktime AutoETS prediction intervals are affected by a "
            "simulation argument bug in sktime 1.2. Use create_model('ets', ...) "
            "to use PyCaret's compatibility workaround.",
            UserWarning,
            stacklevel=3,
        )


class PyCaretAutoETS(AutoETS):
    """Pass simulation arguments directly to statsmodels on affected sktime."""

    def _predict_interval(self, fh, X, coverage):
        if not _needs_ets_prediction_interval_workaround():
            return super()._predict_interval(fh=fh, X=X, coverage=coverage)

        # ETSResults.get_prediction accepts **simulate_kwargs, so the seed must
        # be a direct keyword argument rather than a nested simulate_kwargs dict.
        absolute_fh = fh.to_absolute_int(self._y_first_index, self.cutoff)
        start, end = absolute_fh[[0, -1]]
        fh_int = absolute_fh - self._y_len
        fh_int = fh_int - fh_int[0]

        prediction_results = self._fitted_forecaster.get_prediction(
            start=start, end=end, random_state=self.random_state
        )
        var_names = self._get_varnames()
        columns = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_statsmodels = self._extract_conf_int(prediction_results, 1 - coverage[0])
        pred_int = pd.DataFrame(
            index=pred_statsmodels.iloc[fh_int].index, columns=columns
        )

        for c in coverage:
            pred_statsmodels = self._extract_conf_int(prediction_results, 1 - c)
            pred_int[(var_names[0], c, "lower")] = pred_statsmodels.iloc[fh_int][
                "lower"
            ]
            pred_int[(var_names[0], c, "upper")] = pred_statsmodels.iloc[fh_int][
                "upper"
            ]

        return pred_int
