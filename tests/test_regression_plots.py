import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

import pycaret.datasets
import pycaret.regression


@pytest.mark.plotting
def test_plot():
    data = pycaret.datasets.get_data("boston")
    assert isinstance(data, pd.DataFrame)

    pycaret.regression.setup(
        data,
        target="medv",
        log_experiment=True,
        log_plots=True,
        html=False,
        session_id=123,
        fold=2,
        n_jobs=1,
    )

    model = pycaret.regression.create_model("rf", max_depth=2, n_estimators=5)

    exp = pycaret.regression.RegressionExperiment()
    available_plots = exp._available_plots

    skip_plots = set()
    if not _check_soft_dependencies("matplotlib<3.8", severity="none"):
        skip_plots.add("cooks")

    for plot in available_plots:
        if plot in skip_plots:
            continue
        pycaret.regression.plot_model(model, plot=plot)

    models = [
        pycaret.regression.create_model("et"),
        pycaret.regression.create_model("xgboost"),
    ]

    # no pfi due to dependency hell
    available_shap = ["summary", "correlation", "reason", "pdp", "msa"]

    for model in models:
        for plot in available_shap:
            pycaret.regression.interpret_model(model, plot=plot)
            pycaret.regression.interpret_model(
                model, plot=plot, X_new_sample=data.drop("medv", axis=1).iloc[:10]
            )

    assert 1 == 1


@pytest.mark.plotting
def test_cooks_plot_rejects_unsupported_matplotlib(monkeypatch):
    data = pycaret.datasets.get_data("boston")
    pycaret.regression.setup(
        data,
        target="medv",
        html=False,
        session_id=123,
        fold=2,
        n_jobs=1,
    )
    model = pycaret.regression.create_model("lr")

    calls = []

    def check_soft_dependencies(package, severity):
        calls.append((package, severity))
        return False

    monkeypatch.setattr(
        "skbase.utils.dependencies._check_soft_dependencies",
        check_soft_dependencies,
    )

    with pytest.raises(NotImplementedError) as exc_info:
        pycaret.regression.plot_model(model, plot="cooks")

    assert calls == [("matplotlib<3.8", "none")]
    assert str(exc_info.value) == (
        "The 'cooks' plot is not available with matplotlib >= 3.8.0 "
        "due to an incompatibility in the yellowbrick library. "
        "See https://github.com/DistrictDataLabs/yellowbrick/issues/1234 "
        "for more information. Please use matplotlib < 3.8.0 or choose "
        "a different plot type."
    )


if __name__ == "__main__":
    test_plot()
