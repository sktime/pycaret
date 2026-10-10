"""Tests for ``plot_model`` and ``interpret_model`` of the tabular experiments.

``plot_model`` is called with each plot name that ``ClassificationExperiment``
and ``RegressionExperiment`` list in ``_available_plots``. ``interpret_model``
is called with each SHAP plot, on the training data and on new data. Both
experiments are set up once, with the fitted models the tests need.
"""

import matplotlib
import pytest

from pycaret.classification import ClassificationExperiment
from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment

# Render without a display; the default backend needs tkinter.
matplotlib.use("Agg")

pytestmark = pytest.mark.plotting


@pytest.fixture(scope="module")
def classification(experiment_name):
    """Classification experiment on the juice dataset, logging its plots to mlflow.

    Returns the experiment, its fitted models by id, and ten rows of features
    that serve as new data for ``interpret_model``.
    """
    data = get_data("juice")
    exp = ClassificationExperiment()
    exp.setup(
        data,
        target="Purchase",
        log_experiment=True,
        log_plots=True,
        experiment_name=experiment_name,
        html=False,
        session_id=123,
        fold=2,
        n_jobs=1,
    )
    models = {
        "rf": exp.create_model("rf", max_depth=2, n_estimators=5),
        "et": exp.create_model("et"),
        "xgboost": exp.create_model("xgboost"),
    }
    return exp, models, data.drop(columns="Purchase").head(10)


@pytest.fixture(scope="module")
def regression(experiment_name):
    """Regression experiment on the boston dataset, logging its plots to mlflow.

    Returns the experiment, its fitted models by id, and ten rows of features
    that serve as new data for ``interpret_model``.
    """
    data = get_data("boston")
    exp = RegressionExperiment()
    exp.setup(
        data,
        target="medv",
        log_experiment=True,
        log_plots=True,
        experiment_name=experiment_name,
        html=False,
        session_id=123,
        fold=2,
        n_jobs=1,
    )
    models = {
        "rf": exp.create_model("rf", max_depth=2, n_estimators=5),
        "et": exp.create_model("et"),
        "xgboost": exp.create_model("xgboost"),
    }
    return exp, models, data.drop(columns="medv").head(10)


@pytest.mark.parametrize(
    "experiment, plot",
    [("classification", plot) for plot in ClassificationExperiment()._available_plots]
    + [("regression", plot) for plot in RegressionExperiment()._available_plots],
)
def test_plot_model(experiment, plot, request):
    """``plot_model`` renders the plot for a fitted model without raising."""
    exp, models, _ = request.getfixturevalue(experiment)
    exp.plot_model(models["rf"], plot=plot)


@pytest.mark.parametrize("experiment", ["classification", "regression"])
@pytest.mark.parametrize("algo", ["et", "xgboost"])
@pytest.mark.parametrize("plot", ["summary", "correlation", "reason", "pdp", "msa"])
def test_interpret_model(experiment, algo, plot, request):
    """``interpret_model`` renders the SHAP plot on training data and on new data."""
    exp, models, new_sample = request.getfixturevalue(experiment)
    exp.interpret_model(models[algo], plot=plot)
    exp.interpret_model(models[algo], plot=plot, X_new_sample=new_sample)
