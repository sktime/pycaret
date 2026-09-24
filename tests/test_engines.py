"""Tests for the ``engine`` argument of the tabular experiments.

Some models can be built with scikit-learn or with scikit-learn-intelex
(``sklearnex``). These tests check that the engine chosen in ``setup``, through
``_set_engine``, or for a single ``create_model`` or ``compare_models`` call is
the one that builds the model, and that a per-call choice does not change the
experiment defaults.

Classification, regression and clustering share the engine API, so every test
runs once per module through the ``setup_*`` helpers below.
"""

import pytest

from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment

# pyproject.toml only installs scikit-learn-intelex on x86_64
pytest.importorskip("sklearnex")


def setup_classification(engine=None):
    """Set up a classification experiment on the juice dataset.

    Parameters
    ----------
    engine : dict, default=None
        Engine to use per model id, for example ``{"lr": "sklearnex"}``.
        By default every model uses scikit-learn.

    Returns
    -------
    ClassificationExperiment
        The experiment after ``setup``.
    """
    exp = ClassificationExperiment()
    exp.setup(
        get_data("juice"),
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
        engine=engine,
    )
    return exp


def setup_regression(engine=None):
    """Set up a regression experiment on the boston dataset.

    Parameters
    ----------
    engine : dict, default=None
        Engine to use per model id, for example ``{"lr": "sklearnex"}``.
        By default every model uses scikit-learn.

    Returns
    -------
    RegressionExperiment
        The experiment after ``setup``.
    """
    exp = RegressionExperiment()
    exp.setup(
        get_data("boston"),
        target="medv",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        html=False,
        session_id=123,
        n_jobs=1,
        engine=engine,
    )
    return exp


def setup_clustering(engine=None):
    """Set up a clustering experiment on the jewellery dataset.

    Parameters
    ----------
    engine : dict, default=None
        Engine to use per model id, for example ``{"kmeans": "sklearnex"}``.
        By default every model uses scikit-learn.

    Returns
    -------
    ClusteringExperiment
        The experiment after ``setup``.
    """
    exp = ClusteringExperiment()
    exp.setup(
        get_data("jewellery"),
        normalize=True,
        log_experiment=True,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        html=False,
        session_id=123,
        n_jobs=1,
        engines=engine,  # unsupervised setups take the plural, supervised the singular
    )
    return exp


def assert_engine(model, engine):
    """Assert that ``model`` was built by the library behind ``engine``.

    The check uses the top-level package of the model's class instead of its
    type, because the sklearnex engine returns either an ``sklearnex`` or a
    ``daal4py`` class depending on the estimator.

    Parameters
    ----------
    model : estimator
        Model returned by ``create_model`` or ``compare_models``.
    engine : {"sklearn", "sklearnex"}
        Engine that should have built the model.
    """
    library = model.__module__.split(".")[0]
    if engine == "sklearnex":
        assert library in ("sklearnex", "daal4py"), model.__module__
    else:
        assert library == engine, model.__module__


@pytest.mark.parametrize(
    "setup, algo",
    [
        (setup_classification, "lr"),
        (setup_regression, "lr"),
        (setup_clustering, "kmeans"),
    ],
)
def test_engine_set_in_setup(setup, algo):
    """The engine passed to ``setup`` becomes the default for that model."""
    exp = setup(engine={algo: "sklearnex"})
    assert exp.get_engine(algo) == "sklearnex"
    assert_engine(exp.create_model(algo), "sklearnex")


@pytest.mark.parametrize(
    "setup, algo",
    [
        (setup_classification, "lr"),
        (setup_regression, "lr"),
        (setup_clustering, "kmeans"),
    ],
)
def test_set_engine_changes_default(setup, algo):
    """``_set_engine`` replaces the default engine for later ``create_model`` calls."""
    exp = setup(engine={algo: "sklearnex"})
    exp._set_engine(algo, "sklearn")
    assert exp.get_engine(algo) == "sklearn"
    assert_engine(exp.create_model(algo), "sklearn")


@pytest.mark.parametrize(
    "setup, algo",
    [
        (setup_classification, "lr"),
        (setup_regression, "lr"),
        (setup_clustering, "kmeans"),
    ],
)
def test_create_model_engine_is_local(setup, algo):
    """An engine passed to ``create_model`` does not change the default."""
    exp = setup()
    assert exp.get_engine(algo) == "sklearn"
    assert_engine(exp.create_model(algo), "sklearn")

    assert_engine(exp.create_model(algo, engine="sklearnex"), "sklearnex")
    assert exp.get_engine(algo) == "sklearn"


@pytest.mark.parametrize("setup", [setup_classification, setup_regression])
def test_compare_models_engine_is_local(setup):
    """An engine passed to ``compare_models`` does not change the default."""
    exp = setup()
    assert_engine(exp.compare_models(include=["lr"]), "sklearn")
    assert exp.get_engine("lr") == "sklearn"

    model = exp.compare_models(include=["lr"], engine={"lr": "sklearnex"})
    assert_engine(model, "sklearnex")
    assert exp.get_engine("lr") == "sklearn"
    assert_engine(exp.compare_models(include=["lr"]), "sklearn")


@pytest.mark.parametrize(
    "setup, algo",
    [
        (setup_classification, "lr"),
        (setup_classification, "knn"),
        (setup_classification, "rbfsvm"),
        (setup_regression, "lr"),
        (setup_regression, "lasso"),
        (setup_regression, "ridge"),
        (setup_regression, "en"),
        (setup_regression, "knn"),
        (setup_regression, "svm"),
        (setup_clustering, "kmeans"),
        (setup_clustering, "dbscan"),
    ],
)
def test_every_model_with_sklearnex_engine(setup, algo):
    """Every model that offers a sklearnex engine can be created with it."""
    exp = setup()
    assert_engine(exp.create_model(algo), "sklearn")
    assert_engine(exp.create_model(algo, engine="sklearnex"), "sklearnex")
