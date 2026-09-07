"""Tests for the functional API of the anomaly module.

All tests share one experiment and one fitted instance of each model. Each
test checks a single step on them: assigning, predicting, plotting, saving,
configuration or mlflow logging.
"""

import uuid

import mlflow
import pandas as pd
import pytest
from mlflow_test_utils import mlflow_run_tags

import pycaret.anomaly
from pycaret.datasets import get_data


@pytest.fixture(scope="module")
def data():
    """Dataset that the experiment is set up on."""
    return get_data("anomaly")


@pytest.fixture(scope="module")
def experiment_name(tmp_path_factory):
    """Name of an mlflow experiment with its own temporary storage.

    Runs and artifacts logged during the tests go there instead of into the
    repository.
    """
    root = tmp_path_factory.mktemp("mlflow")
    previous = mlflow.get_tracking_uri() if mlflow.is_tracking_uri_set() else None
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")

    name = uuid.uuid4().hex
    mlflow.create_experiment(name, artifact_location=str(root / "artifacts"))
    yield name
    mlflow.set_tracking_uri(previous)


@pytest.fixture(scope="module")
def experiment(data, experiment_name):
    """Experiment set up once for the module, with mlflow logging and custom tags."""
    return pycaret.anomaly.setup(
        data,
        normalize=True,
        log_experiment=True,
        experiment_name=experiment_name,
        experiment_custom_tags={"tag": 1},
        log_plots=True,
        html=False,
        session_id=123,
        n_jobs=1,
    )


@pytest.fixture(autouse=True)
def current_experiment(experiment):
    """Activate the module's experiment before each test.

    The functional API works on a global current experiment, which the test
    suite resets after every test.
    """
    pycaret.anomaly.set_current_experiment(experiment)


@pytest.fixture(scope="module", params=["iforest", "knn", "cluster"])
def model(request, experiment):
    """Fitted model, created once per model id for the module.

    Module fixtures run before ``current_experiment``, so the experiment is
    activated here as well.
    """
    pycaret.anomaly.set_current_experiment(experiment)
    return pycaret.anomaly.create_model(
        request.param, experiment_custom_tags={"tag": 1}
    )


def test_assign_model(model, data):
    """``assign_model`` gives every training row an anomaly flag and a score."""
    result = pycaret.anomaly.assign_model(model)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    assert {"Anomaly", "Anomaly_Score"} <= set(result.columns)


def test_predict_model(model, data):
    """``predict_model`` gives every new row an anomaly flag and a score."""
    predictions = pycaret.anomaly.predict_model(model, data=data)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(data)
    assert {"Anomaly", "Anomaly_Score"} <= set(predictions.columns)


@pytest.mark.plotting
def test_plot_model(model):
    """The default plot renders for every model."""
    pycaret.anomaly.plot_model(model)


def test_load_model_predicts_without_setup(model, data, tmp_path):
    """A saved model predicts after loading into an experiment without ``setup``."""
    path = str(tmp_path / "model")
    pycaret.anomaly.save_model(model, path)
    pycaret.anomaly.set_current_experiment(pycaret.anomaly.AnomalyExperiment())

    loaded = pycaret.anomaly.load_model(path)
    predictions = pycaret.anomaly.predict_model(loaded, data=data)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(data)


def test_custom_tags_are_logged(model, experiment_name):
    """Every run carries the tags given to ``setup`` and ``create_model``."""
    tags = mlflow_run_tags(experiment_name)
    assert tags
    assert all(run_tags["tag"] == "1" for run_tags in tags)


def test_get_and_set_config():
    """``get_config`` reads and ``set_config`` writes experiment attributes."""
    assert isinstance(pycaret.anomaly.get_config("X"), pd.DataFrame)
    seed = pycaret.anomaly.get_config("seed")
    assert isinstance(seed, int)

    pycaret.anomaly.set_config("seed", seed + 1)
    assert pycaret.anomaly.get_config("seed") == seed + 1
    # Restore the seed, the experiment is shared with the other tests.
    pycaret.anomaly.set_config("seed", seed)


def test_models():
    """``models`` lists the available models by id."""
    all_models = pycaret.anomaly.models()
    assert isinstance(all_models, pd.DataFrame)
    assert {"iforest", "knn", "cluster"} <= set(all_models.index)
