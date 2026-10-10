"""Tests for the classification module on a target with more than two classes.

Binary classification is covered by ``test_classification.py``. This module
repeats only the steps whose behaviour depends on the number of classes:
metrics, calibration, plotting, prediction and model selection. All tests
share one experiment on the iris dataset, the three models that
``compare_models`` ranks highest, and a logistic regression.
"""

import os

import pandas as pd
import pytest
from sklearn.base import is_classifier

import pycaret.classification
from pycaret.datasets import get_data


@pytest.fixture(scope="module")
def data():
    """Dataset that the experiment is set up on."""
    return get_data("iris")


@pytest.fixture(scope="module")
def experiment(data):
    """Experiment set up once for the module."""
    return pycaret.classification.setup(
        data,
        target="species",
        log_experiment=True,
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
    pycaret.classification.set_current_experiment(experiment)


@pytest.fixture(scope="module")
def top3(experiment):
    """Top three models from ``compare_models``.

    Module fixtures run before ``current_experiment``, so the experiment is
    activated here as well.
    """
    pycaret.classification.set_current_experiment(experiment)
    return pycaret.classification.compare_models(n_select=3)


@pytest.fixture(scope="module")
def lr(experiment):
    """Fitted logistic regression for the steps that need a single model."""
    pycaret.classification.set_current_experiment(experiment)
    return pycaret.classification.create_model("lr")


def test_compare_models_returns_classifiers(top3):
    """``compare_models`` ranks the models and returns the best three."""
    assert len(top3) == 3
    assert all(is_classifier(model) for model in top3)


def test_tune_model(top3):
    """``tune_model`` returns a classifier."""
    assert is_classifier(pycaret.classification.tune_model(top3[0], n_iter=3))


def test_ensemble_model(top3):
    """``ensemble_model`` returns a bagged classifier."""
    assert is_classifier(pycaret.classification.ensemble_model(top3[0]))


def test_blend_models(top3):
    """``blend_models`` returns a voting classifier."""
    assert is_classifier(pycaret.classification.blend_models(top3))


def test_stack_models(top3):
    """``stack_models`` returns a classifier that predicts on the hold-out set."""
    stacked = pycaret.classification.stack_models(estimator_list=top3)
    predictions = pycaret.classification.predict_model(stacked)
    assert "prediction_label" in predictions.columns


def test_calibrate_model(lr):
    """``calibrate_model`` returns a calibrated classifier."""
    assert is_classifier(pycaret.classification.calibrate_model(lr))


@pytest.mark.plotting
def test_plot_model(lr, tmp_path):
    """The default plot is saved to the requested directory."""
    path = pycaret.classification.plot_model(lr, save=str(tmp_path), scale=5)
    assert os.path.exists(path)


def test_automl(top3):
    """``automl`` returns the best classifier trained so far."""
    assert is_classifier(
        pycaret.classification.automl(optimize="MCC", use_holdout=True)
    )
    assert is_classifier(pycaret.classification.automl(optimize="MCC"))


def test_predict_model_on_holdout(lr, data):
    """``predict_model`` scores the hold-out set and predicts every class."""
    predictions = pycaret.classification.predict_model(lr)
    assert len(predictions) == len(pycaret.classification.get_config("X_test"))
    assert set(predictions["prediction_label"]) == set(data["species"])


def test_predict_model_on_data_without_target(lr, data):
    """``predict_model`` labels new data that has no target column."""
    predictions = pycaret.classification.predict_model(
        lr, data=data.drop("species", axis=1)
    )
    assert len(predictions) == len(data)
    assert {"prediction_label", "prediction_score"} <= set(predictions.columns)


def test_finalize_model(lr, data):
    """``finalize_model`` returns a pipeline that predicts on new data."""
    final = pycaret.classification.finalize_model(lr)
    predictions = pycaret.classification.predict_model(final, data=data)
    assert len(predictions) == len(data)


def test_load_model_predicts_without_setup(lr, data, tmp_path):
    """A saved model predicts after loading into an experiment without ``setup``."""
    path = str(tmp_path / "model")
    pycaret.classification.save_model(lr, path)
    pycaret.classification.set_current_experiment(
        pycaret.classification.ClassificationExperiment()
    )

    loaded = pycaret.classification.load_model(path)
    predictions = pycaret.classification.predict_model(loaded, data=data)
    assert len(predictions) == len(data)


def test_get_config_returns_the_split_data(data):
    """``get_config`` exposes the train and test split of the data."""
    X_train = pycaret.classification.get_config("X_train")
    X_test = pycaret.classification.get_config("X_test")
    y_train = pycaret.classification.get_config("y_train")
    y_test = pycaret.classification.get_config("y_test")
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) + len(X_test) == len(data)
    assert len(y_train) + len(y_test) == len(data)
