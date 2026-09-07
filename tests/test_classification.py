"""Tests for the functional API of the classification module.

All tests share one experiment on the juice dataset, the three models that
``compare_models`` ranks highest, and a logistic regression. Each test checks
a single step of the workflow on them.
"""

import os

import pandas as pd
import pytest
from mlflow_test_utils import mlflow_run_tags
from sklearn.base import is_classifier
from sklearn.metrics import recall_score

import pycaret.classification
from pycaret.datasets import get_data


@pytest.fixture(scope="module")
def data():
    """Dataset that the experiment is set up on."""
    return get_data("juice")


@pytest.fixture(scope="module")
def experiment(data, experiment_name):
    """Experiment set up once for the module, with mlflow logging and custom tags."""
    return pycaret.classification.setup(
        data,
        target="Purchase",
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        experiment_name=experiment_name,
        experiment_custom_tags={"tag": 1},
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
def comparison(experiment):
    """Top three models from ``compare_models`` and the metrics table it displayed.

    Module fixtures run before ``current_experiment``, so the experiment is
    activated here as well.
    """
    pycaret.classification.set_current_experiment(experiment)
    models = pycaret.classification.compare_models(
        n_select=3, errors="raise", experiment_custom_tags={"pytest": "testing"}
    )
    return models, pycaret.classification.pull()


@pytest.fixture(scope="module")
def top3(comparison):
    """Top three models from ``compare_models``."""
    return comparison[0]


@pytest.fixture(scope="module")
def lr(experiment):
    """Fitted logistic regression for the steps that need a single model."""
    pycaret.classification.set_current_experiment(experiment)
    return pycaret.classification.create_model("lr")


def test_compare_models_scores_every_model(comparison):
    """Every model except the dummy scores above zero on every metric."""
    _, metrics = comparison
    scores = metrics.drop(index="dummy", errors="ignore")
    scores = scores.drop(columns=["Model", "TT (Sec)"])
    assert (scores != 0).all().all()


def test_create_model_reports_cross_validation_scores():
    """The metrics table has one row per fold and a mean and standard deviation."""
    pycaret.classification.create_model("lr", fold=3)
    assert list(pycaret.classification.pull().index) == [0, 1, 2, "Mean", "Std"]


def test_create_model_with_return_train_score():
    """With ``return_train_score`` the metrics table also holds the training scores."""
    pycaret.classification.create_model("lr", fold=3, return_train_score=True)
    index = pycaret.classification.pull().index
    assert {"CV-Train", "CV-Val"} <= set(index.get_level_values(0))
    assert "Mean" in index.get_level_values(1)


def test_create_model_without_cross_validation():
    """Without cross-validation the model is scored on the hold-out set only."""
    pycaret.classification.create_model("dt", cross_validation=False)
    assert list(pycaret.classification.pull().index) == ["Test"]


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_tune_model(top3, rank):
    """``tune_model`` returns a classifier for each of the best models."""
    tuned = pycaret.classification.tune_model(top3[rank], n_iter=3)
    assert is_classifier(tuned)


def test_tune_model_choose_better(top3):
    """With ``choose_better`` the result is still a classifier."""
    tuned = pycaret.classification.tune_model(top3[0], n_iter=3, choose_better=True)
    assert is_classifier(tuned)


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


def test_automl(comparison):
    """``automl`` returns the best classifier trained so far."""
    assert is_classifier(
        pycaret.classification.automl(optimize="MCC", use_holdout=True)
    )
    assert is_classifier(pycaret.classification.automl(optimize="MCC"))


def test_predict_model_on_holdout(lr):
    """``predict_model`` scores the hold-out set when given no data."""
    predictions = pycaret.classification.predict_model(lr)
    assert len(predictions) == len(pycaret.classification.get_config("X_test"))
    assert {"prediction_label", "prediction_score"} <= set(predictions.columns)


def test_predict_model_on_new_data(lr, data):
    """``predict_model`` labels every row of new data."""
    predictions = pycaret.classification.predict_model(lr, data=data)
    assert len(predictions) == len(data)
    assert {"prediction_label", "prediction_score"} <= set(predictions.columns)


def test_finalize_model(lr, data):
    """``finalize_model`` returns a pipeline that predicts on new data."""
    final = pycaret.classification.finalize_model(lr)
    predictions = pycaret.classification.predict_model(final, data=data)
    assert len(predictions) == len(data)


def test_load_model_into_the_experiment(lr, tmp_path):
    """A model loaded into a set-up experiment shares the experiment's cache."""
    path = str(tmp_path / "model")
    pycaret.classification.save_model(lr, path)
    loaded = pycaret.classification.load_model(path)
    assert loaded.memory is pycaret.classification.get_config("memory")


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


def test_add_metric(lr):
    """A custom metric is scored during cross-validation and prediction."""

    def specificity(y_true, y_pred, **kwargs):
        return recall_score(y_true, y_pred, pos_label=0, zero_division=1)

    pycaret.classification.add_metric(
        "specificity", "specificity", specificity, greater_is_better=True
    )
    try:
        pycaret.classification.create_model("lr")
        assert pycaret.classification.pull()["specificity"].sum() != 0

        pycaret.classification.predict_model(lr)
        assert pycaret.classification.pull()["specificity"].sum() != 0
    finally:
        # Remove the metric, the experiment is shared with the other tests.
        pycaret.classification.remove_metric("specificity")


def test_custom_tags_are_logged(comparison, experiment_name):
    """The runs of ``setup`` and of ``compare_models`` carry the tags given to each."""
    tags = mlflow_run_tags(experiment_name)
    assert any(run_tags.get("tag") == "1" for run_tags in tags)
    assert any(run_tags.get("pytest") == "testing" for run_tags in tags)


@pytest.mark.parametrize(
    "custom_tags", ["custom_tag", 1, ("pytest", "True"), True, 1.0]
)
def test_setup_rejects_custom_tags_that_are_not_a_dict(data, custom_tags):
    """``setup`` raises when ``experiment_custom_tags`` is not a dictionary."""
    with pytest.raises(TypeError):
        pycaret.classification.ClassificationExperiment().setup(
            data,
            target="Purchase",
            log_experiment=True,
            html=False,
            session_id=123,
            n_jobs=1,
            experiment_custom_tags=custom_tags,
        )


def test_models():
    """``models`` lists the available models by id."""
    all_models = pycaret.classification.models()
    assert isinstance(all_models, pd.DataFrame)
    assert {"lr", "dt", "rf"} <= set(all_models.index)


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


def test_set_config():
    """``set_config`` writes an experiment attribute that ``get_config`` reads."""
    seed = pycaret.classification.get_config("seed")
    pycaret.classification.set_config("seed", seed + 1)
    assert pycaret.classification.get_config("seed") == seed + 1
    # Restore the seed, the experiment is shared with the other tests.
    pycaret.classification.set_config("seed", seed)
