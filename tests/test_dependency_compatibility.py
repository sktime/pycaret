"""Regressions for dependency API changes exposed by sktime 1.2."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fugue import transform
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from pycaret.classification import ClassificationExperiment
from pycaret.containers.models.classification import AdaBoostClassifierContainer
from pycaret.containers.models.regression import DecisionTreeRegressorContainer
from pycaret.parallel import FugueBackend
from pycaret.parallel.fugue_backend import _DisplayUtil
from pycaret.utils.generic import to_df
from pycaret.utils.time_series import clean_time_index
from pycaret.utils.time_series.forecasting import model_selection


def test_decision_tree_tuning_criteria_fit_after_set_params():
    container = DecisionTreeRegressorContainer(SimpleNamespace(seed=42))
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0])

    for criterion in container.tune_grid["criterion"]:
        model = container.class_def(**container.args)
        model.set_params(criterion=criterion).fit(X, y)
        assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize("error_score", [np.nan, "raise"])
def test_forecasting_search_failed_fit_preserves_cutoff_and_error(
    monkeypatch, error_score
):
    class FailedPipeline:
        cutoff = None
        is_fitted = False

        def fit(self, y, X, **params):
            raise ValueError("invalid candidate parameter")

    y = pd.Series(
        [1.0, 2.0, 3.0], index=pd.period_range("2020-01", periods=3, freq="M")
    )
    monkeypatch.setattr(
        model_selection, "_get_imputed_data", lambda **kwargs: (y, None)
    )

    def evaluate():
        return model_selection._fit_and_score(
            pipeline=FailedPipeline(),
            y=y,
            X=None,
            scoring={"mae": "neg_mean_absolute_error"},
            train=np.array([0, 1]),
            test=np.array([2]),
            parameters=None,
            fit_params={},
            return_train_score=False,
            alpha=None,
            coverage=0.9,
            error_score=error_score,
        )

    if error_score == "raise":
        with pytest.raises(ValueError, match="invalid candidate parameter"):
            evaluate()
    else:
        scores, _, _, cutoff = evaluate()
        assert np.isnan(scores["mae"])
        assert cutoff == y.index[1]


@pytest.mark.parametrize(
    "freq, offset",
    [("2H", pd.offsets.Hour(2)), ("A-JUN", pd.offsets.YearEnd(month=6))],
)
def test_clean_time_index_legacy_alias_with_missing_period(freq, offset):
    dates = pd.date_range("2019-01-01", periods=4, freq=offset)
    data = pd.DataFrame({"date": dates.astype(str), "value": [1.0, 2.0, 3.0, 4.0]})
    # A string index column with row labels that do not include zero.
    data.index = [10, 20, 30, 40]
    data = data.drop(index=20)

    result = clean_time_index(data, freq=freq, index_col="date")

    assert isinstance(result.index, pd.PeriodIndex)
    assert result.index.freq == offset
    assert len(result) == 4
    assert pd.isna(result.iloc[1]["value"])
    assert result.iloc[[0, 2, 3]]["value"].tolist() == [1.0, 3.0, 4.0]


def test_scipy_sparse_input_preserves_zeros_and_missing_values():
    values = np.array([[0.0, 1.0], [np.nan, 0.0]])
    result = to_df(csr_matrix(values))

    assert all(isinstance(dtype, pd.SparseDtype) for dtype in result.dtypes)
    np.testing.assert_equal(result.sparse.to_dense().to_numpy(), values)


def test_qda_handles_collinear_features():
    X, y = make_classification(
        n_samples=80, n_features=4, n_redundant=0, random_state=42
    )
    data = pd.DataFrame(X)
    data["duplicate"] = data[0]
    experiment = ClassificationExperiment()
    experiment.setup(
        data, target=y, fold=2, session_id=42, n_jobs=1, html=False, verbose=False
    )

    model = experiment.create_model("qda", verbose=False)

    assert np.isfinite(model.predict_proba(experiment.X_test_transformed)).all()


def test_adaboost_tuning_uses_supported_parameters():
    container = AdaBoostClassifierContainer(SimpleNamespace(seed=42))
    model = container.class_def(**container.args)

    # Exercise every grid parameter through sklearn's validation API.
    model.set_params(**{key: values[0] for key, values in container.tune_grid.items()})
    assert ("algorithm" in container.tune_grid) == ("algorithm" in model.get_params())


def test_fugue_transform_without_rpc_callback(monkeypatch):
    class Experiment:
        def compare_models(self, include, **params):
            return include[0]

        def pull(self):
            return pd.DataFrame({"score": [1.0]})

    backend = FugueBackend()
    backend._params = {"include": ["model"], "n_select": 1}
    monkeypatch.setattr(backend, "remote_setup", Experiment)

    result = transform(
        pd.DataFrame({"idx": [0]}),
        backend._remote_compare_models_without_report,
        schema="output:binary",
    )

    assert len(result) == 1
    assert isinstance(result.iloc[0, 0], bytes)


def test_remote_display_accepts_json_payload():
    class Display:
        def move_progress(self, count):
            self.count = count

        def display(self, frame, final_display):
            self.frame = frame

    display = Display()
    utility = _DisplayUtil(display, progress=1, verbose=False, sort="score", asc=False)
    scores = pd.DataFrame({"Model": ["model"], "score": [0.875]}, index=["id"])

    utility.update(scores.to_json(orient="split", double_precision=15))

    assert display.count == 1
    pd.testing.assert_frame_equal(display.frame, scores)
