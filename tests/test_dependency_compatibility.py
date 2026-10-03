"""Regressions for dependency API changes exposed by sktime 1.2."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from fugue import transform
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from pycaret.classification import ClassificationExperiment
from pycaret.containers.models.classification import AdaBoostClassifierContainer
from pycaret.parallel import FugueBackend
from pycaret.parallel.fugue_backend import _DisplayUtil
from pycaret.utils.generic import to_df


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
