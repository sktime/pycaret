from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sklearn.pipeline import Pipeline

pytest.importorskip("mlflow")

from pycaret.loggers.mlflow_logger import MlflowLogger


def test_log_sklearn_pipeline_uses_cloudpickle_serialization():
    logger = MlflowLogger()
    default_conda_env = {
        "dependencies": ["python=3.10", "pip", {"pip": []}],
    }

    with (
        patch(
            "pycaret.loggers.mlflow_logger.set_active_mlflow_run",
            return_value=nullcontext(),
        ),
        patch("mlflow.sklearn.get_default_conda_env", return_value=default_conda_env),
        patch("mlflow.sklearn.log_model", autospec=True) as log_model,
    ):
        logger.log_sklearn_pipeline(
            SimpleNamespace(exp_name_log="test-experiment"),
            Pipeline([]),
            Pipeline([]),
        )

    assert log_model.call_args.kwargs["serialization_format"] == "cloudpickle"
