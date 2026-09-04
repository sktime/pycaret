"""Module to test time_series models"""

from types import SimpleNamespace

import pandas as pd  # type: ignore
import pytest

from pycaret.containers.models import time_series
from pycaret.time_series import TSForecastingExperiment

##########################
# Tests Start Here ####
##########################


def test_naive_models(load_pos_and_neg_data):
    """Tests enabling and disabling of naive models"""

    exp = TSForecastingExperiment()
    data = load_pos_and_neg_data

    # Seasonal Period != 1 ----
    # All naive models should be enabled here
    exp.setup(data=data, verbose=False)
    expected = ["naive", "grand_means", "snaive"]
    for model in expected:
        assert model in exp.models().index

    # Seasonal Period == 1 ----
    # snaive should be disabled here
    exp.setup(data=data, seasonal_period=1, verbose=False)
    expected = ["naive", "grand_means"]
    for model in expected:
        assert model in exp.models().index
    not_expected = ["snaive"]
    for model in not_expected:
        assert model not in exp.models().index


@pytest.mark.parametrize(
    "container_class", [time_series.BATSContainer, time_series.TBATSContainer]
)
def test_bats_containers_disabled_when_numpy_2_is_installed(
    monkeypatch, container_class
):
    """BATS adapters must not be initialized when their NumPy constraint fails."""

    def check_dependencies(package, *args, **kwargs):
        assert package == "numpy<2"
        assert kwargs == {"severity": "none"}
        return False

    monkeypatch.setattr(time_series, "_check_soft_dependencies", check_dependencies)

    container = container_class(SimpleNamespace(seed=42))

    assert not container.active


def test_bats_models_omitted_when_numpy_2_is_installed(
    monkeypatch, load_pos_and_neg_data
):
    """Inactive BATS containers are excluded from model and tuning registries."""
    check_soft_dependencies = time_series._check_soft_dependencies

    def check_dependencies(package, *args, **kwargs):
        if package == "numpy<2":
            return False
        return check_soft_dependencies(package, *args, **kwargs)

    monkeypatch.setattr(time_series, "_check_soft_dependencies", check_dependencies)
    exp = TSForecastingExperiment()
    exp.setup(data=load_pos_and_neg_data, verbose=False)

    unavailable_models = {"bats", "tbats"}
    assert unavailable_models.isdisjoint(exp.models().index)
    assert unavailable_models.isdisjoint(exp._all_models_internal)


def test_custom_models(load_pos_data):
    """Tests working with custom models"""

    exp = TSForecastingExperiment()
    data = load_pos_data

    exp.setup(
        data=data,
        fh=12,
        session_id=42,
    )

    # Create a sktime pipeline with preprocessing ----
    from sktime.forecasting.arima import ARIMA
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.transformations.series.boxcox import LogTransformer
    from sktime.transformations.series.impute import Imputer

    forecaster = TransformedTargetForecaster(
        [
            ("impute", Imputer()),
            ("log", LogTransformer()),
            ("model", ARIMA(seasonal_order=(0, 1, 0, 12))),
        ]
    )

    ##################################
    # Test Create Custom Model ----
    ##################################
    my_custom_model = exp.create_model(forecaster)
    assert type(my_custom_model) is type(forecaster)

    ################################
    # Test Tune Custom Model ----
    ################################
    impute_values = ["drift", "bfill", "ffill"]
    my_grid = {"impute__method": impute_values}
    tuned_model, tuner = exp.tune_model(
        my_custom_model,
        custom_grid=my_grid,
        return_tuner=True,
    )
    assert type(tuned_model) is type(forecaster)
    assert "param_forecaster__model__impute__method" in pd.DataFrame(tuner.cv_results_)
    for index, method in enumerate(
        tuner.cv_results_.get("param_forecaster__model__impute__method")
    ):
        assert method == impute_values[index]

    ############################
    # Test Tuning raises ----
    ############################
    # No custom grid passed when tuning custom model
    with pytest.raises(ValueError) as errmsg:
        _ = exp.tune_model(my_custom_model)

    # Capture Error message
    exceptionmsg = errmsg.value.args[0]

    # Check exact error received
    assert (
        "When passing a model not in PyCaret's model library, the custom_grid parameter must be provided."
        in exceptionmsg
    )
