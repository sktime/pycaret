from unittest.mock import patch

import pandas as pd
import pytest

from pycaret.datasets import (
    DatasetReference,
    DatasetRegistry,
    _dataset_registry,
    get_data,
)


def test_datasets():
    #########################
    # Load Local File ####
    #########################

    # # loading dataset
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # data = get_data("test_files/dummy_dataset")
    # assert isinstance(data, pd.DataFrame)
    # rows, cols = data.shape
    # assert rows >= 1
    # assert cols >= 1

    ##############################
    # GitHub Common folder ####
    ##############################

    # loading list of datasets
    index = get_data("index")
    assert isinstance(index, pd.DataFrame)
    rows, cols = index.shape
    assert rows > 1
    assert cols == 8

    # loading dataset
    data = get_data("credit")
    assert isinstance(data, pd.DataFrame)
    rows, cols = data.shape
    assert rows == 24000
    assert cols == 24
    assert data.size == 576000

    ################################
    # GitHub Specific folder ####
    ################################

    folder = "time_series/seasonal"
    # loading list of datasets
    index = get_data("index", folder=folder)
    assert isinstance(index, pd.DataFrame)
    rows, cols = index.shape
    assert rows > 1
    assert cols == 12

    # loading dataset
    data = get_data("1", folder=folder)
    assert isinstance(data, pd.DataFrame)
    rows, cols = data.shape
    assert rows >= 1
    assert cols >= 1

    ###########################
    # `sktime` datasets ####
    ###########################

    # loading dataset
    data = get_data("airline")
    assert isinstance(data, pd.Series)
    rows = len(data)
    assert rows >= 1

    ###########################
    # Incorrect dataset ####
    ###########################

    with pytest.raises(ValueError) as errmsg:
        _ = get_data("wrong")

    exceptionmsg = errmsg.value.args[0]
    assert exceptionmsg == "Data could not be read. Please check your inputs..."


def test_dataset_reference_callable_loader():
    """Test DatasetReference delegating to loader callable."""
    expected = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ref = DatasetReference(
        name="dummy_ref",
        folder="common",
        loader=lambda: expected,
    )
    assert ref.name == "dummy_ref"
    assert ref.folder == "common"
    loaded = ref.load()
    pd.testing.assert_frame_equal(loaded, expected)


def test_dataset_reference_generic_loader():
    """Test DatasetReference loader returns arbitrary object without modification."""
    custom_obj = {"custom": [1, 2, 3]}
    ref = DatasetReference(name="custom_ref", loader=lambda: custom_obj)
    assert ref.load() == custom_obj


def test_sktime_loader_tuple_conversion():
    """Test sktime loader wrapper unpacks (y, X) tuples into a single DataFrame with correct values."""
    y = pd.Series([10.5, 20.5], name="Growth")
    X = pd.DataFrame({"Consumption": [1.2, 3.4], "Income": [5.6, 7.8]})
    expected_df = pd.DataFrame(
        {
            "Growth": [10.5, 20.5],
            "Consumption": [1.2, 3.4],
            "Income": [5.6, 7.8],
        }
    )

    with patch("pycaret.datasets._check_soft_dependencies"):
        with patch(
            "sktime.datasets.load_uschange",
            return_value=(y, X),
            create=True,
        ):
            loaded = get_data("uschange", verbose=False)
            assert isinstance(loaded, pd.DataFrame)
            pd.testing.assert_frame_equal(loaded, expected_df)


def test_registered_loader_returning_none_raises():
    """Test that a registered loader returning None raises ValueError without invoking remote fallback."""
    none_ref = DatasetReference(name="none_ds", loader=lambda: None)
    _dataset_registry.register(none_ref)
    try:
        with (
            patch("pycaret.datasets.requests.head") as mock_head,
            pytest.raises(
                ValueError,
                match="Loader for registered dataset reference 'none_ds' returned no data.",
            ),
        ):
            get_data("none_ds", verbose=False)
        mock_head.assert_not_called()
    finally:
        _dataset_registry._datasets.pop("none_ds", None)


def test_dataset_reference_no_loader_raises():
    """Test DatasetReference without loader raises ValueError."""
    ref = DatasetReference(name="empty_ref")
    with pytest.raises(ValueError, match="No loader configured"):
        ref.load()


def test_dataset_registry_operations():
    """Test DatasetRegistry register and get behavior."""
    reg = DatasetRegistry()
    ref1 = DatasetReference(name="d1", folder="common")
    ref2 = DatasetReference(name="d2", folder="time_series")
    reg.register(ref1)
    reg.register(ref2)

    assert reg.get("d1") is ref1
    assert reg.get("d2") is ref2
    assert reg.get("nonexistent") is None


def test_registered_loader_exception_propagates():
    """Test that errors in registered loaders propagate rather than being swallowed."""

    def broken_loader():
        raise RuntimeError("Custom database connection error")

    broken_ref = DatasetReference(name="broken_ds", loader=broken_loader)
    _dataset_registry.register(broken_ref)
    try:
        with pytest.raises(RuntimeError, match="Custom database connection error"):
            get_data("broken_ds", verbose=False)
    finally:
        _dataset_registry._datasets.pop("broken_ds", None)


def test_sktime_missing_dependency_raises():
    """Test that missing sktime produces ModuleNotFoundError/ImportError."""
    with patch(
        "pycaret.datasets._check_soft_dependencies",
        side_effect=ModuleNotFoundError("sktime is not installed"),
    ):
        with pytest.raises(ModuleNotFoundError, match="sktime is not installed"):
            get_data("airline", verbose=False)


def test_local_file_precedence(tmp_path, monkeypatch):
    """Test that local file has precedence over registered and remote datasets."""
    monkeypatch.chdir(tmp_path)
    local_file = "airline.csv"
    df_local = pd.DataFrame({"local_col": [999, 888]})
    df_local.to_csv(local_file, index=False)

    loaded = get_data("airline", verbose=False)
    assert isinstance(loaded, pd.DataFrame)
    assert "local_col" in loaded.columns
    assert loaded["local_col"].tolist() == [999, 888]


def test_registered_dataset_save_copy(tmp_path, monkeypatch):
    """Test save_copy=True works as expected with registered datasets."""
    monkeypatch.chdir(tmp_path)
    dummy_df = pd.DataFrame({"val": [1, 2, 3]})
    save_file = "test_save_copy_ds.csv"
    dummy_ref = DatasetReference(name="test_save_copy_ds", loader=lambda: dummy_df)
    _dataset_registry.register(dummy_ref)
    try:
        loaded = get_data("test_save_copy_ds", save_copy=True, verbose=False)
        assert (tmp_path / save_file).is_file()
        pd.testing.assert_frame_equal(loaded, dummy_df)
        saved_disk = pd.read_csv(save_file)
        pd.testing.assert_frame_equal(saved_disk, dummy_df)
    finally:
        _dataset_registry._datasets.pop("test_save_copy_ds", None)


def test_custom_address():
    """Test that an explicit address parameter is used to resolve the dataset URL.

    Verifies that requests.head and pandas.read_csv are both called with
    ``<address>/<dataset>.csv`` and that the returned DataFrame matches the mock.
    """
    mock_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    class _MockResponse:
        status_code = 200

    with (
        patch(
            "pycaret.datasets.requests.head", return_value=_MockResponse()
        ) as mock_head,
        patch("pandas.read_csv", return_value=mock_df) as mock_read,
    ):
        result = get_data(
            "my_dataset", address="https://custom.mirror.org", verbose=False
        )

    mock_head.assert_called_once_with("https://custom.mirror.org/my_dataset.csv")
    mock_read.assert_called_once_with("https://custom.mirror.org/my_dataset.csv")
    pd.testing.assert_frame_equal(result, mock_df)
