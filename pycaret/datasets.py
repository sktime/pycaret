"""Module to get datasets in pycaret"""

from typing import Any, Callable, Dict, Optional

import requests

from pycaret.utils._dependencies import _check_soft_dependencies

# Module-level constants
DEFAULT_GITHUB_ROOT = "https://raw.githubusercontent.com/pycaret/datasets/main/"
DATA_DIR = "data/"
META_DIR = "meta/"
DEFAULT_FOLDER = "common"
SKTIME_DATASETS = ("airline", "lynx", "uschange")


class DatasetReference:
    """Reference to a dataset holding identity and a loader callable.

    This class separates dataset identity from the actual loading mechanism,
    allowing alternative backends (such as Hugging Face or custom stores)
    without altering get_data() callers.
    """

    def __init__(
        self,
        name: str,
        folder: str = DEFAULT_FOLDER,
        loader: Optional[Callable[[], Any]] = None,
    ):
        """Initialize dataset reference.

        Parameters
        ----------
        name : str
            Dataset name / identifier.
        folder : str, default='common'
            Dataset category folder.
        loader : Optional[Callable[[], Any]], default=None
            Callable that executes loading and returns the dataset.
        """
        self.name = name
        self.folder = folder
        self.loader = loader

    def load(self) -> Any:
        """Load and return the dataset.

        Returns
        -------
        Any
            Dataset object returned by the loader callable.
        """
        if self.loader is None:
            raise ValueError(
                f"No loader configured for dataset reference '{self.name}'"
            )
        return self.loader()


class DatasetRegistry:
    """Registry for dataset references.

    Provides centralized management of dataset references and allows
    registering custom loaders or alternative storage backends.
    """

    def __init__(self):
        self._datasets: Dict[str, DatasetReference] = {}

    def register(self, dataset_ref: DatasetReference) -> None:
        """Register a dataset reference.

        Parameters
        ----------
        dataset_ref : DatasetReference
            Dataset reference to register.
        """
        self._datasets[dataset_ref.name] = dataset_ref

    def get(self, name: str) -> Optional[DatasetReference]:
        """Get dataset reference by name.

        Parameters
        ----------
        name : str
            Dataset name.

        Returns
        -------
        Optional[DatasetReference]
            Dataset reference if found, None otherwise.
        """
        return self._datasets.get(name)


# Global dataset registry
_dataset_registry = DatasetRegistry()


def _make_sktime_loader(dataset_name: str) -> Callable[[], Any]:
    """Create a lazy loader for sktime datasets that validates dependency when called."""

    def _loader():
        _check_soft_dependencies("sktime", severity="error")
        import importlib

        import pandas as pd

        sktime_datasets = importlib.import_module("sktime.datasets")
        func_name = f"load_{dataset_name}"
        loader_func = getattr(sktime_datasets, func_name, None)
        if loader_func is None:
            raise ValueError(f"Unknown sktime dataset: {dataset_name}")
        data = loader_func()
        if isinstance(data, tuple):
            y = data[0]
            X = data[1]
            data = pd.concat([y, X], axis=1)
        return data

    return _loader


def _initialize_registry() -> None:
    """Initialize registry with built-in reference entries.

    Registers sktime datasets lazily. No dependency checks or imports
    occur at module import time.
    """
    for name in SKTIME_DATASETS:
        _dataset_registry.register(
            DatasetReference(
                name=name,
                folder=DEFAULT_FOLDER,
                loader=_make_sktime_loader(name),
            )
        )


# Initialize built-in dataset references
_initialize_registry()


def get_data(
    dataset: str = "index",
    folder: Optional[str] = None,
    save_copy: bool = False,
    profile: bool = False,
    verbose: bool = True,
    address: Optional[str] = None,
):
    """
    Function to load sample datasets.

    Order of read:
    (1) Tries to read dataset from local folder first.
    (2) Then tries to read from registered dataset loaders (e.g., sktime)
    (3) Then tries to read dataset from folder in GitHub "address" (see below)
    (4) Raises error if none exist

    List of available datasets on GitHub can be checked using
    (1) ``get_data('index')`` or
    (2) ``get_data('index', folder='time_series/seasonal)``
    (see available "folder" options below)


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> all_datasets = get_data('index')
    >>> juice = get_data('juice')


    dataset: str, default = 'index'
        Index value of dataset.


    folder: Optional[str], default = None
        The folder from which to get the data.
        If 'None', gets it from the "common" folder. Other options are:
            - time_series/seasonal
            - time_series/random_walk
            - time_series/white_noise


    save_copy: bool, default = False
        When set to true, it saves a copy in current working directory.


    profile: bool, default = False
        When set to true, an interactive EDA report is displayed.


    verbose: bool, default = True
        When set to False, head of data is not displayed.

    address: string, default = None
        Download url of dataset. Defaults to None which fetches the dataset from
        "https://raw.githubusercontent.com/pycaret/datasets/main/". For people
        having difficulty linking to github, they can change the default address
        to their own
        (e.g. "https://gitee.com/IncubatorShokuhou/pycaret/raw/master/datasets/")


    Returns:
        pandas.DataFrame


    Warnings
    --------
    - Use of ``get_data`` requires internet connection.


    Raises
    ------
    ImportError
        (1) When trying to import time series datasets that require sktime,
        but sktime has not been installed.
        (2) If the data does not exist
    """

    import os.path

    import pandas as pd

    from pycaret.internal.display import CommonDisplay

    extension = ".csv"
    filename = str(dataset) + extension

    folder_name = DEFAULT_FOLDER if folder is None else folder

    if address is None:
        root = DEFAULT_GITHUB_ROOT
        if dataset == "index":
            complete_address = root + META_DIR + folder_name + "/" + filename
        else:
            complete_address = root + DATA_DIR + folder_name + "/" + filename
    else:
        complete_address = address + "/" + filename

    data = None

    # (1) Read the file name from local folder first
    if os.path.isfile(filename):
        data = pd.read_csv(filename)

    # (2) Try loading from registered dataset reference (e.g., sktime)
    if data is None:
        dataset_ref = _dataset_registry.get(dataset)
        if dataset_ref is not None:
            data = dataset_ref.load()
            if data is None:
                raise ValueError(
                    f"Loader for registered dataset reference '{dataset}' returned no data."
                )

    # (3) Try HTTP request from GitHub or specified address
    if data is None and requests.head(complete_address).status_code == 200:
        data = pd.read_csv(complete_address)

    # (4) If still no data, raise error
    if data is None:
        raise ValueError("Data could not be read. Please check your inputs...")

    data = data.infer_objects()

    if save_copy:
        save_name = filename
        data.to_csv(save_name, index=False)

    display = CommonDisplay(
        verbose=verbose,
        html_param=True,
    )

    if dataset == "index":
        display.display(data)

    else:
        if profile:
            _check_soft_dependencies(
                "fg-data-profiling",
                extra="analysis",
                severity="error",
            )
            from data_profiling import ProfileReport

            # create a copy for pandas profiler
            data_for_profiling = data.copy()

            pf = ProfileReport(data_for_profiling)
            display.display(pf)

        else:
            if verbose:
                display.display(data.head())

    return data
