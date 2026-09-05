from typing import List, Union, Optional, Tuple

import pandas as pd

from pycaret.internal.logging import get_logger
from pycaret.utils.time_series import TSAllowedPlotDataTypes

logger = get_logger()


# Data Types allowed for each plot type ----
# First one in the list is the default (if requested is None)
ALLOWED_PLOT_DATA_TYPES = {
    "pipeline": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.TRANSFORMED.value,
    ],
    "ts": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.TRANSFORMED.value,
    ],
    "train_test_split": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.TRANSFORMED.value,
    ],
    "cv": [TSAllowedPlotDataTypes.ORIGINAL.value],
    "acf": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "pacf": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "decomp": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "decomp_stl": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "diagnostics": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "diff": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "forecast": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
    ],
    "insample": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
    ],
    "residuals": [
        TSAllowedPlotDataTypes.ORIGINAL.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
    ],
    "periodogram": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "fft": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
    "ccf": [
        TSAllowedPlotDataTypes.TRANSFORMED.value,
        TSAllowedPlotDataTypes.IMPUTED.value,
        TSAllowedPlotDataTypes.ORIGINAL.value,
    ],
}


# Are multiple plot types allowed at once ----
MULTIPLE_PLOT_TYPES_ALLOWED_AT_ONCE = {
    "ts": True,
    "train_test_split": True,
    "cv": False,
    "acf": True,
    "pacf": True,
    "decomp": True,
    "decomp_stl": True,
    "diagnostics": True,
    "diff": False,
    "forecast": False,
    "insample": False,
    "residuals": False,
    "periodogram": True,
    "fft": True,
    "ccf": False,
}


def _reformat_dataframes_for_plots(
    data: List[Union[pd.Series, pd.DataFrame]], labels_suffix: List[str]
) -> List[pd.DataFrame]:
    """Take the input list of dataframes (assuming all dataframes have the same columns)
    and converts them into a list of new dataframes with each new dataframe containing
    the same column from all of the input dataframe.

    e.g. 1
    If input list has 2 dataframes D1 and D2 with columns A, B, and C then the
    output will be a list of 3 dataframes with 2 columns each
        Output dataframe 1 containing D1.A, D2.A
        Output dataframe 2 containing D1.B, D2.B
        Output dataframe 3 containing D1.C, D2.C

    e.g. 2
    If the input list has series, they are just concatenated together to produce one
    output dataframe.

    Parameters
    ----------
    data : List[Union[pd.Series, pd.DataFrame]]
        Input list of dataframes or series
    labels_suffix : List[str]
        The suffix to use for the output dataframes column names.
        Must be the same length as the number of input dataframes

        In the example above, if suffix is ["original", "transformed"], then the
            Output dataframe 1 will have columns ["A (original)", "A (transformed)"]
            Output dataframe 2 will have columns ["B (original)", "B (transformed)"]
            Output dataframe 2 will have columns ["C (original)", "C (transformed)"]

    Returns
    -------
    List[pd.DataFrame]
        Output list of dataframes

    Raises
    ------
    ValueError
        When the number of labels provided does not match the number of input dataframes
    """
    num_labels = len(labels_suffix)
    num_input_dfs = len(data)
    if num_labels != num_input_dfs:
        raise ValueError(
            f"Number of labels provided ({num_labels}) does not match the number of input "
            "dataframes ({num_input_dfs})"
        )

    cols = pd.DataFrame(data[0]).columns

    data = pd.concat(data, axis=1)
    output = []
    for col in cols:
        temp = pd.DataFrame(data[col])
        column_names = [f"{col} ({suffix})" for suffix in labels_suffix]
        temp.columns = column_names
        output.append(temp)

    return output


def _get_data_types_to_plot(
    plot: str, data_types_requested: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """Returns the data types to plot based on the requested ones. If all are allowed
    for the requested plot, they are returned as is, else this function will trim them
    down to the allowed types only.

    NOTE: Some plots only support one data type. If multiple data types are requested
    for such plots, only the first one is used (appropriate warning issued).

    Parameters
    ----------
    plot : str
        The plot for which the data types are being requested
    data_types_requested : Optional[Union[str, List[str]]], optional
        The data types being requested for the plot, by default None
        If None, it picks the default from the internal list.

    Returns
    -------
    List[str]
        The allowed data types for the requested plot based on user inputs

    Raises
    ------
    ValueError
        If none of the requested data types are supported by the plot
    """

    # Get default if not provided ----
    if data_types_requested is None:
        # First one is the default
        data_types_requested = [ALLOWED_PLOT_DATA_TYPES.get(plot)[0]]

    # Convert string to list ----
    if isinstance(data_types_requested, str):
        data_types_requested = [data_types_requested]

    # Is the data type allowed for the requested plot?
    all_plot_data_types = [member.value for member in TSAllowedPlotDataTypes]
    data_types_allowed = [
        (
            True
            if data_type_requested in ALLOWED_PLOT_DATA_TYPES.get(plot)
            and data_type_requested in all_plot_data_types
            else False
        )
        for data_type_requested in data_types_requested
    ]

    # Clean up list based on allowed data types
    cleaned_data_types = []
    for requested, allowed in zip(data_types_requested, data_types_allowed):
        if allowed:
            cleaned_data_types.append(requested)
        else:
            msg = (
                f"Data Type: '{requested}' is not supported for plot: '{plot}'. "
                "This will be ignored."
            )
            logger.warning(msg)
            print(msg)

    if len(cleaned_data_types) == 0:
        raise ValueError(
            "No data to plot. Please check to make sure that you have requested "
            "an allowed data type for plot."
            f"\n Allowed values are: {ALLOWED_PLOT_DATA_TYPES.get(plot)}"
        )

    if (
        not MULTIPLE_PLOT_TYPES_ALLOWED_AT_ONCE.get(plot)
        and len(cleaned_data_types) > 1
    ):
        msg = (
            f"Data Type requested for plot '{plot}' = {cleaned_data_types}, "
            "but this plot only supports a single data type at a time. "
            f"\nThe first one (i.e. '{cleaned_data_types[0]}') will be used."
        )
        logger.warning(msg)
        print(msg)
        cleaned_data_types = [cleaned_data_types[0]]

    return cleaned_data_types


def _clean_model_results_labels(
    model_results: List[pd.DataFrame], model_labels: List[str]
) -> Tuple[List[pd.DataFrame], List[str]]:
    """Cleans the model results and names to remove models that did not produce
    any results, e.g. no residuals, insample predictions, etc.

    Parameters
    ----------
    model_results : List[pd.DataFrame]
        List of dataframes containing the model results (one dataframe per model)
        Some values might be None if the model did not produce a result. These
        will get dropped by this function.
    model_labels : List[str]
        The labels of the models producing the results.

    Returns
    -------
    Tuple[List[pd.DataFrame], List[str]]
        The cleaned model results and names (after removing those that did not
        produce a result).
    """
    includes = [
        True if model_result is not None else False for model_result in model_results
    ]

    # Remove None results (produced when insample or residuals can not be obtained)
    model_results = [
        model_result
        for include, model_result in zip(includes, model_results)
        if include
    ]
    model_labels = [
        model_name for include, model_name in zip(includes, model_labels) if include
    ]

    return model_results, model_labels
