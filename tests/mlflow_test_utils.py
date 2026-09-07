"""Helpers for checking what pycaret logged to mlflow."""

from mlflow.tracking import MlflowClient


def mlflow_run_tags(experiment_name):
    """Return the tags of each run in an mlflow experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the mlflow experiment.

    Returns
    -------
    list of dict
        Tags of each run as ``{name: value}``. mlflow stores tag values as
        strings.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    return [run.data.tags for run in client.search_runs(experiment.experiment_id)]
