# Module: containers.models.clustering
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>

# The purpose of this module is to serve as a central repository of clustering models. The `clustering` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `ClassifierContainer`
# as a base (but not `ClassifierContainer` itself). In order to add a new model, you only need to create a new class that has
# `ClassifierContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

from typing import Any, Dict, List, Optional

import numpy as np
from skbase.utils.dependencies import _safe_import

import pycaret.containers.base_container
from pycaret.containers.models.base_model import ModelContainer

# from pycaret.internal.cuml_wrappers import get_dbscan, get_kmeans
from pycaret.utils._dependencies import _check_soft_dependencies
from pycaret.utils.generic import get_logger, param_grid_to_lists

_DEFAULT_N_CLUSTERS = 4

# First one in the list is the default ----
ALL_ALLOWED_ENGINES: Dict[str, List[str]] = {
    "kmeans": ["sklearn", "sklearnex"],
    "dbscan": ["sklearn", "sklearnex"],
}


def get_container_default_engines() -> Dict[str, str]:
    """Get the default engines from all models
    Returns
    -------
    Dict[str, str]
        Default engines for all containers. If unspecified, it is not included
        in the return dictionary.
    """
    default_engines = {}
    for id, all_engines in ALL_ALLOWED_ENGINES.items():
        default_engines[id] = all_engines[0]
    return default_engines


class ClusterContainer(ModelContainer):
    """
    Base clustering model container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    eq_function : type, default = None
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    is_special : bool, default = False
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    is_gpu_enabled : bool, default = None
        If None, will try to automatically determine.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    eq_function : type
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict, default = {} (empty dict)
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool, default = False
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list, default = {} (empty dict)
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution, default = {} (empty dict)
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict, default = {} (empty dict)
        The arguments to always pass to the tuner.
    is_gpu_enabled : bool
        If None, will try to automatically determine.
    """

    def __init__(
        self,
        id: str,
        name: str,
        eq_function: Optional[type] = None,
        is_special: bool = False,
        is_gpu_enabled: Optional[bool] = None,
    ) -> None:

        super().__init__(
            id=id,
            name=name,
            eq_function=eq_function,
            is_special=is_special,
        )

        if is_gpu_enabled is not None:
            self.is_gpu_enabled = is_gpu_enabled
        else:
            self.is_gpu_enabled = bool(self.get_package_name() == "cuml")

    @property
    def args(self):
        return self._args()

    def _args(self):
        return {}

    @property
    def tune_grid(self):
        return param_grid_to_lists(self._tune_grid())

    def _tune_grid(self):
        return {}

    @property
    def tune_distribution(self):
        return self._tune_distribution()

    def _tune_distribution(self):
        return {}

    @property
    def tune_args(self):
        return self._tune_args()

    def _tune_args(self):
        return {}

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.

        Parameters
        ----------
        internal : bool, default = True
            If True, will return all properties. If False, will only
            return properties intended for the user to see.

        Returns
        -------
        dict of str : Any

        """
        d = [
            ("ID", self.id),
            ("Name", self.name),
            ("Reference", self.reference),
        ]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Equality", self.eq_function),
                ("Args", self.args),
                ("Tune Grid", self.tune_grid),
                ("Tune Distributions", self.tune_distribution),
                ("Tune Args", self.tune_args),
                ("GPU Enabled", self.is_gpu_enabled),
            ]

        return dict(d)

    def _class_def(self):
        pth, pkg_name = self._get_cls_path()
        return _safe_import(pth, pkg_name=pkg_name)

    def _get_cls_path(self):
        pth = self.get_tag("cls_path")
        pkg_name = pth.split(".")[0]
        if pkg_name == "sklearn":
            pkg_name = "scikit-learn"
        return pth, pkg_name


class _SklearnMixin:

    def _get_cls_path(self):
        pth = self.get_tag("cls_path")
        pkg_name = "scikit-learn"
        check_sd = False

        if not hasattr(self, "engine"):
            return pth, pkg_name

        if self.engine == "sklearnex":
            pth = pth.replace("sklearn.", "sklearnex.", 1)
            pkg_name = "scikit-learn-intelex"
            check_sd = True

        if self.experiment.gpu_param == "force" or self.experiment.gpu_param:
            pth = pth.replace("sklearn.", "cuml.", 1)
            pkg_name = "cuml"
            if self.experiment.gpu_param:
                check_sd = True

        if check_sd:
            _check_soft_dependencies(pkg_name, extra=None, severity="warning")

        return pth, pkg_name


class KMeansClusterContainer(_SklearnMixin, ClusterContainer):

    _tags = {"cls_path": "sklearn.cluster.KMeans"}

    def __init__(self, experiment):
        self.experiment = experiment

        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "kmeans"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if experiment.gpu_param == "force" or experiment.gpu_param:
            if experiment.gpu_param == "force":
                severity = "error"
            else:
                severity = "warning"

            _check_soft_dependencies("cuml", extra=None, severity=severity)

            logger.info("Imported cuml.cluster.KMeans")
            gpu_imported = True

        # if gpu_imported:
        #     KMeans = get_kmeans()

        super().__init__(
            id=id,
            name="K-Means Clustering",
            is_gpu_enabled=gpu_imported,
        )

    def _args(self):
        return {
            "n_clusters": _DEFAULT_N_CLUSTERS,
            "random_state": self.experiment.seed,
        }


class AffinityPropagationClusterContainer(_SklearnMixin, ClusterContainer):

    _tags = {"cls_path": "sklearn.cluster.AffinityPropagation"}

    def __init__(self, experiment):
        self.experiment = experiment

        get_logger()
        np.random.seed(experiment.seed)

        super().__init__(
            id="ap",
            name="Affinity Propagation",
        )


class MeanShiftClusterContainer(_SklearnMixin, ClusterContainer):

    _tags = {"cls_path": "sklearn.cluster.MeanShift"}

    def __init__(self, experiment):
        self.experiment = experiment

        get_logger()
        np.random.seed(experiment.seed)

        super().__init__(
            id="meanshift",
            name="Mean Shift Clustering",
        )

    def _args(self):
        return {"n_jobs": self.experiment.n_jobs_param}


class SpectralClusteringClusterContainer(_SklearnMixin, ClusterContainer):

    _tags = {"cls_path": "sklearn.cluster.SpectralClustering"}

    def __init__(self, experiment):
        self.experiment = experiment

        get_logger()
        np.random.seed(experiment.seed)

        super().__init__(
            id="sc",
            name="Spectral Clustering",
        )

    def _args(self):
        return {
            "n_clusters": _DEFAULT_N_CLUSTERS,
            "random_state": self.experiment.seed,
            "n_jobs": self.experiment.n_jobs_param,
        }


class AgglomerativeClusteringClusterContainer(_SklearnMixin, ClusterContainer):

    _tags = {"cls_path": "sklearn.cluster.AgglomerativeClustering"}

    def __init__(self, experiment):
        self.experiment = experiment

        get_logger()
        np.random.seed(experiment.seed)

        super().__init__(
            id="hclust",
            name="Agglomerative Clustering",
        )

    def _args(self):
        return {"n_clusters": _DEFAULT_N_CLUSTERS}


class DBSCANClusterContainer(_SklearnMixin, ClusterContainer):

    _tags = {"cls_path": "sklearn.cluster.DBSCAN"}

    def __init__(self, experiment):
        self.experiment = experiment

        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        id = "dbscan"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if experiment.gpu_param == "force" or experiment.gpu_param:
            if experiment.gpu_param == "force":
                severity = "error"
            else:
                severity = "warning"

            _check_soft_dependencies("cuml", extra=None, severity=severity)

            logger.info("Imported cuml.cluster.KMeans")
            gpu_imported = True

        # if not gpu_imported:
        #     args["n_jobs"] = experiment.n_jobs_param
        # else:
        #     DBSCAN = get_dbscan()

        super().__init__(
            id=id,
            name="Density-Based Spatial Clustering",
            is_gpu_enabled=gpu_imported,
        )

    def _args(self):
        if self.is_gpu_enabled:
            return {}
        else:
            return {"n_jobs": self.experiment.n_jobs_param}


class OPTICSClusterContainer(_SklearnMixin, ClusterContainer):

    _tags = {"cls_path": "sklearn.cluster.OPTICS"}

    def __init__(self, experiment):
        self.experiment = experiment

        get_logger()
        np.random.seed(experiment.seed)

        super().__init__(
            id="optics",
            name="OPTICS Clustering",
        )

    def _args(self):
        return {"n_jobs": self.experiment.n_jobs_param}


class BirchClusterContainer(_SklearnMixin, ClusterContainer):

    _tags = {"cls_path": "sklearn.cluster.Birch"}

    def __init__(self, experiment):
        self.experiment = experiment

        get_logger()
        np.random.seed(experiment.seed)

        super().__init__(
            id="birch",
            name="Birch Clustering",
        )

    def _args(self):
        return {"n_clusters": _DEFAULT_N_CLUSTERS}


class KModesClusterContainer(ClusterContainer):

    _tags = {"cls_path": "kmodes.kmodes.KModes"}

    def __init__(self, experiment):
        self.experiment = experiment

        get_logger()
        np.random.seed(experiment.seed)

        if not _check_soft_dependencies("kmodes", extra="models", severity="warning"):
            self.active = False
            return

        super().__init__(
            id="kmodes",
            name="K-Modes Clustering",
        )

    def _args(self):
        return {
            "n_clusters": _DEFAULT_N_CLUSTERS,
            "random_state": self.experiment.seed,
            "n_jobs": self.experiment.n_jobs_param,
        }


def get_all_model_containers(
    experiment: Any, raise_errors: bool = True
) -> Dict[str, ClusterContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), experiment, ClusterContainer, raise_errors
    )
