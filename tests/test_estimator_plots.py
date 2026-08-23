"""Unit tests for pycaret.internal.plots.estimator_plots.

The plot functions are called directly on small scikit-learn models, without
``setup``. The end-to-end ``plot_model`` paths are covered by
``test_classification_plots.py`` and ``test_regression_plots.py``.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs, make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from pycaret.internal.plots import estimator_plots

matplotlib.use("Agg")

pytestmark = [pytest.mark.plotting]


@pytest.fixture(scope="module")
def binary():
    """Return a fitted binary classifier with its train and test data."""
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LogisticRegression().fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def multiclass():
    """Return a fitted multiclass classifier with its train and test data."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def regression():
    """Return a fitted regressor with its train and test data."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


@pytest.fixture(scope="module")
def clustering():
    """Return a fitted clusterer with the data it was fitted on."""
    X, _ = make_blobs(n_samples=150, centers=3, random_state=0)
    model = KMeans(n_clusters=3, n_init=1, random_state=0).fit(X)
    return model, X


def assert_figure(fig):
    """Assert that ``fig`` is a matplotlib figure, then close it."""
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.parametrize(
    "plot",
    [
        estimator_plots.plot_roc_auc,
        estimator_plots.plot_precision_recall,
        estimator_plots.plot_confusion_matrix,
        estimator_plots.plot_class_prediction_error,
        estimator_plots.plot_classification_report,
    ],
)
@pytest.mark.parametrize("fixture", ["binary", "multiclass"])
def test_classifier_plots(plot, fixture, request):
    """Assert that the hold-out classifier plots return a figure."""
    model, _, _, X_test, y_test = request.getfixturevalue(fixture)
    assert_figure(plot(model, X_test, y_test))


def test_discrimination_threshold(binary, multiclass):
    """Assert that the threshold plot works for binary targets only."""
    model, X_train, y_train, _, _ = binary
    fig = estimator_plots.plot_discrimination_threshold(
        model, X_train, y_train, n_trials=3, random_state=0
    )
    assert_figure(fig)

    model, X_train, y_train, _, _ = multiclass
    with pytest.raises(TypeError, match="binary"):
        estimator_plots.plot_discrimination_threshold(model, X_train, y_train)


def test_decision_boundary(multiclass):
    """Assert that the decision boundary plot needs exactly two features."""
    model, X_train, y_train, X_test, y_test = multiclass
    fig = estimator_plots.plot_decision_boundary(
        model, X_train[:, :2], y_train, X_test[:, :2], y_test
    )
    assert_figure(fig)

    with pytest.raises(ValueError, match="two features"):
        estimator_plots.plot_decision_boundary(model, X_train, y_train, X_test, y_test)


def test_regressor_plots(regression):
    """Assert that the regressor plots return a figure."""
    model, X_train, y_train, X_test, y_test = regression
    assert_figure(
        estimator_plots.plot_residuals(model, X_train, y_train, X_test, y_test)
    )
    assert_figure(estimator_plots.plot_prediction_error(model, X_test, y_test))
    assert_figure(estimator_plots.plot_cooks_distance(X_train, y_train))


def test_model_selection_plots(regression):
    """Assert that the cross-validated curve plots return a figure."""
    model, X_train, y_train, _, _ = regression
    assert_figure(estimator_plots.plot_rfecv(model, X_train, y_train, cv=2))
    assert_figure(estimator_plots.plot_learning_curve(model, X_train, y_train, cv=2))
    assert_figure(
        estimator_plots.plot_validation_curve(
            model, X_train, y_train, "fit_intercept", [True, False], cv=2
        )
    )


def test_feature_plots(multiclass, regression):
    """Assert that the manifold and RadViz plots handle discrete and continuous targets."""
    _, X_train, y_train, _, _ = multiclass
    assert_figure(
        estimator_plots.plot_manifold(X_train, y_train, random_state=0, perplexity=5)
    )
    assert_figure(estimator_plots.plot_radviz(X_train, y_train))

    _, X_train, y_train, _, _ = regression
    assert_figure(
        estimator_plots.plot_manifold(X_train, y_train, random_state=0, perplexity=5)
    )


def test_clustering_plots(clustering):
    """Assert that the clustering plots return a figure and reject unsuitable models."""
    model, X = clustering
    assert_figure(estimator_plots.plot_elbow(model, X, k_range=range(2, 5)))
    assert_figure(estimator_plots.plot_silhouette(model, X))
    assert_figure(estimator_plots.plot_intercluster_distance(model, X, random_state=0))

    with pytest.raises(TypeError, match="n_clusters"):
        estimator_plots.plot_elbow(LinearRegression(), X)


def test_style_does_not_leak(binary):
    """Assert that plotting leaves the global matplotlib configuration unchanged."""
    model, _, _, X_test, y_test = binary
    before = {key: matplotlib.rcParams[key] for key in estimator_plots.PLOT_STYLE}
    plt.close(estimator_plots.plot_roc_auc(model, X_test, y_test))
    after = {key: matplotlib.rcParams[key] for key in estimator_plots.PLOT_STYLE}
    assert after == before


def test_show_matplotlib_figure(tmp_path):
    """Assert that the renderer saves the figure, closes it, and returns the file name."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    filename = estimator_plots.show_matplotlib_figure(
        fig, "My Plot", save=str(tmp_path)
    )
    assert filename == str(tmp_path / "My Plot.png")
    assert (tmp_path / "My Plot.png").exists()
    assert not plt.fignum_exists(fig.number)

    fig, _ = plt.subplots()
    filename = estimator_plots.show_matplotlib_figure(fig, "Other", system=False)
    assert filename == "Other.png"
    assert not plt.fignum_exists(fig.number)


def test_elbow_index():
    """Assert that the elbow is the point farthest from the chord of the curve."""
    assert estimator_plots._elbow_index([1, 2, 3, 4, 5], [100, 20, 10, 8, 7]) == 1
    assert estimator_plots._elbow_index([1, 2, 3], [3, 2, 1]) == 0
    assert estimator_plots._elbow_index([1], [5]) == 0


def test_class_scores_falls_back_to_decision_function():
    """Assert that class scores come from decision_function when there is no predict_proba."""

    class Scorer:
        classes_ = np.array([0, 1])

        def decision_function(self, X):
            return np.zeros(len(X))

    scores = estimator_plots._class_scores(Scorer(), np.zeros((4, 2)))
    assert scores.shape == (4, 2)

    with pytest.raises(TypeError, match="predict_proba"):
        estimator_plots._class_scores(object(), np.zeros((4, 2)))
