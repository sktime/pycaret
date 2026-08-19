"""Model plots built on matplotlib and scikit-learn.

These functions replace the ``yellowbrick`` visualizers used by pycaret up to
version 3.5.x, see https://github.com/sktime/pycaret/issues/49.

Each ``plot_*`` function builds and returns a ``matplotlib.figure.Figure``.
``show_matplotlib_figure`` saves, shows, or streams that figure and returns the
plot file name, as ``pycaret.internal.pycaret_experiment`` expects.
"""

import functools
import os
import time
from copy import deepcopy
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from cycler import cycler
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from packaging import version
from sklearn.feature_selection import RFECV
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    r2_score,
    roc_curve,
    silhouette_samples,
)
from sklearn.model_selection import (
    LearningCurveDisplay,
    ShuffleSplit,
    ValidationCurveDisplay,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target

from pycaret.internal.logging import get_logger

# Default look of all plots in this module. The values mirror the style and
# context that yellowbrick applied, so that plots look as they did before
# 3.6.0. They are applied per plot through ``_styled`` and never leak into the
# caller's matplotlib configuration.
PLOT_STYLE = {
    # yellowbrick's color cycle
    "axes.prop_cycle": cycler(
        color=["#0272a2", "#9fc377", "#ca0b03", "#a50258", "#d7c703", "#88cada"]
    ),
    # figure and font sizes
    "figure.figsize": (8, 5.5),
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 1,
    # lines and markers
    "lines.linewidth": 1.75,
    "lines.markersize": 7,
    # white axes with a light gray frame and grid, no tick marks
    "axes.facecolor": "white",
    "axes.edgecolor": ".8",
    "axes.linewidth": 1.25,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": ".8",
    "grid.linestyle": "-",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
}


def _styled(plot):
    """Decorator running a plot function with ``PLOT_STYLE`` applied."""

    @functools.wraps(plot)
    def wrapper(*args, **kwargs):
        with plt.rc_context(PLOT_STYLE):
            return plot(*args, **kwargs)

    return wrapper


def _colors(n: int) -> list:
    """Return ``n`` colors cycling through the default color cycle."""
    palette = PLOT_STYLE["axes.prop_cycle"].by_key()["color"]
    return [palette[i % len(palette)] for i in range(n)]


def show_matplotlib_figure(
    fig: Figure,
    name: str,
    scale: float = 1,
    save: Union[str, bool] = False,
    display_format: Optional[str] = None,
    system: bool = True,
) -> str:
    """Save, show, or stream a matplotlib figure and return its file name.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to render. It is closed afterwards.
    name : str
        Base name of the plot; the file is saved as ``"{name}.png"``.
    scale : float, default 1
        Multiplier applied to the figure DPI.
    save : bool or str, default False
        If True, save the figure in the current directory. If a string, save it
        in that directory. If False, display it instead.
    display_format : str, optional
        ``"streamlit"`` renders the figure with ``streamlit.write``, anything
        else with ``matplotlib.pyplot.show``.
    system : bool, default True
        Whether to display the figure when it is not saved.

    Returns
    -------
    str
        The plot file name, used by callers for logging artifacts.
    """
    logger = get_logger()
    fig.set_dpi(fig.dpi * scale)

    plot_filename = f"{name}.png"
    if save:
        if not isinstance(save, bool):
            plot_filename = os.path.join(save, plot_filename)
        logger.info(f"Saving '{plot_filename}'")
        fig.savefig(plot_filename, bbox_inches="tight")
    elif display_format == "streamlit":
        import streamlit as st

        st.write(fig)
    elif system:
        plt.show()

    plt.close(fig)
    logger.info("Visual Rendered Successfully")
    return plot_filename


def _class_scores(estimator, X) -> np.ndarray:
    """Return per-class scores of shape (n_samples, n_classes)."""
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores
    raise TypeError(
        f"{type(estimator).__name__} has neither predict_proba nor "
        "decision_function, which this plot requires."
    )


def _label_curve_display(display, xlabel: str) -> None:
    """Label a learning or validation curve display the way yellowbrick did."""
    for line, label in zip(
        display.lines_, ("Training Score", "Cross Validation Score")
    ):
        line.set_label(label)
    display.ax_.set_xlabel(xlabel)
    display.ax_.set_ylabel("Score")
    display.ax_.legend(loc="best")


def _fit_params(fit_kwargs: Optional[dict]) -> dict:
    """Keyword argument forwarding ``fit_kwargs`` to scikit-learn's curve displays.

    ``fit_params`` was renamed to ``params`` in scikit-learn 1.6.
    """
    if not fit_kwargs:
        return {}
    if version.parse(sklearn.__version__) >= version.parse("1.6"):
        return {"params": fit_kwargs}
    return {"fit_params": fit_kwargs}


@_styled
def plot_roc_auc(estimator, X, y, **kwargs) -> Figure:
    """Plot ROC curves of a fitted classifier.

    One-vs-rest curves are drawn per class together with the micro and macro
    averages, as yellowbrick's ``ROCAUC`` did.

    Parameters
    ----------
    estimator : fitted classifier
    X, y : hold-out features and target
    **kwargs
        Line properties applied to every ROC curve.

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = np.asarray(estimator.classes_)
    y_bin = label_binarize(np.asarray(y), classes=classes)
    if y_bin.shape[1] == 1:  # label_binarize returns one column for binary y
        y_bin = np.column_stack([1 - y_bin, y_bin])
    y_score = _class_scores(estimator, X)

    fig, ax = plt.subplots()
    for i, label in enumerate(classes):
        display = RocCurveDisplay.from_predictions(
            y_bin[:, i], y_score[:, i], name=f"ROC of class {label}", ax=ax
        )
        display.line_.set(**kwargs)
    display = RocCurveDisplay.from_predictions(
        y_bin.ravel(), y_score.ravel(), name="micro-average ROC curve", ax=ax
    )
    display.line_.set(linestyle="--", **kwargs)
    curves = [roc_curve(y_bin[:, i], y_score[:, i])[:2] for i in range(len(classes))]
    fpr = np.unique(np.concatenate([c[0] for c in curves]))
    tpr = np.mean([np.interp(fpr, *c) for c in curves], axis=0)
    ax.plot(
        fpr,
        tpr,
        linestyle="--",
        label=f"macro-average ROC curve (AUC = {auc(fpr, tpr):0.2f})",
    )

    ax.plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves for {type(estimator).__name__}")
    ax.legend(loc="lower right")
    return fig


@_styled
def plot_precision_recall(
    estimator,
    X,
    y,
    per_class: bool = False,
    **kwargs,
) -> Figure:
    """Plot the precision-recall curve of a fitted classifier.

    Binary problems show the curve of the positive class, multiclass problems
    the micro-average curve, both with the area under the curve filled and the
    average precision marked, as yellowbrick's ``PrecisionRecallCurve`` did.

    Parameters
    ----------
    estimator : fitted classifier
    X, y : hold-out features and target
    per_class : bool, default False
        For multiclass problems, draw one curve per class instead of the
        micro average.
    **kwargs
        Line properties applied to every precision-recall curve.

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = np.asarray(estimator.classes_)
    y_bin = label_binarize(np.asarray(y), classes=classes)
    y_score = _class_scores(estimator, X)

    fig, ax = plt.subplots()
    if len(classes) == 2:
        curves = [(y_bin[:, 0], y_score[:, 1], "Binary PR curve")]
    elif per_class:
        curves = [
            (y_bin[:, i], y_score[:, i], f"PR for class {label}")
            for i, label in enumerate(classes)
        ]
    else:
        curves = [(y_bin.ravel(), y_score.ravel(), "Micro-average PR for all classes")]

    for y_true, score, name in curves:
        display = PrecisionRecallDisplay.from_predictions(
            y_true, score, name=name, ax=ax
        )
        display.line_.set(**kwargs)
        ax.fill_between(
            display.recall,
            display.precision,
            step="post",
            alpha=0.2,
            color=display.line_.get_color(),
        )
        if len(curves) == 1:
            ax.axhline(
                display.average_precision,
                linestyle="--",
                color="#ca0b03",
                label=f"Avg. precision={display.average_precision:0.2f}",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve for {type(estimator).__name__}")
    ax.legend(loc="lower left")
    return fig


@_styled
def plot_confusion_matrix(
    estimator,
    X,
    y,
    fontsize: int = 15,
    cmap: str = "Greens",
    **kwargs,
) -> Figure:
    """Plot the confusion matrix of a fitted classifier.

    Parameters
    ----------
    estimator : fitted classifier
    X, y : hold-out features and target
    fontsize : int, default 15
        Font size of the cell counts.
    cmap : str, default "Greens"
        Colormap of the cells.
    **kwargs
        Passed to ``ConfusionMatrixDisplay.from_estimator``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(
        estimator,
        X,
        y,
        cmap=cmap,
        colorbar=False,
        text_kw={"fontsize": fontsize},
        ax=ax,
        **kwargs,
    )
    ax.grid(False)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(f"{type(estimator).__name__} Confusion Matrix")
    return fig


@_styled
def plot_discrimination_threshold(
    estimator,
    X,
    y,
    n_trials: int = 50,
    test_size: float = 0.1,
    random_state=None,
    fit_kwargs: Optional[dict] = None,
    **kwargs,
) -> Figure:
    """Plot precision, recall, F1 and queue rate against the decision threshold.

    As yellowbrick's ``DiscriminationThreshold`` did, the data is split
    ``n_trials`` times with ``ShuffleSplit``, a copy of the estimator is fitted
    on each training part and scored on the test part, and the median of every
    metric is drawn with its 10 %-90 % quantile band. The threshold maximizing
    the median F1 is marked. Binary classification only.

    Parameters
    ----------
    estimator : binary classifier
    X, y : features and target to split
    n_trials : int, default 50
        Number of shuffle splits.
    test_size : float, default 0.1
        Fraction of samples scored in every trial.
    random_state : int, optional
        Seed of the shuffle splits.
    fit_kwargs : dict, optional
        Passed to ``estimator.fit``.
    **kwargs
        Passed to ``Axes.plot``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = np.asarray(estimator.classes_)
    if len(classes) != 2:
        raise TypeError(
            "The threshold plot is only available for binary classification."
        )
    X = pd.DataFrame(X)
    y = (np.asarray(y) == classes[1]).astype(int)
    thresholds = np.linspace(0, 1, 101)

    trials = {"precision": [], "recall": [], "f1": [], "queue rate": []}
    splitter = ShuffleSplit(
        n_splits=n_trials, test_size=test_size, random_state=random_state
    )
    for train, test in splitter.split(X):
        model = deepcopy(estimator).fit(X.iloc[train], y[train], **(fit_kwargs or {}))
        score = _class_scores(model, X.iloc[test])[:, 1]
        if not hasattr(model, "predict_proba"):
            score = (score - score.min()) / ((score.max() - score.min()) or 1)
        predicted = score[None, :] >= thresholds[:, None]
        n_predicted = predicted.sum(axis=1)
        true_positives = (predicted & (y[test] == 1)).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(n_predicted > 0, true_positives / n_predicted, 1.0)
            recall = true_positives / max(y[test].sum(), 1)
            f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
        trials["precision"].append(precision)
        trials["recall"].append(recall)
        trials["f1"].append(f1)
        trials["queue rate"].append(n_predicted / len(test))

    fig, ax = plt.subplots()
    medians = {}
    for label, values in trials.items():
        lower, medians[label], upper = np.quantile(values, [0.1, 0.5, 0.9], axis=0)
        (line,) = ax.plot(thresholds, medians[label], label=label, **kwargs)
        ax.fill_between(thresholds, lower, upper, alpha=0.35, color=line.get_color())
    # the largest threshold reaching the best median F1
    best = thresholds[
        np.flatnonzero(np.isclose(medians["f1"], medians["f1"].max()))[-1]
    ]
    ax.axvline(
        best,
        linestyle="--",
        color="black",
        linewidth=1,
        label=f"$t_{{f1}} = {best:0.2f}$",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("discrimination threshold")
    ax.set_ylabel("score")
    ax.set_title(f"Threshold Plot for {type(estimator).__name__}")
    ax.legend(loc="best")
    return fig


@_styled
def plot_class_prediction_error(estimator, X, y, **kwargs) -> Figure:
    """Plot stacked bars of predicted classes for every actual class.

    Parameters
    ----------
    estimator : fitted classifier
    X, y : hold-out features and target
    **kwargs
        Passed to ``Axes.bar``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = np.asarray(estimator.classes_)
    labels = [str(label) for label in estimator.classes_]
    matrix = confusion_matrix(np.asarray(y), estimator.predict(X), labels=classes)

    fig, ax = plt.subplots()
    positions = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for j, label in enumerate(labels):
        ax.bar(positions, matrix[:, j], bottom=bottom, label=label, **kwargs)
        bottom += matrix[:, j]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("actual class")
    ax.set_ylabel("number of predicted class")
    ax.set_title(f"Class Prediction Error for {type(estimator).__name__}")
    ax.set_ylim(0, 1.1 * bottom.max())
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    return fig


@_styled
def plot_classification_report(
    estimator,
    X,
    y,
    cmap: str = "YlOrRd",
    **kwargs,
) -> Figure:
    """Plot a heatmap of per-class precision, recall, F1 and support.

    Parameters
    ----------
    estimator : fitted classifier
    X, y : hold-out features and target
    cmap : str, default "YlOrRd"
        Colormap of the cells. Support is colored by its share of all samples.
    **kwargs
        Passed to ``Axes.imshow``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = np.asarray(estimator.classes_)
    labels = [str(label) for label in estimator.classes_]
    precision, recall, f1, support = precision_recall_fscore_support(
        np.asarray(y), estimator.predict(X), labels=classes, zero_division=0
    )
    support = np.asarray(support, dtype=float)
    cells = np.column_stack([precision, recall, f1, support / (support.sum() or 1)])
    colormap = plt.get_cmap(cmap)

    fig, ax = plt.subplots()
    image = ax.imshow(cells, cmap=colormap, vmin=0, vmax=1, aspect="auto", **kwargs)
    fig.colorbar(image, ax=ax)
    for i in range(len(classes)):
        for j in range(4):
            text = f"{support[i]:.0f}" if j == 3 else f"{cells[i, j]:0.3f}"
            r, g, b, _ = colormap(cells[i, j])
            dark_cell = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if dark_cell else "black",
            )
    ax.set_xticks(range(4))
    ax.set_xticklabels(["precision", "recall", "f1", "support"])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title(f"{type(estimator).__name__} Classification Report")
    return fig


@_styled
def plot_decision_boundary(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    fit_kwargs: Optional[dict] = None,
    cmap=None,
    **kwargs,
) -> Figure:
    """Plot the decision boundary of a classifier refitted on two features.

    A copy of ``estimator`` is fitted on ``X_train``, which must have exactly
    two columns. The boundary is drawn with ``DecisionBoundaryDisplay`` and the
    hold-out points are scattered on top.

    Parameters
    ----------
    estimator : classifier
    X_train, y_train : two-column training features and target
    X_test, y_test : two-column hold-out features and target
    fit_kwargs : dict, optional
        Passed to ``estimator.fit``.
    cmap : str or Colormap, optional
        Colormap shared by the decision regions and the hold-out points. By
        default, the module's color cycle is used.
    **kwargs
        Passed to ``DecisionBoundaryDisplay.from_estimator``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    if X_train.shape[1] != 2:
        raise ValueError("The decision boundary plot needs exactly two features.")

    estimator = deepcopy(estimator).fit(X_train, y_train, **(fit_kwargs or {}))
    n_classes = len(estimator.classes_)
    if cmap is None:
        colormap = ListedColormap(_colors(n_classes))
    else:
        colormap = plt.get_cmap(cmap, n_classes)

    fig, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X_train,
        response_method="predict",
        cmap=colormap,
        # one contour level per class, so that region i gets colormap(i)
        levels=np.arange(n_classes + 1) - 0.5,
        alpha=0.8,
        xlabel="Feature One",
        ylabel="Feature Two",
        ax=ax,
        **kwargs,
    )
    labels = np.asarray(estimator.classes_)
    for i, (cls, label) in enumerate(zip(estimator.classes_, labels)):
        points = X_test[y_test == cls]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=20,
            color=colormap(i),
            edgecolor="black",
            linewidth=0.5,
            label=str(label),
        )
    ax.set_title(f"Decision Boundary for {type(estimator).__name__}")
    ax.legend(loc="best")
    return fig


@_styled
def plot_residuals(estimator, X_train, y_train, X_test, y_test, **kwargs) -> Figure:
    """Plot residuals against predictions for the train and hold-out sets.

    Residuals are ``y_pred - y_true``, following yellowbrick's convention. A
    histogram of the residuals is drawn next to the scatter plot.

    Parameters
    ----------
    estimator : fitted regressor
    X_train, y_train : training features and target
    X_test, y_test : hold-out features and target
    **kwargs
        Passed to ``Axes.scatter``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax, ax_hist) = plt.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05}
    )
    for (X, y), label in (((X_train, y_train), "Train"), ((X_test, y_test), "Test")):
        y = np.asarray(y, dtype=float)
        y_pred = np.asarray(estimator.predict(X), dtype=float)
        residuals = y_pred - y
        points = ax.scatter(
            y_pred,
            residuals,
            alpha=0.5,
            label=f"{label} $R^2 = {r2_score(y, y_pred):0.3f}$",
            **kwargs,
        )
        ax_hist.hist(
            residuals,
            bins=50,
            orientation="horizontal",
            alpha=0.5,
            color=points.get_facecolor()[0],
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals for {type(estimator).__name__}")
    ax.legend(loc="best")
    ax_hist.set_xlabel("Distribution")
    ax_hist.tick_params(axis="y", left=False)
    return fig


@_styled
def plot_prediction_error(estimator, X, y, **kwargs) -> Figure:
    """Plot predicted against actual values of a fitted regressor.

    The identity line, the least-squares fit of the points and the R² score
    are shown for reference.

    Parameters
    ----------
    estimator : fitted regressor
    X, y : hold-out features and target
    **kwargs
        Passed to ``Axes.scatter``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y, dtype=float)
    y_pred = np.asarray(estimator.predict(X), dtype=float)
    limits = np.array(
        [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    )
    slope, intercept = np.polyfit(y_true, y_pred, 1)

    fig, ax = plt.subplots()
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.75,
        label=f"$R^2 = {r2_score(y_true, y_pred):0.3f}$",
        **kwargs,
    )
    ax.plot(
        limits,
        slope * limits + intercept,
        linestyle="--",
        color="black",
        label="best fit",
    )
    ax.plot(limits, limits, linestyle="--", color="grey", label="identity")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$\\hat{y}$")
    ax.set_title(f"Prediction Error for {type(estimator).__name__}")
    ax.legend(loc="upper left")
    return fig


@_styled
def plot_cooks_distance(X, y, **kwargs) -> Figure:
    """Plot Cook's distance of every observation under an ordinary least squares fit.

    Observations above the ``4 / n`` influence threshold are counted in the
    legend.

    Parameters
    ----------
    X, y : numeric training features and target
    **kwargs
        Passed to ``Axes.stem``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    residuals = y - LinearRegression().fit(X, y).predict(X)
    # leverage is the diagonal of the hat matrix of X with an intercept column
    design = np.column_stack([np.ones(n_samples), X])
    leverage = (design * np.linalg.pinv(design).T).sum(axis=1)
    mse = residuals @ residuals / (n_samples - np.linalg.matrix_rank(design))
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = residuals / np.sqrt(mse * (1 - leverage))
        distance = studentized**2 / (n_features + 1) * leverage / (1 - leverage)
    distance = np.nan_to_num(distance, nan=0.0, posinf=0.0)

    threshold = 4 / n_samples
    outlier_percentage = 100 * np.mean(distance > threshold)

    fig, ax = plt.subplots()
    ax.stem(distance, linefmt="C0-", markerfmt=",", basefmt="k-", **kwargs)
    ax.axhline(
        threshold,
        linestyle="--",
        color="tab:red",
        linewidth=1,
        label=f"{outlier_percentage:0.2f}% > $I_t$ ($I_t$ = 4/n)",
    )
    ax.set_xlabel("instance index")
    ax.set_ylabel("influence (I)")
    ax.set_title("Cook's Distance Outlier Detection")
    ax.legend(loc="best")
    return fig


@_styled
def plot_rfecv(estimator, X, y, cv=None, groups=None, **kwargs) -> Figure:
    """Plot the cross-validated score against the number of selected features.

    Parameters
    ----------
    estimator : estimator exposing ``coef_`` or ``feature_importances_``
    X, y : training features and target
    cv : cross-validation splitter, optional
    groups : array-like, optional
        Group labels for ``GroupKFold``-style splitters.
    **kwargs
        Passed to ``sklearn.feature_selection.RFECV``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rfecv = RFECV(estimator, cv=cv, **kwargs).fit(X, y, groups=groups)
    results = rfecv.cv_results_
    mean, std = results["mean_test_score"], results["std_test_score"]
    if "n_features" in results:  # scikit-learn >= 1.5
        n_features = np.asarray(results["n_features"])
    else:
        n_features = rfecv.min_features_to_select + rfecv.step * np.arange(len(mean))

    fig, ax = plt.subplots()
    ax.plot(n_features, mean, marker="o", label="mean test score")
    ax.fill_between(n_features, mean - std, mean + std, alpha=0.25)
    ax.axvline(
        rfecv.n_features_,
        linestyle="--",
        color="black",
        linewidth=1,
        label=f"n_features = {rfecv.n_features_}, score = {mean.max():0.3f}",
    )
    ax.set_xlabel("Number of Features Selected")
    ax.set_ylabel("Score")
    ax.set_title(f"RFECV for {type(estimator).__name__}")
    ax.legend(loc="best")
    return fig


@_styled
def plot_learning_curve(
    estimator,
    X,
    y,
    cv=None,
    groups=None,
    n_jobs: Optional[int] = None,
    fit_kwargs: Optional[dict] = None,
    **kwargs,
) -> Figure:
    """Plot training and cross-validation scores against the training set size.

    Parameters
    ----------
    estimator : estimator
    X, y : training features and target
    cv : cross-validation splitter, optional
    groups : array-like, optional
        Group labels for ``GroupKFold``-style splitters.
    n_jobs : int, optional
        Number of parallel jobs.
    fit_kwargs : dict, optional
        Passed to ``estimator.fit``.
    **kwargs
        Passed to ``LearningCurveDisplay.from_estimator``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    display = LearningCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        train_sizes=np.linspace(0.3, 1.0, 10),
        cv=cv,
        groups=groups,
        n_jobs=n_jobs,
        score_type="both",
        line_kw={"marker": "o"},
        ax=ax,
        **_fit_params(fit_kwargs),
        **kwargs,
    )
    _label_curve_display(display, "Training Instances")
    ax.set_title(f"Learning Curve for {type(estimator).__name__}")
    return fig


@_styled
def plot_validation_curve(
    estimator,
    X,
    y,
    param_name: str,
    param_range,
    cv=None,
    groups=None,
    n_jobs: Optional[int] = None,
    fit_kwargs: Optional[dict] = None,
    **kwargs,
) -> Figure:
    """Plot training and cross-validation scores against one hyperparameter.

    Parameters
    ----------
    estimator : estimator
    X, y : training features and target
    param_name : str
        Name of the hyperparameter to vary.
    param_range : array-like
        Values of the hyperparameter to evaluate.
    cv : cross-validation splitter, optional
    groups : array-like, optional
        Group labels for ``GroupKFold``-style splitters.
    n_jobs : int, optional
        Number of parallel jobs.
    fit_kwargs : dict, optional
        Passed to ``estimator.fit``.
    **kwargs
        Passed to ``ValidationCurveDisplay.from_estimator``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    display = ValidationCurveDisplay.from_estimator(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        groups=groups,
        n_jobs=n_jobs,
        score_type="both",
        line_kw={"marker": "d"},
        ax=ax,
        **_fit_params(fit_kwargs),
        **kwargs,
    )
    _label_curve_display(display, param_name)
    ax.set_title(f"Validation Curve for {type(estimator).__name__}")
    return fig


@_styled
def plot_manifold(X, y, random_state=None, **kwargs) -> Figure:
    """Plot a two-dimensional t-SNE embedding of ``X`` colored by the target.

    Discrete targets get one color per class, continuous targets a colorbar.

    Parameters
    ----------
    X, y : numeric features and target
    random_state : int, optional
        Seed of the t-SNE embedding.
    **kwargs
        Passed to ``sklearn.manifold.TSNE``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y = np.asarray(y)
    start = time.perf_counter()
    embedding = TSNE(n_components=2, random_state=random_state, **kwargs).fit_transform(
        np.asarray(X, dtype=float)
    )
    elapsed = time.perf_counter() - start

    fig, ax = plt.subplots()
    if type_of_target(y) in ("binary", "multiclass"):
        classes = np.unique(y)
        for cls in classes:
            points = embedding[y == cls]
            ax.scatter(points[:, 0], points[:, 1], alpha=0.7, label=str(cls))
        ax.legend(loc="best")
    else:
        points = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=y, cmap="RdBu", alpha=0.7
        )
        fig.colorbar(points, ax=ax)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_xlabel(f"Using {np.shape(X)[1]} features")
    ax.set_title(f"t-SNE Manifold (fit in {elapsed:0.2f} seconds)")
    return fig


@_styled
def plot_radviz(X, y, **kwargs) -> Figure:
    """Plot a RadViz projection of the columns of ``X`` colored by class.

    Parameters
    ----------
    X, y : numeric features and discrete target
    **kwargs
        Passed to ``pandas.plotting.radviz``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    X, y = np.asarray(X), np.asarray(y)

    frame = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    frame["class"] = y

    fig, ax = plt.subplots()
    kwargs.setdefault("color", _colors(len(np.unique(y))))
    pd.plotting.radviz(frame, "class", alpha=0.25, ax=ax, **kwargs)
    # pandas draws the unit circle without an edge color, which is invisible
    # under matplotlib's defaults, so draw it explicitly
    ax.add_patch(plt.Circle((0, 0), 1, facecolor="none", edgecolor=".6"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"RadViz for {X.shape[1]} Features")
    return fig


def _cluster_labels(estimator, X) -> np.ndarray:
    """Return the cluster label of every sample in ``X``."""
    if hasattr(estimator, "labels_"):
        return np.asarray(estimator.labels_)
    if hasattr(estimator, "fit_predict"):
        return np.asarray(deepcopy(estimator).fit_predict(X))
    raise TypeError(f"{type(estimator).__name__} does not expose cluster labels.")


def _distortion(X, labels) -> float:
    """Return the sum of squared distances of every sample to its cluster mean."""
    X = np.asarray(X, dtype=float)
    return sum(
        ((X[labels == label] - X[labels == label].mean(axis=0)) ** 2).sum()
        for label in np.unique(labels)
    )


def _elbow_index(x, y) -> int:
    """Return the index of the point farthest from the chord between the curve ends."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    points = np.column_stack(
        [(x - x.min()) / (np.ptp(x) or 1), (y - y.min()) / (np.ptp(y) or 1)]
    )
    chord = points[-1] - points[0]
    chord /= np.linalg.norm(chord) or 1
    offsets = points - points[0]
    return int(np.argmax(np.abs(offsets[:, 0] * chord[1] - offsets[:, 1] * chord[0])))


@_styled
def plot_elbow(
    estimator, X, k_range=range(2, 11), fit_kwargs: Optional[dict] = None, **kwargs
) -> Figure:
    """Plot the distortion score against the number of clusters and mark the elbow.

    A copy of ``estimator`` is refitted for every ``k``, so the estimator needs
    an ``n_clusters`` parameter.

    Parameters
    ----------
    estimator : clusterer with an ``n_clusters`` parameter
    X : features
    k_range : iterable of int, default range(2, 11)
        Numbers of clusters to evaluate.
    fit_kwargs : dict, optional
        Passed to ``estimator.fit``.
    **kwargs
        Passed to ``Axes.plot``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "n_clusters" not in estimator.get_params():
        raise TypeError(f"{type(estimator).__name__} has no n_clusters parameter.")

    ks = list(k_range)
    scores = []
    for k in ks:
        model = (
            deepcopy(estimator).set_params(n_clusters=k).fit(X, **(fit_kwargs or {}))
        )
        scores.append(_distortion(X, _cluster_labels(model, X)))
    elbow = _elbow_index(ks, scores)

    fig, ax = plt.subplots()
    ax.plot(ks, scores, marker="o", **kwargs)
    ax.axvline(
        ks[elbow],
        linestyle="--",
        color="black",
        linewidth=1,
        label=f"elbow at $k = {ks[elbow]}$, $score = {scores[elbow]:0.3f}$",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("distortion score")
    ax.set_title(f"Distortion Score Elbow for {type(estimator).__name__} Clustering")
    ax.legend(loc="best")
    return fig


@_styled
def plot_silhouette(estimator, X, **kwargs) -> Figure:
    """Plot the silhouette coefficient of every sample, grouped by cluster.

    Parameters
    ----------
    estimator : fitted clusterer
    X : features the estimator was fitted on
    **kwargs
        Passed to ``sklearn.metrics.silhouette_samples``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = _cluster_labels(estimator, X)
    clusters = np.unique(labels)
    if len(clusters) < 2:
        raise TypeError("The silhouette plot needs at least two clusters.")
    samples = silhouette_samples(X, labels, **kwargs)
    colors = _colors(len(clusters))

    fig, ax = plt.subplots()
    y_lower = 10
    for i, cluster in enumerate(clusters):
        values = np.sort(samples[labels == cluster])
        y_upper = y_lower + len(values)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            facecolor=colors[i],
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * len(values), str(cluster))
        y_lower = y_upper + 10
    ax.axvline(samples.mean(), color="red", linestyle="--", label="average silhouette")
    ax.set_yticks([])
    ax.set_xlabel("silhouette coefficient values")
    ax.set_ylabel("cluster label")
    ax.set_title(
        f"Silhouette Plot of {type(estimator).__name__} Clustering "
        f"for {len(labels)} Samples in {len(clusters)} Centers"
    )
    ax.legend(loc="best")
    return fig


@_styled
def plot_intercluster_distance(estimator, X, random_state=None, **kwargs) -> Figure:
    """Plot cluster centers embedded in two dimensions with MDS, sized by membership.

    Parameters
    ----------
    estimator : fitted clusterer with a ``cluster_centers_`` attribute
    X : features the estimator was fitted on
    random_state : int, optional
        Seed of the MDS embedding.
    **kwargs
        Passed to ``Axes.scatter``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not hasattr(estimator, "cluster_centers_"):
        raise TypeError(
            f"{type(estimator).__name__} has no cluster_centers_ attribute."
        )
    centers = np.asarray(estimator.cluster_centers_, dtype=float)
    labels = _cluster_labels(estimator, X)
    counts = np.bincount(labels[labels >= 0], minlength=len(centers))

    embedding = MDS(n_components=2, n_init=4, random_state=random_state).fit_transform(
        centers
    )
    sizes = 300 + 2500 * counts / (counts.max() or 1)

    fig, ax = plt.subplots()
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=sizes,
        alpha=0.5,
        edgecolor="black",
        **kwargs,
    )
    for i, (x, y) in enumerate(embedding):
        ax.annotate(str(i), (x, y), ha="center", va="center")
    ax.margins(0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{type(estimator).__name__} Intercluster Distance Map (via MDS)")
    return fig
