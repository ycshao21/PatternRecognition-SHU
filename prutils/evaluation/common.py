import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics  as skmetrics
import seaborn as sns


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return np.mean(pred == truth)


def f1_score(pred: np.ndarray, truth: np.ndarray) -> float:
    cm = skmetrics.confusion_matrix(truth, pred)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    return 2 * precision * recall / (precision + recall)


def confusion_mat(
    pred: np.ndarray,
    truth: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    camp: str = "Blues",
    figsize: tuple = (8, 6),
    fontsize: int = 14,
    dpi: int = 100,
    save_path: str = None,
    show: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cm = skmetrics.confusion_matrix(truth, pred)
    # Devide each element by the sum of the corresponding row
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # Plot
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        data=cm_percent,
        cmap=camp,
        annot=True,
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title, fontsize=fontsize)
    plt.xlabel("Predicted", fontsize=fontsize)
    plt.ylabel("True", fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=45)
    plt.yticks(fontsize=fontsize, rotation=45)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return (cm, cm_percent)
