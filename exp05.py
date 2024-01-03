import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

import logging
from models import classifier
from prutils.math import evaluation as eval
import initialize

logger = logging.getLogger(name="Test")

LABELS = ["Female", "Male"]
FEATURES = ["height(cm)", "weight(kg)", "shoe_size"]


def task_01(data):
    X = data[FEATURES].values.astype(float)
    y = data["sex"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Create KNN classifier
    n_neighbors = 3
    knn = classifier.KNN(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Check accuracy of our model on the test data
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    eval.plot_confusion_mat(
        pred=y_pred, truth=y_test,
        class_names=LABELS,
        title=f"Task 01: KNN (n_neighbors={n_neighbors})",
        show=False
    )
    plt.show()


def task_02(data):
    X = data[FEATURES].values.astype(float)
    y = data["sex"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Create KNN classifier
    n_neighbors_list = [1, 3, 5, 7, 9, 11]
    plt.figure(figsize=(16, 9))
    y_scores = []
    for i, n_neighbors in enumerate(n_neighbors_list):
        knn = classifier.KNN(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        y_score = knn.predict_prob(X_test)[:, 1]
        y_scores.append(y_score)

        # Check accuracy of our model on the test data
        acc = eval.accuracy(pred=y_pred, truth=y_test)
        f1 = eval.f1_score(pred=y_pred, truth=y_test)
        logger.critical(
            f"[n_neighbors={n_neighbors}] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}"
        )

        # Confusion matrix
        ax_cm = plt.subplot(2, 3, i + 1)
        eval.plot_confusion_mat(
            pred=y_pred, truth=y_test, class_names=LABELS, show=False
        )
        ax_cm.set_title(f"n_neighbors={n_neighbors}")

    plt.figure(figsize=(10, 6))
    for n_neighbors, y_score in zip(n_neighbors_list, y_scores):
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"n_neighbors={n_neighbors}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess", color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.show()


def task_03(data):
    X = data[FEATURES[:2]].values.astype(float)
    y = data["sex"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Sample editing
    knn = classifier.KNN(n_neighbors=20)
    knn.fit(
        X_train,
        y_train,
        method="multi-edit",
        split=3,
        target_count_of_no_misclassified=3,
        whether_visualize=True,
    )


if __name__ == "__main__":
    initialize.init()
    data = pd.read_csv("dataset/genderdata/preprocessed/all.csv")
    task_01(data)
    task_02(data)
    # task_03(data)
