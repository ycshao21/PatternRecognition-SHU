import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

import logging
from models import classifier
from prutils.math import evaluation as eval 
import initialize

logger = logging.getLogger(name="Test")


def task_01(data):
    X = data[["身高(cm)", "体重(kg)", "鞋码"]].values.astype(float)
    y = data["性别"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Create KNN classifier
    knn = classifier.KNN(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Check accuracy of our model on the test data
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    eval.confusion_mat(
        pred=y_pred, truth=y_test, class_names=["Female", "Male"], title="KNN", show=True
    )

def task_02(data):
    X = data[["身高(cm)", "体重(kg)", "鞋码"]].values.astype(float)
    y = data["性别"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Create KNN classifier
    for n_neighbors in (1, 3, 5):
        knn = classifier.KNN(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Check accuracy of our model on the test data
        acc = eval.accuracy(pred=y_pred, truth=y_test)
        f1 = eval.f1_score(pred=y_pred, truth=y_test)
        logger.critical(f"[n_neighbors={n_neighbors}] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        eval.confusion_mat(
            pred=y_pred, truth=y_test, class_names=["Female", "Male"],
            title=f"KNN (n_neighbors={n_neighbors})", show=True
        )

def task_03(data):
    X = data[["身高(cm)", "体重(kg)", "鞋码"]].values.astype(float)
    y = data["性别"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Sample editing
    m = 100
    s = 5

    X_train_edited = X_train
    y_train_edited = y_train
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(
        X_train_edited[y_train_edited == 0][:, 0],
        X_train_edited[y_train_edited == 0][:, 1],
        X_train_edited[y_train_edited == 0][:, 2],
        c='r'
    )
    ax.scatter3D(
        X_train_edited[y_train_edited == 1][:, 0],
        X_train_edited[y_train_edited == 1][:, 1],
        X_train_edited[y_train_edited == 1][:, 2],
        c='b'
    )
    ax.set_title("Data")
    plt.show()

    for _ in range(m):
        # 1. Split the data into s subsets randomly
        indices = np.random.permutation(len(X_train_edited))
        X_train_edited = X_train_edited[indices]
        y_train_edited = y_train_edited[indices]
        X_train_subsets: list[np.ndarray] = np.array_split(X_train_edited, s)
        y_train_subsets: list[np.ndarray] = np.array_split(y_train_edited, s)

        edited = False
        for i in range(s):
            # 2. Classify
            knn = classifier.KNN(n_neighbors=3)
            knn.fit(X_train_subsets[(i + 1) % s], y_train_subsets[(i + 1) % s])
            y_pred = knn.predict(X_train_subsets[i])

            # 3. Edit
            if np.any(y_pred != y_train_subsets[i], axis=0):
                edited = True
            np.delete(X_train_subsets[i], np.where(y_pred != y_train_subsets[i]), axis=0)
            np.delete(y_train_subsets[i], np.where(y_pred != y_train_subsets[i]), axis=0)

        # 4. Mix
        X_train_edited = np.concatenate(X_train_subsets, axis=0)
        y_train_edited = np.concatenate(y_train_subsets, axis=0)

        if not edited:
            break

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(
        X_train_edited[y_train_edited == 0][:, 0],
        X_train_edited[y_train_edited == 0][:, 1],
        X_train_edited[y_train_edited == 0][:, 2],
        c='r'
    )

    ax.scatter3D(
        X_train_edited[y_train_edited == 1][:, 0],
        X_train_edited[y_train_edited == 1][:, 1],
        X_train_edited[y_train_edited == 1][:, 2],
        c='b'
    )
    ax.set_title("Data")
    plt.show()
    # Create KNN classifier
    knn = classifier.KNN(n_neighbors=3)
    knn.fit(X_train_edited, y_train_edited)
    y_pred = knn.predict(X_test)

    # Check accuracy of our model on the test data
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    eval.confusion_mat(
        pred=y_pred, truth=y_test, class_names=["Female", "Male"], title=f"KNN", show=True
    )


if __name__ == "__main__":
    initialize.init()
    data = pd.read_csv("dataset/genderdata/preprocessed/all.csv")
    task_01(data)
    task_02(data)
    task_03(data)