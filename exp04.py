import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import logging
from models import classifier
from prutils.math import evaluation as eval, decomposition
import initialize

logger = logging.getLogger(name="Test")

LABELS = ["Female", "Male"]
FEATURES = ["height(cm)", "weight(kg)", "shoe_size"]

def task_01(data):
    """
    对整个样本集进行K-L 变换（即PCA），提取二个主分量，
    并将计算出的新特征方向表示在二维平面上，
    考察投影到本征值最大的方向后男女样本的分布情况并用该主成分进行分类
    """
    X = data[FEATURES].values.astype(float)
    y = data["sex"].values.astype(int)

    # Standardize data
    X = StandardScaler().fit_transform(X)

    # Visualize data
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Task 01")

    # 1. raw data >>>>>>
    ax1 = plt.subplot(131, projection='3d')
    ax1.scatter3D(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2], c='r')
    ax1.scatter3D(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], c='b')
    ax1.set_title("Raw Data")
    # <<<<<<

    # 2. PCA data >>>>>>
    X_pca = decomposition.PCA(n_components=2).fit_transform(X)

    ax2 = plt.subplot(132)
    ax2.scatter(X_pca[y == 0][:, 0], X_pca[y == 0][:, 1], c='r')
    ax2.scatter(X_pca[y == 1][:, 0], X_pca[y == 1][:, 1], c='b')
    ax2.set_title("PCA")
    # <<<<<<

    # 3. PCA data (sklearn) >>>>>>
    X_pca_skl = PCA(n_components=2).fit_transform(X)

    ax3 = plt.subplot(133)
    ax3.scatter(X_pca_skl[y == 0][:, 0], X_pca_skl[y == 0][:, 1], c='r')
    ax3.scatter(X_pca_skl[y == 1][:, 0], X_pca_skl[y == 1][:, 1], c='b')
    ax3.set_title("PCA (sklearn)")
    # <<<<<<
    plt.show()

def task_02(data, model_name: str):
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

    # Fit the model
    model_name = model_name.capitalize()
    if model_name == 'Fisher':
        model = classifier.Fisher()
        model_PCA = classifier.Fisher()
        model_PCA_skl = classifier.Fisher()
    elif model_name == 'Bayes':
        model = classifier.MinimumErrorBayes()
        model_PCA = classifier.MinimumErrorBayes()
        model_PCA_skl = classifier.MinimumErrorBayes()
    else:
        raise ValueError("Invalid model name.")

    # Test the model
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle("Task 02")

    # 1. Not using PCA >>>>>>
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"[{model_name}_Raw] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    roc = roc_curve(y_test, y_pred)
    print(roc)

    ax1 = plt.subplot(131)
    eval.plot_confusion_mat(
        pred=y_pred, truth=y_test, class_names=LABELS, title=model_name, show=False
    )
    ax1.set_title("Raw Data")
    # <<<<<<

    # 2. Using PCA >>>>>>
    pca = decomposition.PCA(n_components=2)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)
    X_test_PCA = pca.transform(X_test)

    model_PCA.fit(X_train_PCA, y_train)

    y_pred = model_PCA.predict(X_test_PCA)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"[{model_name}_PCA] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    ax2 = plt.subplot(132)
    eval.plot_confusion_mat(
        pred=y_pred, truth=y_test, class_names=LABELS, title=f"{model_name}_PCA", show=False
    )
    ax2.set_title("PCA")
    # <<<<<<

    # 3. Using PCA (sklearn) >>>>>>
    pca_skl = PCA(n_components=2)
    pca_skl.fit(X_train)
    X_train_PCA_skl = pca_skl.transform(X_train)
    X_test_PCA_skl = pca_skl.transform(X_test)

    model_PCA_skl.fit(X_train_PCA_skl, y_train)

    y_pred_skl = model_PCA_skl.predict(X_test_PCA_skl)
    acc = eval.accuracy(pred=y_pred_skl, truth=y_test)
    f1 = eval.f1_score(pred=y_pred_skl, truth=y_test)
    logger.critical(f"[{model_name}_PCA_sklearn] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    ax3 = plt.subplot(133)
    eval.plot_confusion_mat(
        pred=y_pred_skl, truth=y_test, class_names=LABELS, title=f"{model_name}_PCA_sklearn", show=False
    )
    ax3.set_title("PCA (sklearn)")
    # <<<<<<
    plt.show()

def task_03(data):
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

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Task 03: Bayes")

    # PCA >>>>>>
    pca = decomposition.PCA(n_components=2)
    pca.fit(X_train)
    X_train_PCA = pca.transform(X_train)
    X_test_PCA = pca.transform(X_test)

    model = classifier.MinimumErrorBayes()
    model.fit(X_train_PCA, y_train)

    y_pred = model.predict(X_test_PCA)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"[PCA] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    ax1 = plt.subplot(1, 2, 1)
    eval.plot_confusion_mat(
        pred=y_pred, truth=y_test, class_names=LABELS, show=False
    )
    ax1.set_title("PCA")
    # <<<<<<

    # FLDA >>>>>>
    flda = decomposition.FLDA(n_components=2)
    flda.fit(X_train, y_train)
    X_train_FLDA = flda.transform(X_train)
    X_test_FLDA = flda.transform(X_test)

    model = classifier.MinimumErrorBayes()
    model.fit(X_train_FLDA, y_train)

    y_pred = model.predict(X_test_FLDA)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"[FLDA] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    ax2 = plt.subplot(1, 2, 2)
    eval.plot_confusion_mat(
        pred=y_pred, truth=y_test, class_names=LABELS, show=False
    )
    ax2.set_title("FLDA")
    # <<<<<<

    plt.show()

if __name__ == "__main__":
    initialize.init()
    data = pd.read_csv("dataset/genderdata/preprocessed/all.csv")
    task_01(data)
    task_02(data, model_name='Bayes')
    task_03(data)