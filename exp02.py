import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import logging
from models import classifier
from prutils.math import evaluation as eval
import initialize

logger = logging.getLogger(name="Test")


def task_01(data: pd.DataFrame) -> None:
    """
    应用单个特征进行实验：以(a)身高、(b)体重或(c)鞋码尺寸数据作为特征，
    在正态分布假设下利用最大似然法或者贝叶斯估计法估计分布密度参数（只利用训练数据估计密度），
    建立最小错误率Bayes分类器，写出得到的决策规则，将该分类器应用到测试样本，考察测试错误情况。
    在分类器设计时可以考察采用不同先验概率（如0.5 vs. 0.5, 0.75 vs. 0.25, 0.9 vs. 0.1 等）进行实验，
    考察对决策规则和错误率的影响。
    """
    features = ("身高(cm)", "体重(kg)", "鞋码")
    features_en = ("Height", "Weight", "Shoe Size")

    for feature, feature_en in zip(features, features_en):
        X = data[feature].values.astype(float)
        y = data["性别"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

        # Fit the model
        model = classifier.MinimumErrorBayes(prior_probs=None, use_parzen=False)
        model.fit(X_train, y_train)

        # Test the model
        y_pred = model.predict(X_test)
        acc = eval.accuracy(pred=y_pred, truth=y_test)
        f1 = eval.f1_score(pred=y_pred, truth=y_test)
        logger.critical(f"{feature} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        plt.figure(figsize=(16, 6))

        ax1 = plt.subplot(1, 2, 1)
        eval.plot_confusion_mat(
            pred=y_pred,
            truth=y_test,
            class_names=["Female", "Male"],
            show=False,
            fontsize=10,
        )
        ax1.set_title("Confusion Matrix", fontsize=12)

        # Visualize the model
        mean0, mean1 = model.feature_means
        var0, var1 = model.feature_vars
        a = 1.0 / (2.0 * var0) - 1.0 / (2.0 * var1)
        b = mean1 / var1 - mean0 / var0
        c = - mean1 ** 2 / (2 * var1) + mean0 ** 2 / (2 * var0) + np.log(model.prior_probs[0] / model.prior_probs[1])

        # a = var0 - var1
        # b = 2.0 * (var1 * mean0 - var0 * mean1) 
        # c = var0 * (mean1 ** 2) - var1 * (mean0 ** 2)
        # - 2.0 * var0 * var1 * np.log((model.prior_probs[1] * np.sqrt(var0)) / (model.prior_probs[0] * np.sqrt(var1)))
        poly = np.poly1d([a[0], b[0], c[0]])
        roots = np.roots(poly)

        ax2 = plt.subplot(1, 2, 2)
        x_min = np.min(X) - 10.0
        x_max = np.max(X) + 10.0
        x_range = np.arange(x_min, x_max + 1, 0.5)

        probs = np.array([model.cal_posterior_probs(x) for x in x_range])
        plt.plot(x_range, probs[:, 0], label="Female")
        plt.plot(x_range, probs[:, 1], label="Male")
        for root in roots:
            plt.axvline(x=root, c="r", linestyle="--")
        plt.xlim(x_min, x_max)
        plt.legend()
        ax2.set_title("Posterior Probability", fontsize=12)

        plt.show()


def task_02(data: pd.DataFrame) -> None:
    """
    应用两个特征进行实验：同时采用身高和体重数据作为特征，假设二者不相关，
    在正态分布假设下估计概率密度，建立最小错误率Bayes分类器，写出得到的决策规则，
    将该分类器应用到训练/测试样本，考察训练/测试错误情况。
    在分类器设计时可以考察采用不同先验概率（如0.5 vs. 0.5, 0.75 vs. 0.25, 0.9 vs. 0.1 等）进行实验，
    考察对决策规则和错误率的影响。
    """
    X = data[["身高(cm)", "体重(kg)"]].values.astype(float)
    y = data["性别"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Fit the model
    model = classifier.MinimumErrorBayes(prior_probs=None, use_parzen=False)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    plt.figure(figsize=(10, 8))
    eval.plot_confusion_mat(
        pred=y_pred,
        truth=y_test,
        class_names=["Female", "Male"],
        title="Task 02: Confusion Matrix - Height & Weight",
        show=False
    )

    # Visualize the model (surface)
    fig2 = plt.figure(figsize=(10, 8))

    x_min = np.min(X[:, 0]) - 1
    x_max = np.max(X[:, 0]) + 1
    y_min = np.min(X[:, 1]) - 1
    y_max = np.max(X[:, 1]) + 1
    X_Mesh, y_Mesh = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )

    ZA = np.zeros((X_Mesh.shape[0], X_Mesh.shape[1]))
    ZB = np.zeros((X_Mesh.shape[0], X_Mesh.shape[1]))
    for i in range(X_Mesh.shape[0]):
        for j in range(X_Mesh.shape[1]):
            ZA[i, j] = model.multivar_density(
                x=np.array([X_Mesh[i, j], y_Mesh[i, j]]), which_class=0
            )
            ZB[i, j] = model.multivar_density(
                x=np.array([X_Mesh[i, j], y_Mesh[i, j]]), which_class=1
            )


    ax = fig2.add_subplot(111, projection="3d")
    ax.plot_surface(X_Mesh, y_Mesh, ZA, cmap="autumn", alpha=0.6)
    ax.plot_surface(X_Mesh, y_Mesh, ZB, cmap="winter", alpha=0.6)

    # Adding contour lines to the Z-axis
    contour_levels = np.linspace(np.min(ZA), np.max(ZB), 20)  # Adjust the number of levels as needed
    ax.contour(X_Mesh, y_Mesh, ZA, levels=contour_levels, offset=np.min(ZA), cmap="autumn", linestyles="solid")
    ax.contour(X_Mesh, y_Mesh, ZB, levels=contour_levels, offset=np.min(ZB), cmap="winter", linestyles="solid")
    ax.set_xlabel("Height")
    ax.set_ylabel("Weight")
    ax.set_zlabel("Density")
    ax.set_title("Density of Multivariate Normal Distribution, class")

    plt.show()


def task_03(data: pd.DataFrame) -> None:
    """
    采用Parzen窗法估计概率密度
    """
    features = ("身高(cm)", "体重(kg)", "鞋码")
    features_en = ("Height", "Weight", "Shoe Size")

    for feature, feature_en in zip(features, features_en):
        X = data[feature].values.astype(float)
        y = data["性别"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

        # Fit the model
        model = classifier.MinimumErrorBayes(prior_probs=None, use_parzen=False)
        model_parzen = classifier.MinimumErrorBayes(prior_probs=None, use_parzen=True)
        model.fit(X_train, y_train)
        model_parzen.fit(X_train, y_train)

        # Test the model
        plt.figure(figsize=(16, 6))

        # Not using parzen >>>>>>
        y_pred = model.predict(X_test)
        acc = eval.accuracy(pred=y_pred, truth=y_test)
        f1 = eval.f1_score(pred=y_pred, truth=y_test)
        logger.critical(f"[No parzen] {feature} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        ax1 = plt.subplot(1, 2, 1)
        eval.plot_confusion_mat(
            pred=y_pred,
            truth=y_test,
            class_names=["Female", "Male"],
            show=False
        )
        ax1.set_title(f"Not using Parzen")
        # <<<<<<

        # Using parzen >>>>>>
        y_pred = model_parzen.predict(X_test)
        acc = eval.accuracy(pred=y_pred, truth=y_test)
        f1 = eval.f1_score(pred=y_pred, truth=y_test)
        logger.critical(f"[Parzen] {feature} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        ax2 = plt.subplot(1, 2, 2)
        eval.plot_confusion_mat(
            pred=y_pred,
            truth=y_test,
            class_names=["Female", "Male"],
            show=False
        )
        ax2.set_title(f"Using Parzen")
        # <<<<<<

        # Visualize the model
        x_min = np.min(X) - 10.0
        x_max = np.max(X) + 10.0
        x_range = np.arange(x_min, x_max + 1, 0.5)
        probs = np.array([model.cal_posterior_probs(x) for x in x_range])
        probs_parzen = np.array([model_parzen.cal_posterior_probs(x) for x in x_range])

        plt.figure(figsize=(10, 8))
        plt.plot(x_range, probs[:, 0], label="Female (No parzen)", c="chocolate")
        plt.plot(x_range, probs[:, 1], label="Male (No parzen)", c="mediumblue")

        plt.plot(x_range, probs_parzen[:, 0], label="Female (Parzen)", c="gold")
        plt.plot(x_range, probs_parzen[:, 1], label="Male (Parzen)", c="deepskyblue")
        plt.title(f"Task 03: Decision Boundary - {feature_en}")
        plt.legend()
        plt.show()

def task_04(data: pd.DataFrame) -> None:
    """
    采用最小风险贝叶斯决策
    """
    X = data["体重(kg)"].values.astype(float)
    y = data["性别"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Fit the model
    risk = np.array([[0,6],[1,0]])
    model = classifier.MinimumErrorBayes(prior_probs=None, use_risk=True, risk=risk)
    model.fit(X_train, y_train)
    # Test the model
    y_pred = model.predict(X_test)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    eval.plot_confusion_mat(
        pred=y_pred, truth=y_test, class_names=["Female", "Male"], show=True
    )

    # Visualize the model
    x_min, x_max = np.min(X) - 1, np.max(X) + 1
    x_range = np.arange(x_min, x_max + 1, 0.5)
    probs = np.array([model.cal_posterior_probs(x) for x in x_range])
    plt.plot(range(x_min, x_max, 1), probs[:, 0], label="Female_risk")
    plt.plot(range(x_min, x_max, 1), probs[:, 1], label="Male_risk")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    initialize.init()
    data = pd.read_csv("dataset/genderdata/preprocessed/all.csv")
    task_01(data)
    task_02(data)
    task_03(data)
    task_04(data)
