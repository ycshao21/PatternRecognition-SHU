import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    X = data["身高(cm)"].values.astype(float)
    y = data["性别"].values.astype(int)

    # Fit the model
    model = classifier.MinimumErrorBayes(prior_probs=None, use_parzen=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    eval.confusion_mat(
        pred=y_pred, truth=y_test, class_names=["Female", "Male"], show=True
    )

    # Visualize the model
    xMin, xMax = 140, 200
    classA, classB = [], []
    for x in range(xMin, xMax, 1):
        probA, probB = model.cal_posterior_probs(np.array([x]))
        classA.append(probA)
        classB.append(probB)
    plt.plot(range(xMin, xMax, 1), classA, label="Female")
    plt.plot(range(xMin, xMax, 1), classB, label="Male")
    plt.legend()
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

    # Fit the model
    model = classifier.MinimumErrorBayes(prior_probs=None, use_parzen=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    eval.confusion_mat(
        pred=y_pred, truth=y_test, class_names=["Female", "Male"], show=True
    )

    # Visualize the model (surface)
    xMin, xMax = 140, 200
    yMin, yMax = 20, 100
    X_Mesh, y_Mesh = np.meshgrid(
        np.linspace(xMin, xMax, 100), np.linspace(yMin, yMax, 100)
    )
    classA, classB = [], []
    for i in range(X_Mesh.shape[0]):
        for j in range(X_Mesh.shape[1]):
            probA, probB = model.cal_posterior_probs(
                np.array([X_Mesh[i, j], y_Mesh[i, j]])
            )
            classA.append(probA)
            classB.append(probB)
    classA = np.array(classA).reshape(X_Mesh.shape)
    classB = np.array(classB).reshape(X_Mesh.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X_Mesh, y_Mesh, classA, cmap="autumn", alpha=0.6)
    ax.plot_surface(X_Mesh, y_Mesh, classB, cmap="winter", alpha=0.6)
    ax.set_xlabel("Height")
    ax.set_ylabel("Weight")

    plt.show()


def task_03(data: pd.DataFrame) -> None:
    """
    采用Parzen窗法估计概率密度
    """
    X = data["体重(kg)"].values.astype(float)
    y = data["性别"].values.astype(int)

    # Fit the model
    model = classifier.MinimumErrorBayes(prior_probs=None, use_parzen=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    f1 = eval.f1_score(pred=y_pred, truth=y_test)
    logger.critical(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    eval.confusion_mat(
        pred=y_pred, truth=y_test, class_names=["Female", "Male"], show=True
    )

    # Visualize the model
    xMin, xMax = 40, 100
    classA, classB = [], []
    for x in range(xMin, xMax, 1):
        probA, probB = model.cal_posterior_probs(np.array([x]))
        classA.append(probA)
        classB.append(probB)
    plt.plot(range(xMin, xMax, 1), classA, label="Female")
    plt.plot(range(xMin, xMax, 1), classB, label="Male")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    initialize.init()
    data = pd.read_csv("dataset/genderdata/preprocessed/all.csv")
    task_01(data)
    task_02(data)
    task_03(data)