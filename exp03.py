import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
from models import classifier
from prutils.math import evaluation as eval
import initialize
from matplotlib.colors import ListedColormap

logger = logging.getLogger(name="Test")


def task_01(data):
    """
    同时采用身高和体重数据作为特征，用Fisher线性判别方法求分类器，
    将该分类器应用到训练和测试样本，考察训练和测试错误情况。
    将训练样本和求得的决策边界画到图上，同时把以往用Bayes方法求得的分类器也画到图上，比
    较结果的异同。
    """
    X = data[["身高(cm)", "体重(kg)"]].values.astype(float)
    y = data["性别"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    p = X_train[:, 0]
    q = X_train[:, 1]

    # Standardize data
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # x_transform = X_train

    # Fit the model
    fisher = classifier.Fisher()
    bayes = classifier.MinimumErrorBayes()

    fisher.fit(X_train, y_train)
    bayes.fit(X_train, y_train)

    # Test the model
    y_pred_fisher = fisher.predict(X_test)
    y_pred_bayes = bayes.predict(X_test)

    acc_fisher = eval.accuracy(pred=y_pred_fisher, truth=y_test)
    f1_fisher = eval.f1_score(pred=y_pred_fisher, truth=y_test)
    logger.critical(f"[Fisher] Accuracy: {acc_fisher:.4f}, F1 Score: {f1_fisher:.4f}")
    eval.confusion_mat(
        pred=y_pred_fisher,
        truth=y_test,
        class_names=["Female", "Male"],
        title="Fisher",
        show=True,
    )

    acc_bayes = eval.accuracy(pred=y_pred_bayes, truth=y_test)
    f1_bayes = eval.f1_score(pred=y_pred_bayes, truth=y_test)
    logger.critical(f"[Bayes] Accuracy: {acc_bayes:.4f}, F1 Score: {f1_bayes:.4f}")
    eval.confusion_mat(
        pred=y_pred_bayes,
        truth=y_test,
        class_names=["Female", "Male"],
        title="Bayes",
        show=True,
    )

    # Visualize the training data
    # Fisher
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Decision Boundary Analysis')
    # x_min, x_max = -5, 5
    # y_min, y_max = -5, 5
    # x = x_transform[:, 0]
    # y = x_transform[:, 1]
    # t = np.array([x_min, x_max])
    # # line: k,b b=-w_0, k=-w_1/w_2
    # k = -fisher.projection[0] / fisher.projection[1]
    # b = -fisher.threshold
    # y_pred = k * t + b
    # ax1.scatter(x, y, c=y_train, cmap=plt.cm.coolwarm, edgecolors="k")
    # ax1.plot(t, y_pred, color="green", linestyle="-", linewidth=2, marker="o")
    # ax1.set_xlim(x_min, x_max)
    # ax1.set_ylim(y_min, y_max)
    # ax1.set_title("Fisher Linear Discriminant1")

    sex = ["female","male"]
    # Fisher predict decision boundary
    axes = [140, 200, 20, 100]
    xp = np.linspace(axes[0], axes[1], 200)
    yp = np.linspace(axes[2], axes[3], 200)
    xx, yy = np.meshgrid(xp, yp)
    G = np.c_[xx.ravel(), yy.ravel()]
    Z = fisher.predict(G)
    Z = Z.reshape(xx.shape)
    custom_cmap = ListedColormap(["#fafab0", "#9898ff"])
    ax1.contourf(xx, yy, Z, cmap=custom_cmap, linewidth=5, alpha=0.5)
    p1 = ax1.scatter(p[y_train == 0], q[y_train == 0], c="r", s=50, edgecolors="k")
    p2 = ax1.scatter(p[y_train == 1], q[y_train == 1], c="b", s=50, edgecolors="k")
    ax1.legend([p1,p2], sex, loc = 'upper right')
    ax1.set_title("Fisher")

    # Bayes
    xp = np.linspace(axes[0], axes[1], 200)
    yp = np.linspace(axes[2], axes[3], 200)
    xx, yy = np.meshgrid(xp, yp)
    X = np.c_[xx.ravel(), yy.ravel()]
    Z = bayes.predict(X)
    Z = Z.reshape(xx.shape)
    custom_cmap = ListedColormap(["#fafab0", "#9898ff"])
    ax2.contourf(xx, yy, Z, cmap=custom_cmap, linewidth=5, alpha=0.5)
    p1 = ax2.scatter(p[y_train == 0], q[y_train == 0], c="r", s=50, edgecolors="k")
    p2 = ax2.scatter(p[y_train == 1], q[y_train == 1], c="b", s=50, edgecolors="k")
    ax2.legend([p1,p2], sex, loc = 'upper right')
    ax2.set_title("Bayes")

    plt.show()


def task_02(data):
    X = data[["身高(cm)", "体重(kg)"]].values.astype(float)
    y = data["性别"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the model
    model = classifier.Fisher()

    loo_err = 0
    loo = LeaveOneOut()
    loo.get_n_splits(X_train)
    for train_index, test_index in loo.split(X_train):
        X_train_loo, X_test_loo = X_train[train_index], X_train[test_index]
        y_train_loo, y_test_loo = y_train[train_index], y_train[test_index]
        model.fit(X_train_loo, y_train_loo)

        y_pred = model.predict(X_test_loo)
        acc = eval.accuracy(pred=y_pred, truth=y_test_loo)
        err = 1 - acc
        loo_err += err
    loo_err /= len(X_train)
    logger.critical(f"Leave-One-Out Error: {loo_err:.4f}")

    # Test the model
    test_err = 0
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = eval.accuracy(pred=y_pred, truth=y_test)
    test_err += 1 - acc
    logger.critical(f"Test Error: {test_err:.4f}")


if __name__ == "__main__":
    initialize.init()
    data = pd.read_csv("dataset/genderdata/preprocessed/all.csv")
    task_01(data)
    task_02(data)
