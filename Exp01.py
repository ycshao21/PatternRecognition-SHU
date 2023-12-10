import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Models import MinimumErrorBayes
from Utils.Metrics import PrintAccuracy, DisplayConfusionMatrix


def Test(X: np.ndarray, y: np.ndarray, use_parzen: bool, prior_probs=None) -> None:
    model = MinimumErrorBayes.MinimumErrorBayes(prior_probs=prior_probs, use_parzen=use_parzen)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    model.Fit(X_train, y_train)

    y_pred = model.Predict(X_test)
    PrintAccuracy(y_pred, y_test)
    DisplayConfusionMatrix(y_pred, y_test)


def Task01(data: pd.DataFrame) -> None:
    """
    应用单个特征进行实验：以(a)身高、(b)体重或(c)鞋码尺寸数据作为特征，
    在正态分布假设下利用最大似然法或者贝叶斯估计法估计分布密度参数（只利用训练数据估计密度），
    建立最小错误率Bayes分类器，写出得到的决策规则，将该分类器应用到测试样本，考察测试错误情况。
    在分类器设计时可以考察采用不同先验概率（如0.5 vs. 0.5, 0.75 vs. 0.25, 0.9 vs. 0.1 等）进行实验，
    考察对决策规则和错误率的影响。
    """
    X = data['体重(kg)'].values.astype(float)
    y = data['性别'].values.astype(int)
    Test(X, y, use_parzen=False)


def Task02(data: pd.DataFrame) -> None:
    """
    应用两个特征进行实验：同时采用身高和体重数据作为特征，假设二者不相关，
    在正态分布假设下估计概率密度，建立最小错误率Bayes分类器，写出得到的决策规则，
    将该分类器应用到训练/测试样本，考察训练/测试错误情况。
    在分类器设计时可以考察采用不同先验概率（如0.5 vs. 0.5, 0.75 vs. 0.25, 0.9 vs. 0.1 等）进行实验，
    考察对决策规则和错误率的影响。
    """
    X = data[['身高(cm)', '体重(kg)']].values.astype(float)
    y = data['性别'].values.astype(int)
    Test(X, y, use_parzen=False)
    

def Task03(data: pd.DataFrame) -> None:
    """
    采用Parzen窗法估计概率密度
    """
    X = data['体重(kg)'].values.astype(float)
    y = data['性别'].values.astype(int)
    Test(X, y, use_parzen=True)


if __name__ == '__main__':
    data = pd.read_csv("Dataset/genderData_Merged.csv")
    Task03(data)