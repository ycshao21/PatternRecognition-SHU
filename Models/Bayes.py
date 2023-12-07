import math
import numpy as np

def MLE(data: np.ndarray):
    """
    Maximum Likelihood Estimation

    Parameters
    ----------
    data : np.ndarray
        Data to be estimated.

    Returns
    -------
    mean : float
        The estimated mean.
    var : float
        The estimated variance.
    """
    return np.mean(data), np.var(data)

################### Parzen Window ###################

def UniformKernel(x: float):
    if abs(x) < 0.5:
        return 1.0
    else:
        return 0.0

def GaussianKernel(x: float):
    return np.exp(-x ** 2 / 2.0) / np.sqrt(2.0 * math.pi)

def ParzenWindow(x: float, data: np.ndarray, kernelFn: callable):
    n = len(data)
    d = 1
    h = 1.06 * np.std(data) * n ** (-1 / (d + 4))
    V = h ** d

    k = sum([kernelFn((xi - x) / h) for xi in data])
    return k / (n * V)


class Bayes:
    def __init__(self, prior_probs: None | list[float] = None, use_parzen : bool = False) -> None:
        """
        Initialize the Bayes classifier.

        Parameters
        ----------
        prior_probs : None | list[float], optional
            Prior probabilities of each class, by default None.
        """
        self.priorProbs: None | list[float] = prior_probs
        self.featureNum: int = None
        self.classNum: int = None

        self.meanOfEachFeat: list[list[float]] = None
        self.varOfEachFeat: list[list[float]] = None

        self.useParzen: bool = use_parzen
        self.X_Train: None | np.ndarray = None
        self.y_Train: None | np.ndarray = None


    def Fit(self, X, y) -> None:
        """
        Fit the classifier.

        Parameters
        ----------
        X : list[list] | np.ndarray
            Training data.
        y : list | np.ndarray
            Labels.

        Raises
        ------
        Exception
            If the number of classes is greater than 2.
        """

        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.featureNum = X.shape[1]
        self.classNum = len(set(y))
        if self.classNum != 2:
            raise Exception("The classifier only supports binary classification.")
        
        # Prior probability
        if self.priorProbs is None:
            self.priorProbs = []
            for label in range(self.classNum):
                # P(y) = count(y) / count(Y)
                self.priorProbs.append(len(y[y == label]) / len(y))

        if not self.useParzen:
            # Mean and standard deviation of each feature
            self.meanOfEachFeat: list[list] = []
            self.varOfEachFeat: list[list] = []
            for label in range(self.classNum):
                means, vars = [], []
                for feat in range(self.featureNum):
                    mean, var = MLE(X[y == label, feat])
                    means.append(mean)
                    vars.append(var)
                self.meanOfEachFeat.append(means)
                self.varOfEachFeat.append(vars)
        else:
            self.X_Train = X
            self.y_Train = y


    def Predict(self, X) -> np.ndarray:
        """
        Predict the labels of the given data.

        Parameters
        ----------
        X : list[list] | np.ndarray
            Data to be predicted.

        Returns
        -------
        np.ndarray
            Predicted labels of the given data.

        Raises
        ------
        Exception
            If the number of features of the given data is not equal to the number of features of the training data.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.featureNum:
            raise Exception(f"Feature number mismatched. Expected {self.featureNum} features, got {X.shape[1]} features.")

        y_pred = []
        for i in range(X.shape[0]):  # For each sample
            posteriorProbs: list[float] = []
            if not self.useParzen:
                # Conditional probability (irrelevant to the denominator): P(x|y) = P(x1|y) * P(x2|y) * ... * P(xn|y)
                condProbs: list[float] = []
                for label in range(self.classNum):
                    condProbs.append(1.0)
                    for feat in range(self.featureNum):
                        mean = self.meanOfEachFeat[label][feat]
                        var = self.varOfEachFeat[label][feat]
                        # Normal distribution
                        prob = math.exp(-((X[i][feat] - mean) ** 2) / (2.0 * var)) / math.sqrt(2.0 * math.pi * var) 
                        condProbs[label] *= prob

                # Total probability: P(x) = sum(P(x|y) * P(y))
                totalProb = sum(self.priorProbs[label] * condProbs[label] for label in range(self.classNum))

                # Posterior probability: P(y|x) = P(x|y) * P(y) / P(x)
                for label in range(self.classNum):
                    posteriorProb = self.priorProbs[label] * condProbs[label] / totalProb
                    posteriorProbs.append(posteriorProb)
            else:
                for label in range(self.classNum):
                    # Estimate density with Parzen window
                    density = 1.0
                    for j in range(self.featureNum):
                        density *= ParzenWindow(X[i, j], self.X_Train[self.y_Train == label, j], GaussianKernel)

                    # Posterior probability
                    posteriorProb = self.priorProbs[label] * density
                    posteriorProbs.append(posteriorProb)

            # To find the label with the minimum error rate in binary classification,
            # we need to find the label with the maximum posterior probability.
            bestClass = np.argmax(posteriorProbs)
            y_pred.append(bestClass)

        return np.array(y_pred)
