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

def ParzenWindow(x: np.ndarray, data: np.ndarray, kernelFn: callable):
    N = len(data)
    d = 1
    h = 1.06 * np.std(data) * N ** (-1 / (d + 4))
    V = h ** d

    k = sum([kernelFn((xi - x) / h) for xi in data])
    return k / (N * V)


class MinimumErrorBayes:
    def __init__(self, prior_probs: None | list[float] = None, use_parzen : bool = False, kernel = 'Gaussian') -> None:
        """
        Initialize the classifier.

        Parameters
        ----------
        prior_probs : None | list[float], optional
            Prior probabilities, by default None
        use_parzen : bool, optional
            Whether to use parzen window, by default False
        kernel : str, optional
            Type of the kernel function of the parzen window, by default 'Gaussian'
            Kernels available: 'Gaussian', 'Uniform'
        """
        self.priorProbs: None | list[float] = prior_probs
        self.featureNum: int = None
        self.classNum: int = None

        self.meanOfEachFeat: list[list[float]] = None
        self.varOfEachFeat: list[list[float]] = None

        self.useParzen: bool = use_parzen
        self.X_Train: None | np.ndarray = None
        self.y_Train: None | np.ndarray = None
        if kernel == 'Gaussian':
            self.kernelFn: callable = GaussianKernel
        elif kernel == 'Uniform':
            self.kernelFn: callable = UniformKernel


    def fit(self, X, y) -> None:
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
            # P(y) = count(y) / count(Y)
            for label in range(self.classNum):
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
    

    def CalculatePosteriorProbs(self, sample: np.ndarray) -> list[float]:
        # Conditional probability (irrelevant to the denominator): P(x|y) = P(x1|y) * P(x2|y) * ... * P(xn|y)
        condProbs: list[float] = []
        for label in range(self.classNum):
            condProbs.append(1.0)
            for i in range(self.featureNum):
                if not self.useParzen:
                    mean = self.meanOfEachFeat[label][i]
                    var = self.varOfEachFeat[label][i]
                    # Normal distribution
                    prob = math.exp(-((sample[i] - mean) ** 2) / (2.0 * var)) / math.sqrt(2.0 * math.pi * var) 
                else:
                    # Estimate density with Parzen window
                    prob = ParzenWindow(sample[i], self.X_Train[self.y_Train == label, i], self.kernelFn)
                condProbs[label] *= prob

        # Total probability: P(x) = sum(P(x|y) * P(y))
        totalProb = sum(self.priorProbs[label] * condProbs[label] for label in range(self.classNum))

        # Posterior probability: P(y|x) = P(x|y) * P(y) / P(x)
        posteriorProbs: list[float] = []
        for label in range(self.classNum):
            posteriorProb = self.priorProbs[label] * condProbs[label] / totalProb
            posteriorProbs.append(posteriorProb)
        return posteriorProbs


    def predict(self, X) -> np.ndarray:
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
        for sample in X:
            # To find the label with the minimum error rate in binary classification,
            # we need to find the label with the maximum posterior probability.
            posteriorProbs = self.CalculatePosteriorProbs(sample)
            bestClass = np.argmax(posteriorProbs)
            y_pred.append(bestClass)

        return np.array(y_pred)
