import math
import numpy as np
import pandas as pd
from collections import defaultdict

# from sklearn.naive_bayes import GaussianNB
# bayes = GaussianNB()
# bayes.fit(train_X, train_y)
# bayes.predict(pred_X)

class Bayes:
    def __init__(self, prior_probs: None | list[float] = None) -> None:
        self.priorProbs: None | list[float] = prior_probs
        self.featureNum: int = None
        self.classNum: int = None
        self.meanOfEachFeat: list[list[float]] = None
        self.stdOfEachFeat: list[list[float]] = None


    def Fit(self, X, y) -> None:
        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.featureNum = X.shape[1]
        self.classNum = len(set(y))
        if self.classNum > 2:
            raise Exception("The classifier only supports binary classification.")

        # Prior probability
        if self.priorProbs is None:
            self.priorProbs = []
            for label in range(self.classNum):
                # P(y) = count(y) / count(Y)
                self.priorProbs.append(len(y[y == label]) / len(y))

        # Mean and standard deviation of each feature
        self.meanOfEachFeat: list[list] = []
        self.stdOfEachFeat: list[list] = []
        for label in range(self.classNum):
            mean, std = [], []
            for feat in range(self.featureNum):
                # Maximum Likelihood Estimation
                mean.append(np.mean(X[y == label, feat]))
                std.append(np.std(X[y == label, feat]))
            self.meanOfEachFeat.append(mean)
            self.stdOfEachFeat.append(std)


    def Predict(self, X) -> np.ndarray:
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.featureNum:
            raise Exception(f"Feature number mismatched. Expected {self.featureNum} features, got {X.shape[1]} features.")

        results = []
        for i in range(X.shape[0]):
            # Conditional probability (irrelevant to the denominator): P(x|y) = P(x1|y) * P(x2|y) * ... * P(xn|y)
            condProbs: list[float] = []
            for label in range(self.classNum):
                condProbs.append(1.0)
                for feat in range(self.featureNum):
                    mean = self.meanOfEachFeat[label][feat]
                    std = self.stdOfEachFeat[label][feat]
                    # Normal distribution
                    prob = 1.0 / (math.sqrt(2.0 * math.pi) * std) * math.exp(-((X[i][feat] - mean) ** 2) / (2.0 * (std ** 2)))
                    condProbs[label] *= prob
            
            # Total probability: P(x) = sum(P(x|y) * P(y))
            totalProb = sum(self.priorProbs[label] * condProbs[label] for label in range(self.classNum))

            # Posterior probability
            posteriorProbs: list[float] = []
            for label in range(self.classNum):
                posteriorProb = self.priorProbs[label] * condProbs[label] / totalProb
                posteriorProbs.append(posteriorProb)

            # To find the label with the minimum error rate in binary classification,
            # we need to find the label with the maximum posterior probability.
            bestProb = 0.0
            for label, prob in enumerate(posteriorProbs):
                if prob > bestProb:
                    bestLabel = label
                    bestProb = prob
            results.append(bestLabel)
        return np.array(results)
