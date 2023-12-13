import numpy as np

class Fisher:
    def __init__(self) -> None:
        self.featureNum: int = None
        self.classNum: int = None


    def Fit(self, X, y) -> None:
        X = np.array(X)
        y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.featureNum = X.shape[1]
        self.classNum = len(set(y))
        if self.classNum != 2:
            raise Exception("The classifier only supports binary classification.")

        # Mean vector of each class: m_i = 1/N * sum(x_j), where x_j is a sample of class i
        meanVectorOfEachClass = []
        for label in range(self.classNum):
            meanVector = np.mean(X[y == label], axis=0)
            meanVectorOfEachClass.append(meanVector)
        
        withinClassScatterMatrices = []
        for label in range(self.classNum):
            withinClassScatterMatrix = np.zeros((self.featureNum, self.featureNum))
            for sample in X[y == label]:
                sample, meanVector = sample.reshape(-1, 1), meanVectorOfEachClass[label].reshape(-1, 1)
                withinClassScatterMatrix += (sample - meanVector).dot((sample - meanVector).T)
            withinClassScatterMatrices.append(withinClassScatterMatrix)
        
        # Pooled within-class scatter matrix: S_w = S_1 + S_2
        pooledWithinClassScatterMatrix = np.zeros((self.featureNum, self.featureNum))
        for label in range(self.classNum):
            pooledWithinClassScatterMatrix += withinClassScatterMatrices[label]
        
        # Between-class scatter matrix: S_b = (m_1 - m_2)(m_1 - m_2)^T
        betweenClassScatterMatrix = (meanVectorOfEachClass[0] - meanVectorOfEachClass[1]).dot((meanVectorOfEachClass[0] - meanVectorOfEachClass[1]).T)

        # Fisher's linear discriminant: w = S_w^-1 * (m_1 - m_2)
        w = np.linalg.inv(pooledWithinClassScatterMatrix).dot(meanVectorOfEachClass[0] - meanVectorOfEachClass[1])





    

    def Predict(self, X) -> None:
        pass