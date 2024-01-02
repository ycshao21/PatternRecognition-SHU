import numpy as np

class BaseDecomposition:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.eigenvectors = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.eigenvectors



class PCA(BaseDecomposition):
    def __init__(self, n_components: int = 2):
        super().__init__(n_components)
    
    def fit(self, X: np.ndarray) -> None:
        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.eigenvectors = eigenvectors[:, idx]
       
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class FLDA(BaseDecomposition):
    def __init__(self, n_components: int = 2):
        super().__init__(n_components)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError(
                "It only supports binary classification."
            )
        
        # Mean vector of each class: m_i = 1/N * sum(x_j),
        # where i is the class label,
        #       N is the number of samples of class i,
        #       x_j is a sample of class i,
        #       m_i is a d-dimensional vector, d is the number of features.
        class_means = np.array(
            [
                np.mean(X[y == i], axis=0)
                for i in range(n_classes)
            ]
        )
        mean_diff: np.ndarray = class_means[0] - class_means[1]

        # Within-class scatter matrix: S_w = sum(S_i),
        # where S_i = sum((x_j - m_i) * (x_j - m_i).T), a d*d matrix, j is the sample index,
        #       S_w is also a d*d matrix.
        S_w = np.sum(
            [
                np.dot((X[y == i] - class_means[i]).T, X[y == i] - class_means[i])
                for i in range(n_classes)
            ],
            axis=0
        )

        # Between-class scatter matrix: S_b = (m_1 - m_2) * (m_1 - m_2).T,
        # where S_b is a d*d matrix.
        S_b = np.dot(mean_diff, mean_diff.T)

        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(S_w), S_b))
        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.eigenvectors = eigenvectors[:, idx]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)