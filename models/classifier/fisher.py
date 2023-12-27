import numpy as np

class Fisher:
    def __init__(self):
        """Initialize the classifier."""
        self.n_features: int = None
        self.n_classes: int = None
        self.projection: np.ndarray = None
        self.threshold: float = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier.

        Parameters
        ----------
        X : np.ndarray
            Training data.
            Shape: (n_samples, n_features).
        y : np.ndarray
            Labels.
            Shape: (n_samples, ).

        Raises
        ------
        ValueError
            If the number of classes is not equal to 2.
        """
        self.n_features = X.shape[1]
        self.n_classes = len(set(y))
        if self.n_classes != 2:
            raise ValueError(
                "The classifier only supports binary classification."
            )

        # Mean vector of each class: m_i = 1/N * sum(x_j),
        # where i is the class label,
        #       N is the number of samples of class i,
        #       x_j is a sample of class i,
        #       m_i is a d-dimensional vector, d is the number of features.
        class_means = np.array(
            [
                np.mean(X[y == i], axis=0)
                for i in range(self.n_classes)
            ]
        )
        mean_diff: np.ndarray = class_means[0] - class_means[1]

        # Within-class scatter matrix: S_w = sum(S_i),
        # where S_i = sum((x_j - m_i) * (x_j - m_i).T), a d*d matrix, j is the sample index,
        #       S_w is also a d*d matrix.
        S_w = np.sum(
            [
                np.dot((X[y == i] - class_means[i]).T, X[y == i] - class_means[i])
                for i in range(self.n_classes)
            ],
            axis=0
        )

        # Best projection direction: w = S_w^-1 * (m_1 - m_2),
        # where w is a d-dimensional vector.
        self.projection = np.dot(np.linalg.inv(S_w), mean_diff)

        # Projected data: y = w.T * x,
        means_projected = np.dot(self.projection, class_means.T)

        # Calculate the threshold: w_0 = (m_1 + m_2) / 2
        self.threshold = np.mean(means_projected)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the given data.

        Parameters
        ----------
        X : np.ndarray
            Data to be predicted.

        Returns
        -------
        np.ndarray
            Predicted labels of the given data.

        Raises
        ------
        ValueError
            If the number of features of the given data is not equal to the number of features of the training data.
        """
        # If X is a vector, reshape it to a 2D array.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature number mismatched. Expected {self.n_features} features, got {X.shape[1]} features."
            )

        # If y = w.T * x > w_0, then x belongs to class 0,
        # else x belongs to class 1.
        y_projected = np.dot(self.projection, X.T)
        y_pred = np.where(y_projected > self.threshold, 0, 1)
        return y_pred
