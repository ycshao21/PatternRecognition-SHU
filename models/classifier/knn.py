import numpy as np

from .base_classifier import BaseClassifier

class KNN(BaseClassifier):
    def __init__(self, n_neighbors: int):
        """
        Initialize a KNN classifier.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to use.
        """
        super().__init__()
        self.X_train = None
        self.y_train = None
        self.n_neighbors = n_neighbors
    
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
        """
        self.X_train = self._convert_to_2D_array(X)
        self.y_train = y

        self.n_features = self.X_train.shape[1]
        labels = np.unique(y)
        n_classes = len(labels)

        if n_classes != 2:
            raise ValueError(
                "The classifier only supports binary classification for now."
            )
        self._check_labels(labels, n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the samples.

        Parameters
        ----------
        X : np.ndarray
            Samples to predict.
            Shape: (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels.
            Shape: (n_samples, ).
        """
        X = self._convert_to_2D_array(X)
        self._check_feature_number(X, self.n_features)

        y_pred = np.array(
            [self.predict_single(x) for x in X]
        )
        return y_pred
    
    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the label of a single sample.

        Parameters
        ----------
        x : np.ndarray
            Sample to predict.
            Shape: (n_features, ).

        Returns
        -------
        np.ndarray
            Predicted label.
        """
        # Calculate the distances between x and all the training samples.
        dists = np.square(self.X_train - x).sum(axis=1)
        # Sort distances in ascending order.
        sorted_indices = np.argsort(dists)
        # Get the labels of the k nearest neighbors.
        k_nearest_labels = self.y_train[sorted_indices[:self.n_neighbors]]
        # Count the number of each label.
        label_counts = np.bincount(k_nearest_labels)
        # Return the label with the most counts.
        return np.argmax(label_counts)
