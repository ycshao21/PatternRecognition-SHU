import numpy as np

import logging

class BaseClassifier:
    def __init__(self):
        self.logger = logging.getLogger("PRClassifier")

    def _convert_to_2D_array(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the given data to a 2D array.

        Parameters
        ----------
        X : np.ndarray
            Data to be converted.

        Returns
        -------
        np.ndarray
            Converted data.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _check_feature_number(self, X: np.ndarray, n_features: int) -> None:
        """
        Check whether the number of features of the given data is equal to the number of features of the training data.

        Parameters
        ----------
        X : np.ndarray
            Data to be checked, which must be a 2D array.

        Raises
        ------
        ValueError
            If the number of features of the given data is not equal to the number of features of the training data.
        """
        if X.shape[1] != n_features:
            raise ValueError(
                f"Feature number mismatched. Expected {self.n_features} features, got {X.shape[1]} features."
            )

    def _check_labels(self, labels: np.ndarray, n_classes: int) -> None:
        """
        Check whether the labels are valid.

        Parameters
        ----------
        y : np.ndarray
            Labels.

        Raises
        ------
        ValueError
            If the labels are not ordered from 0 to n_classes - 1.
        """
        if not np.array_equal(labels, np.arange(n_classes)):
            raise ValueError(
                "Labels must be ordered from 0 to n_classes - 1."
            )