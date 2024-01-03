import numpy as np
from .base_bayes_classifier import BaseBayesClassifier

class MinimumErrorBayes(BaseBayesClassifier):
    def __init__(
        self,
        prior_probs: None | list[float] = None,
        kernel_name: str = None,
        h: float = 20
    ):
        """
        Initialize the classifier.

        Parameters
        ----------
        prior_probs : None | list[float], optional
            Prior probabilities, by default None
        use_parzen : bool, optional
            Whether to use parzen window, by default False
        kernel_name : str, optional
            Type of the kernel function of the parzen window, by default 'Gaussian'
            Kernels available: 'Gaussian', 'Uniform'
        """
        super().__init__(
            prior_probs=prior_probs,
            kernel_name=kernel_name,
            h=h
        )

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
        """
        X = self._convert_to_2D_array(X)
        self._check_feature_number(X, self.n_features)

        y_pred = np.argmax(
            np.array([self.cal_posterior_probs(sample) for sample in X]),
            axis=1,
        )
        return y_pred
