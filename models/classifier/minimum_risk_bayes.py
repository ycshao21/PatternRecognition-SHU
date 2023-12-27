import numpy as np
from .base_bayes_classifier import BaseBayesClassifier


class MinimumRiskBayes(BaseBayesClassifier):
    def __init__(
        self,
        prior_probs: None | list[float] = None,
        kernel_name: None | str = None,
        h: float = 20,
        loss: np.ndarray = np.array([[0, 1], [1, 0]]),
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
        loss : np.ndarray, optional
            Loss matrix, by default np.array([[0, 1], [1, 0]]),
                in which case the classifier is the same as the minimum error classifier.
        """
        super().__init__(
            prior_probs=prior_probs,
            kernel_name=kernel_name,
            h=h
        )
        self.loss: np.ndarray = loss

    def cal_risks(self, sample: np.ndarray) -> np.ndarray:
        """
        Calculate the risks of the given sample.

        Parameters
        ----------
        sample : np.ndarray
            A vector of features.
            Shape: (n_features, ).

        Returns
        -------
        np.ndarray
            Risks of the given sample.
            Shape: (n_classes, ).
        """
        # Risk: R(a_i|x) = sum(P(y|x) * L(a_i, y)
        posterior_probs = self.cal_posterior_probs(sample)
        risk = np.dot(posterior_probs, self.loss)
        return risk

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
        X = self._convert_to_2D_array(X)
        self._check_feature_number(X, self.n_features)

        y_pred = np.argmin(
            np.array([self.cal_risks(sample) for sample in X]),
            axis=1
        )
        return y_pred