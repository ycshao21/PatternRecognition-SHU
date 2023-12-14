import math
import numpy as np


class MinimumErrorBayes:
    def __init__(
        self,
        prior_probs: None | list[float] = None,
        use_parzen: bool = False,
        kernel="Gaussian",
    ) -> None:
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
        self.prior_probs: None | np.ndarray = prior_probs
        self.n_features: int = None
        self.n_classes: int = None

        self.feature_means: np.ndarray = None
        self.feature_vars: np.ndarray = None

        self.useParzen: bool = use_parzen
        self.X_train: None | np.ndarray = None
        self.y_Train: None | np.ndarray = None
        if kernel == "Gaussian":
            self.kernel_func: callable = self.GaussianKernel
        elif kernel == "Uniform":
            self.kernel_func: callable = self.UniformKernel

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
        Exception
            If the number of classes is greater than 2.
        """

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features = X.shape[1]
        self.n_classes = np.unique(y).shape[0]
        if self.n_classes != 2:
            raise Exception(
                "The classifier only supports binary classification."
            )

        # Prior probability
        if self.prior_probs is None:
            _, counts = np.unique(y, return_counts=True)
            self.prior_probs = counts / np.sum(counts)

        if not self.useParzen:
            # Mean and standard deviation of each feature
            self.feature_means = np.zeros(
                (self.n_classes, self.n_features), dtype=float
            )
            self.feature_vars = np.zeros(
                (self.n_classes, self.n_features), dtype=float
            )
            for i in range(self.n_classes):
                self.feature_means[i] = np.mean(X[y == i], axis=0)
                self.feature_vars[i] = np.var(X[y == i], axis=0)
        else:
            self.X_train = X
            self.y_Train = y

    def cal_posterior_probs(self, sample: np.ndarray) -> np.ndarray:
        """
        Calculate the posterior probabilities of the given sample.

        Parameters
        ----------
        sample : np.ndarray
            A vector of features.
            Shape: (n_features, ).

        Returns
        -------
        np.ndarray
            Posterior probabilities of the given sample.
            Shape: (n_classes, ).
        """
        if not self.useParzen:
            conditional_probs = np.prod(
                np.exp(
                    -((sample - self.feature_means) ** 2)
                    / (2.0 * self.feature_vars)
                )
                / np.sqrt(2.0 * np.pi * self.feature_vars),
                axis=1,
            )
        else:
            conditional_probs = np.prod(
                np.array(
                    [
                        self.ParzenWindow(
                            sample,
                            self.X_train[self.y_Train == i],
                            self.kernel_func,
                        )
                        for i in range(self.n_classes)
                    ]
                ),
                axis=1,
            )

        # Total probability: P(x) = sum(P(x|y) * P(y))
        total_probability = np.sum(
            self.prior_probs[label] * conditional_probs[label]
            for label in range(self.n_classes)
        )

        # Posterior probability: P(y|x) = P(x|y) * P(y) / P(x)
        posterior_probs = (
            self.prior_probs * conditional_probs / total_probability
        )
        return posterior_probs

    def predict(self, X) -> np.ndarray:
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
        Exception
            If the number of features of the given data is not equal to the number of features of the training data.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features:
            raise Exception(
                f"Feature number mismatched. Expected {self.n_features} features, got {X.shape[1]} features."
            )

        y_pred = np.argmax(
            np.array([self.cal_posterior_probs(sample) for sample in X]),
            axis=1,
        )

        return y_pred

    @staticmethod
    def UniformKernel(x: float):
        if abs(x) < 0.5:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def GaussianKernel(x: float):
        return np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * math.pi)

    @staticmethod
    def ParzenWindow(x: np.ndarray, data: np.ndarray, kernelFn: callable):
        N = len(data)
        d = 1
        h = 1.06 * np.std(data) * N ** (-1 / (d + 4))
        V = h**d

        k = sum([kernelFn((xi - x) / h) for xi in data])
        return k / (N * V)
