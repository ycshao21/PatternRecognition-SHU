import numpy as np
import prutils.math as prmath
from .base_classifier import BaseClassifier
from prutils.math import distribution as dist


class MinimumErrorBayes(BaseClassifier):
    def __init__(
        self,
        prior_probs: None | list[float] = None,
        use_parzen: bool = False,
        kernel_name: str = "Gaussian",
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
        super().__init__()
        self.prior_probs: None | np.ndarray = prior_probs
        self.n_features: int = None
        self.n_classes: int = None

        # Not using parzen window >>>>>>
        self.feature_means: np.ndarray = None  # Shape: (n_classes, n_features)
        self.feature_vars: np.ndarray = None  # Shape: (n_classes, n_features)
        self.feature_cov: np.ndarray = (
            None  # Shape: (n_classes, n_features, n_features)
        )
        # <<<<<<

        # Using parzen window >>>>>>
        self.use_parzen: bool = use_parzen
        self.X_train: None | np.ndarray = None
        self.y_train: None | np.ndarray = None
        kernel_name = kernel_name.capitalize()
        if kernel_name == "Gaussian":
            self.kernel_func: callable = self.gaussian_kernel
        elif kernel_name == "Uniform":
            self.kernel_func: callable = self.uniform_kernel
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")
        # <<<<<<

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
        # If X is a vector, reshape it to a 2D array.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features = X.shape[1]
        self.n_classes = np.unique(y).shape[0]
        if self.n_classes != 2:
            raise ValueError(
                "The classifier only supports binary classification."
            )

        # Prior probability
        if self.prior_probs is None:
            _, counts = np.unique(y, return_counts=True)
            self.prior_probs = counts / np.sum(counts)

        if not self.use_parzen:
            # Mean and standard deviation of each feature
            self.feature_means = np.zeros(
                (self.n_classes, self.n_features), dtype=float
            )
            self.feature_vars = np.zeros(
                (self.n_classes, self.n_features), dtype=float
            )
            self.feature_cov = np.zeros(
                (self.n_classes, self.n_features, self.n_features), dtype=float
            )
            for i in range(self.n_classes):
                self.feature_means[i] = np.mean(X[y == i], axis=0)
                self.feature_vars[i] = np.var(X[y == i], axis=0)
                # [ISSUE] jamesnulliu
                #    Where is my conv function? Performance could be slow when `n_features` is large.
                self.feature_cov[i] = np.cov(X[y == i], rowvar=False)
        else:
            self.X_train = X
            self.y_train = y

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
        if not self.use_parzen:
            conditional_probs = np.prod(
                dist.normal_distribution(
                    sample,
                    mean=self.feature_means,
                    std=np.sqrt(self.feature_vars)
                ),
                axis=1,
            )
        else:
            conditional_probs = np.array(
                [
                    self.parzen_window(
                        sample,
                        self.X_train[self.y_train == i],
                        self.kernel_func,
                    )
                    for i in range(self.n_classes)
                ]
            )

        # Total probability: P(x) = sum(P(x|y) * P(y))
        total_probability = np.sum(
            self.prior_probs[i] * conditional_probs[i]
            for i in range(self.n_classes)
        )

        # Posterior probability: P(y|x) = P(x|y) * P(y) / P(x)
        posterior_probs = self.prior_probs * conditional_probs / total_probability
        return posterior_probs

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
        # If X is a 1D array, reshape it to a 2D array.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature number mismatched. Expected {self.n_features} features, got {X.shape[1]} features."
            )

        return np.argmax(
            np.array([self.cal_posterior_probs(sample) for sample in X]),
            axis=1,
        )


    @staticmethod
    def uniform_kernel(x: float):
        if abs(x) < 0.5:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def gaussian_kernel(x: float):
        return np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi)

    @staticmethod
    def parzen_window(x: np.ndarray, data: np.ndarray, kernel_func: callable):
        N = len(data)
        d = 1
        h = 1.06 * np.std(data) * N ** (-1 / (d + 4))
        V = h**d

        k = np.sum([kernel_func((xi - x) / h) for xi in data])
        # print("k:", k)
        return k / (N * V)

    def multivar_density(
        self, x: np.ndarray, which_class: int = None
    ) -> float | np.ndarray:
        if which_class is None:
            self.logger.warning(
                "The parameter `which_class` is not given."
                "All the classes will be considered."
                "This may lower the performance."
            )
            probs = np.array(
                [
                    self.multivar_density(x, which_class=i)
                    for i in range(self.n_classes)
                ]
            ) # Shape: (n_classes, )
            self.logger.debug(f"probs: {probs}")
            return probs
        else:
            cov = self.feature_cov[
                which_class
            ]  # Shape: (n_features, n_features)
            miu = self.feature_means[which_class]  # Shape: (n_features, )
            prob = (
                np.exp(-0.5 * (x - miu).T @ np.linalg.inv(cov) @ (x - miu))
                / (2 * np.pi) ** (self.n_features / 2)
                / np.sqrt(cov)
            ) # Shape: (1, 1)
            prob = prob[0][0]  # Scalar
            self.logger.debug(f"prob: {prob}")
            return prob
