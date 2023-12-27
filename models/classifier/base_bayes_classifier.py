import numpy as np
from prutils.math import distribution as dist

from .base_classifier import BaseClassifier

class BaseBayesClassifier(BaseClassifier):
    def __init__(
        self,
        prior_probs: None | list[float] = None,
        kernel_name: str = "Gaussian",
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
        h : float, optional
            Parameter of the parzen window, by default 20
        """
        super().__init__()
        self.X_train: np.ndarray = None
        self.y_train: None | np.ndarray = None
        self.n_features: int = None
        self.n_classes: None | int = None

        self.prior_probs: None | np.ndarray = prior_probs

        # Not using parzen window >>>>>>
        self.feature_means: np.ndarray = None
        self.feature_vars: np.ndarray = None
        self.feature_cov: np.ndarray = None
        # <<<<<<

        # Using parzen window >>>>>>
        self.kernel_func: None | callable = None
        self.h = h

        kernel_dict = {
            "Gaussian": self.gaussian_kernel,
            "Uniform": self.uniform_kernel
        }
        if kernel_name is not None:
            try:
                self.kernel_func: callable = kernel_dict[kernel_name.capitalize()]
            except KeyError:
                self.logger.warning(
                    f"Unknown kernel: {kernel_name}. Using Gaussian kernel instead."
                )
                self.kernel_func: callable = kernel_dict["Gaussian"]
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
        self.X_train = self._convert_to_2D_array(X)
        self.y_train = y

        self.n_features = self.X_train.shape[1]
        labels, counts = np.unique(y, return_counts=True)
        self.n_classes = len(labels)

        if self.n_classes != 2:
            raise ValueError(
                "The classifier only supports binary classification for now."
            )
        self._check_labels(labels, self.n_classes)

        # If prior probabilities are not given, estimate them from the training data
        if self.prior_probs is None:
            _, counts = np.unique(y, return_counts=True)
            self.prior_probs = counts / np.sum(counts)

        if self.kernel_func is None:
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
                # [Reply To jamesnulliu] ycshao21 
                #    There are some logical mistakes and typos in your conv function.
                # Once they are corrected, the function will be added back.
                self.feature_cov[i] = np.cov(X[y == i], rowvar=False)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the given data.

        Parameters
        ----------
        X : np.ndarray
            Data to be predicted.
        """
        raise NotImplementedError

    def cal_conditional_probs(self, sample: np.ndarray) -> np.ndarray:
        """
        Calculate the conditional probabilities of a single sample.

        Parameters
        ----------
        sample : np.ndarray
            A vector of features.
            Shape: (n_features, ).

        Returns
        -------
        np.ndarray
            Conditional probabilities of the given sample.
            Shape: (n_classes, ).
        """
        if self.kernel_func is None:
            conditional_probs = np.prod(
                dist.normal_distribution(
                    sample,
                    mean=self.feature_means,
                    std=np.sqrt(self.feature_vars)
                ),
                # Shape: (n_classes, )
                axis=1,
            )
        else:
            conditional_probs = np.array(
                [
                    self.parzen_window(
                        sample,
                        self.X_train[self.y_train == i],
                        self.kernel_func,
                        self.h,
                    )
                    for i in range(self.n_classes)
                ]
            )
        return conditional_probs

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
        conditional_probs = self.cal_conditional_probs(sample)

        # Total probability: P(x) = sum(P(x|y) * P(y))
        total_probability = np.sum(
            self.prior_probs[i] * conditional_probs[i]
            for i in range(self.n_classes)
        )

        # Posterior probability: P(y|x) = P(x|y) * P(y) / P(x)
        posterior_probs = self.prior_probs * conditional_probs / total_probability
        return posterior_probs

    
    @staticmethod
    def uniform_kernel(x: float):
        if abs(x) < 0.5:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def gaussian_kernel(x: float | np.ndarray):
        if isinstance(x, float):
            x = np.array([x])
        return np.prod(np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi))

    @staticmethod
    def parzen_window(
        x: np.ndarray, data: np.ndarray, kernel_func: callable, h: float
    ) -> float:
        N = data.shape[0]
        d = data.shape[1]
        # h = 1.06 * np.std(data) * N ** (-1 / (d + 4))
        V = h**d
        if kernel_func.__name__ == "gaussian_kernel":
            sigma = h / np.sqrt(N)
            k = np.sum(np.array([kernel_func((xi - x) / sigma) for xi in data]))
            return k / N
        else:
            k = np.sum([kernel_func((xi - x) / h) for xi in data])
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
            )  # Shape: (n_classes, )
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
            )  # Shape: (1, 1)
            prob = prob[0][0]  # Scalar
            self.logger.debug(f"prob: {prob}")
            return prob
