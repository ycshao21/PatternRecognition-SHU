import numpy as np

def normal_distribution(
    x: float | np.ndarray, mean: float | np.ndarray, std: float | np.ndarray
) -> np.ndarray:
    """
    Calculate the normal distribution. 

    Formula:
        `f(x) = exp(-0.5 * ((x - mean) / std) ** 2) / (std * sqrt(2 * pi))`

    Parameters
    ----------
    x : np.ndarray
        Input vector.
    mean : np.ndarray
        Mean of the normal distribution.
    std : np.ndarray
        Std of the normal distribution.

    Returns
    -------
    np.ndarray
        Normal distribution.
        Shape: (n_samples, n_features)

    Examples
    --------
    Suppose `x` is a vector with shape (3, ), `mean` is a scalar and `std` is a scalar. \\
    To calculate the normal distribution of each element in `x`, we can do:
    >>> x = np.array([0, 1, 2])
    >>> mean, std = 3, 2
    >>> normal_distribution(x, mean, std)

    Suppose `x` is a matrix with shape (3, 2), `mean` is a scalar and `std` is a scalar. \\
    To calculate the normal distribution of each element in `x`, we can do:
    >>> x = np.array([[0, 1], [2, 3], [4, 5]])
    >>> mean, std = 3, 2
    >>> normal_distribution(x, mean, std)

    Suppose `n_samples` is 3, `n_features` is 2; \\
    `x` is a matrix with shape (3, 2) where each row is a sample and each column is a feature; \\
    `mean` is a vector with shape (2, ) where each element is the mean of the corresponding feature; \\
    `std` is a vector with shape (2, ) where each element is the std of the corresponding feature. \\
    To calculate the normal distribution of each feature for each sample in `x`, we can do:
    >>> x = np.array([[0, 1], [2, 3], [4, 5]])
    >>> mean, std = np.array([3, 4]), np.array([2, 3])
    >>> normal_distribution(x, mean, std)
    """
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
