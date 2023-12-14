import numpy as np


def convariance_mat(
    mat: np.ndarray, rowvar: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conv = np.cov(mat, rowvar=rowvar)
    eigenvalues, eigenvectors = np.linalg.eig(conv)
    return (conv, eigenvalues, eigenvectors)


def PCA(mat: np.ndarray, n_components: int = 2) -> np.ndarray:
    if mat.shape[0] > mat.shape[1]:
        _, eigenvalues, eigenvectors = convariance_mat(mat, rowvar=False)
        eigenvectors = eigenvectors.T
    else:
        _, eigenvalues, eigenvectors = convariance_mat(mat, rowvar=True)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[idx]
    return np.dot(mat, eigenvectors[:, :n_components])