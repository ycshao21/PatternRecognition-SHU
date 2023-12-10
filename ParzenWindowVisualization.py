import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

################### Parzen Window ###################

def UniformKernel(x: float) -> float:
    if abs(x) < 0.5:
        return 1.0
    else:
        return 0.0

def GaussianKernel(x: float) -> float:
    return np.exp(-x ** 2 / 2.0) / np.sqrt(2.0 * math.pi)

def ParzenWindow(x: float, data: np.ndarray, kernelFn: callable) -> float:
    if data.ndim != 1:
        raise ValueError("Data should be 1D array.")

    n = len(data)
    d = 1
    h = 1.06 * np.std(data) * n ** (-1 / (d + 4))
    V = h ** d

    k = sum([kernelFn((xi - x) / h) for xi in data])
    return k / (n * V)


if __name__ == '__main__':
    data = pd.read_csv("Dataset/genderData_Merged.csv")
    X = data[['身高(cm)', '体重(kg)']].values.astype(float)
    y = data['性别'].values.astype(int)

    # Draw plot
    pdfs = []
    for label in (0, 1):
        X_Label = X[y == label]
        X_Mesh, y_Mesh = np.meshgrid(np.linspace(130, 220, 100), np.linspace(20, 100, 100))

        heights_pdf = np.zeros(X_Mesh.shape)
        weights_pdf = np.zeros(y_Mesh.shape)
        for i in range(X_Mesh.shape[0]):
            for j in range(X_Mesh.shape[1]):
                heights_pdf[i, j] = ParzenWindow(X_Mesh[i, j], X_Label[:, 0], GaussianKernel)
                weights_pdf[i, j] = ParzenWindow(y_Mesh[i, j], X_Label[:, 1], GaussianKernel)
        
        heights_pdf = heights_pdf / np.sum(heights_pdf)
        weights_pdf = weights_pdf / np.sum(weights_pdf)
        pdf = heights_pdf * weights_pdf
        pdfs.append(pdf)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X_Mesh, y_Mesh, pdfs[0], cmap='autumn', alpha=0.6)
    ax.plot_surface(X_Mesh, y_Mesh, pdfs[1], cmap='Blues', alpha=0.6)

    ax.set_title(f'Parzen Window')
    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Probability Density')

    plt.show()
