import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import PIL
import os

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
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.n_neighbors = n_neighbors
        self.n_classes: int = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
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
        self.n_classes = len(labels)

        if self.n_classes != 2:
            raise ValueError(
                "The classifier only supports binary classification for now."
            )
        self._check_labels(labels, self.n_classes)

    def _fit_multi_edit(self, X: np.ndarray, y: np.ndarray, s: int, **kwargs) -> None:

        n_samples = X.shape[0]
        if s < 1 or s > n_samples:
            raise ValueError("s must be between 1 and n_samples.")
        # Shuffle and split X to s parts
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        X_parts = np.array_split(X, s)
        y_parts = np.array_split(y, s)

        whether_visualize = kwargs.get("visualize", True)
        if whether_visualize:
            # >>>>>> Preparation for the animation >>>>>>
            axis_0_min, axis_1_max = X[:, 0].min(), X[:, 0].max()
            axis_1_min, axis_1_max = X[:, 1].min(), X[:, 1].max()
            temp_frame_dir = kwargs.get("temp_frame_dir", "./temp_frames")
            os.makedirs(temp_frame_dir, exist_ok=True)
            # update_frame = lambda i: PIL.Image.open(f"{temp_frame_dir}/frame_{i}.png")
            def update_frame(i):
                img = PIL.Image.open(f"{temp_frame_dir}/frame_{i}.png")
                ax.imshow(img)
                plt.tight_layout()
                plt.axis("off")
            colors = np.array(["#2ca02c", "#9467bd", "#8c564b"])
            # <<<<<< Preparation for the animation <<<<<<

        # Iteration to edit samples
        current_s = 0
        epoch = 0
        continuous_no_misclassfied_count = 0
        target_no_misclassified_count = kwargs.get("target_no_misclassified_count", 3)
        while True:
            # Part `it` is the part to edit
            X_edit, y_edit = X_parts[current_s], y_parts[current_s]
            # Other parts are used as training data
            X_train = np.concatenate(X_parts[:current_s] + X_parts[current_s + 1 :])
            y_train = np.concatenate(y_parts[:current_s] + y_parts[current_s + 1 :])
            # Fit the classifier
            self.fit(X_train, y_train)
            # Predict the labels of the samples to edit
            y_pred = self.predict(X_edit)
            # Find the samples that are misclassified
            misclassified_indices = np.where(y_pred != y_edit)[0]

            if whether_visualize:
                # Initial frame
                fig = plt.figure(figsize=(8, 6), dpi=100)
                ax = plt.axes()
                ax.set_xlim(axis_0_min, axis_1_max)
                ax.set_ylim(axis_1_min, axis_1_max)
                ax.set_title("KNN")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal")
                # Trian data: o, Test data: x, Misclassified: square, Color: base on label
                ax.scatter(
                    X_train[:, 0],
                    X_train[:, 1],
                    c=colors[y_train],
                    marker="o",
                    label="Training Data",
                    s=20,
                )
                ax.scatter(
                    X_edit[:, 0],
                    X_edit[:, 1],
                    c=colors[y_edit],
                    marker="s",
                    label="Test Data",
                    s=20,
                )
                ax.scatter(
                    X_edit[misclassified_indices, 0],
                    X_edit[misclassified_indices, 1],
                    c="r",
                    marker="x",
                    label="Misclassified",
                    s=20
                )
                plt.legend()
                # Save the initial frame
                fig.savefig(f"{temp_frame_dir}/frame_{epoch}.png")
                plt.close()

            # Update the number of continuous iterations that have no misclassified samples
            continuous_no_misclassfied_count = (
                continuous_no_misclassfied_count + 1
                if len(misclassified_indices) == 0
                else 0
            )
            if continuous_no_misclassfied_count == target_no_misclassified_count:
                break
            # Drop the misclassified samples
            X_parts[current_s] = np.delete(X_parts[current_s], misclassified_indices, 0)
            y_parts[current_s] = np.delete(y_parts[current_s], misclassified_indices, 0)
            current_s = (current_s + 1) % s
            epoch += 1

        if whether_visualize:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            ani = animation.FuncAnimation(
                fig, update_frame, frames=epoch+1, interval=1000, repeat=True
            )
            ani.save("animation.gif", writer="imagemagick", fps=1)

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

        y_pred = np.array([self._predict_single(x) for x in X])
        return y_pred

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        X = self._convert_to_2D_array(X)
        self._check_feature_number(X, self.n_features)

        probs = np.array([self._predict_single_prob(x) for x in X])
        return probs

    def _predict_single_prob(self, x: np.ndarray) -> list:
        # Calculate the distances between x and all the training samples.
        dists = np.square(self.X_train - x).sum(axis=1)
        # Sort distances in ascending order.
        sorted_indices = np.argsort(dists)
        # Get the labels of the k nearest neighbors.
        k_nearest_labels = self.y_train[sorted_indices[: self.n_neighbors]]
        # Count the number of each label.
        label_counts = np.bincount(k_nearest_labels)
        # If label_counts is less than n_classes, pad it with zeros.
        if len(label_counts) < self.n_classes:
            label_counts = np.pad(
                label_counts,
                (0, self.n_classes - len(label_counts)),
                "constant",
                constant_values=0,
            )
        return (label_counts / self.n_neighbors).tolist()

    def _predict_single(self, x: np.ndarray) -> np.ndarray:
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
        k_nearest_labels = self.y_train[sorted_indices[: self.n_neighbors]]
        # Count the number of each label.
        label_counts = np.bincount(k_nearest_labels)
        # Return the label with the most counts.
        return np.argmax(label_counts)
