import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
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

    def fit(
        self, X: np.ndarray, y: np.ndarray, method: str = "basic", **kwargs
    ) -> None:
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
        method : str
            Method to use. Default: "basic".
            "basic": Basic KNN.
            "multi-edit": Multi-edit KNN.

        Keyword Arguments
        -----------------
        split : int
            Number of parts to split the training data.
            ONLY used when `method` is "multi-edit".
        target_count_of_no_misclassified : int
            Target count of continuous iterations that have no misclassified samples.
            ONLY used when `method` is "multi-edit".
        whether_visualize : bool
            Whether to visualize the process.
            ONLY used when `method` is "multi-edit".
        """
        self.method = method
        if self.method == "basic":
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
        elif self.method == "multi-edit":
            self._fit_multi_edit(X, y, **kwargs)

    def _fit_multi_edit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split: int,
        target_count_of_no_misclassified: int = 3,
        temp_frame_dir: str = "temp_frames",
        whether_visualize: bool = True,
        outputdir: str = "outputs/",
        **kwargs,
    ) -> None:
        n_samples = X.shape[0]
        if split < 1 or split > n_samples:
            raise ValueError("s must be between 1 and n_samples.")
        # Shuffle and split X to s parts
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        X_parts = np.array_split(X, split)
        y_parts = np.array_split(y, split)

        if whether_visualize:
            # >>>>>> Preparation for the animation >>>>>>
            axis_0_min, axis_0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            axis_1_min, axis_1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            # [ToDo]: Extend the color list for multi-class classification
            scatter_colors = np.array(["#183E0C", "#58135E", "#8c564b"])
            decision_boundary_bkg_color = ListedColormap(["#DEF2FF", "#FFFED8"])
            # Remove `temp_frame_dir` if it exist
            if os.path.exists(temp_frame_dir):
                if os.name == "posix":
                    os.system(f"rm -rf {temp_frame_dir}")
                elif os.name == "nt":
                    os.system(f"rmdir /s /q {temp_frame_dir}")
            # Create `temp_frame_dir` to store frames
            os.makedirs(temp_frame_dir, exist_ok=True)
            # Plot the decision boundaries

            # This function is tool for getting and plotting each frame in the dir
            def update_frame(i):
                img = PIL.Image.open(f"{temp_frame_dir}/frame_{i}.png")
                ax.imshow(img)
                # Print the frame number
                ax.text(
                    axis_0_min,
                    axis_1_max,
                    f"Frame: {i}",
                    bbox=dict(facecolor="white", alpha=1),
                )
                plt.tight_layout()
                plt.axis("off")

            # <<<<<< Preparation for the animation <<<<<<

        # Iteration to edit samples
        current_s = 0
        epoch = 0
        continuous_count_of_no_misclassfied = 0
        while True:
            # Part `it` is the part to edit
            X_edit, y_edit = X_parts[current_s], y_parts[current_s]
            # Other parts are used as training data
            X_train = np.concatenate(X_parts[:current_s] + X_parts[current_s + 1 :])
            y_train = np.concatenate(y_parts[:current_s] + y_parts[current_s + 1 :])
            # Fit the classifier
            self.fit(
                X_train,
                y_train,
            )
            # Predict the labels of the samples to edit
            y_pred = self.predict(X_edit)
            # Find the samples that are misclassified
            misclassified_indices = np.where(y_pred != y_edit)[0]

            if whether_visualize:
                # Create a frame for current iteration
                fig = plt.figure(figsize=(6, 6), dpi=200)

                ax = plt.axes()
                ax.set_xlim(axis_0_min, axis_0_max)
                ax.set_ylim(axis_1_min, axis_1_max)
                ax.set_title(f"KNN (n_neighbors={self.n_neighbors})")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal")

                # Create a meshgrid to plot decision boundaries
                h = 0.02  # Step size in the mesh
                xx, yy = np.meshgrid(
                    np.arange(axis_0_min, axis_0_max, h),
                    np.arange(axis_1_min, axis_1_max, h),
                )

                # Get predictions for each point in the meshgrid
                Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=decision_boundary_bkg_color, alpha=1)

                # Trian data: o, Test data: ^, Misclassified: square, Color: base on label
                ax.scatter(
                    X_train[:, 0],
                    X_train[:, 1],
                    facecolors="none",
                    edgecolors=scatter_colors[y_train],
                    marker="o",
                    s=15,
                )
                ax.scatter(
                    X_edit[:, 0],
                    X_edit[:, 1],
                    facecolors="none",
                    edgecolors=scatter_colors[y_edit],
                    marker="^",
                    s=15,
                )
                ax.scatter(
                    X_edit[misclassified_indices, 0],
                    X_edit[misclassified_indices, 1],
                    c="r",
                    marker="x",
                    s=30,
                )
                class_labels = ["Boy", "Girl", "Splited Train", "Splited Test (Edit)", "Misclassified"]
                markersize=6
                legend_elements = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="s",
                        color=scatter_colors[0],
                        label=class_labels[0],
                        markersize=markersize,
                        linestyle="None"
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="s",
                        color=scatter_colors[1],
                        label=class_labels[1],
                        markersize=markersize,
                        linestyle="None"
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        markerfacecolor="none",
                        markeredgecolor="black",
                        label=class_labels[2],
                        markersize=markersize,
                        linestyle="None"
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="^",
                        markerfacecolor="none",
                        markeredgecolor="black",
                        label=class_labels[3],
                        markersize=markersize,
                        linestyle="None"
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="x",
                        color="r",
                        label=class_labels[4],
                        markersize=markersize,
                        linestyle="None"
                    ),
                ]

                plt.legend(handles=legend_elements, fontsize="small", loc="upper left")
                # Save the initial frame
                fig.savefig(f"{temp_frame_dir}/frame_{epoch}.png")
                plt.close()

            # Update the number of continuous iterations that have no misclassified samples
            continuous_count_of_no_misclassfied = (
                continuous_count_of_no_misclassfied + 1
                if len(misclassified_indices) == 0
                else 0
            )
            if continuous_count_of_no_misclassfied == target_count_of_no_misclassified:
                break
            # Drop the misclassified samples
            X_parts[current_s] = np.delete(X_parts[current_s], misclassified_indices, 0)
            y_parts[current_s] = np.delete(y_parts[current_s], misclassified_indices, 0)
            current_s = (current_s + 1) % split
            epoch += 1

        # Overall fitting
        X_train = np.concatenate(X_parts)
        y_train = np.concatenate(y_parts)
        self.fit(X_train, y_train, method="basic")

        # Save the animation
        if whether_visualize:
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
            ani = animation.FuncAnimation(
                fig, update_frame, frames=epoch + 1, interval=1000, repeat=True
            )
            ani.save(f"{outputdir}animation_{self.n_neighbors}.gif", writer="pillow", fps=1)

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
