import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def PrintAccuracy(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def DisplayConfusionMatrix(y_pred: np.ndarray, y_true: np.ndarray, plot: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred)

    if not plot:
        print("Confusion Matrix:")
        print(cm)
    else:
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.ylabel('Number of samples', rotation=-90, va="bottom")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", 
                    color="white" if cm[i, j] > cm.max() / 2. else "black"
                )
        
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()
