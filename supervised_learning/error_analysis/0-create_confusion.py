#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters:
    labels (numpy.ndarray): A one-hot numpy.ndarray of shape (m, classes) containing the correct labels for each data point.
    logits (numpy.ndarray): A one-hot numpy.ndarray of shape (m, classes) containing the predicted labels.

    Returns:
    numpy.ndarray: A confusion numpy.ndarray of shape (classes, classes) with row indices representing the correct labels and column indices representing the predicted labels.
    """
    if labels.shape != logits.shape:
        raise ValueError("Labels and logits must have the same shape")

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)

    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    for true, pred in zip(true_labels, predicted_labels):
        confusion[true, pred] += 1

    return confusion


# Example main file to demonstrate usage
if __name__ == "__main__":
    # Load labels and logits from the provided npz file
    data = np.load('labels_logits.npz')
    labels = data['labels']
    logits = data['logits']

    # Create confusion matrix
    confusion_matrix = create_confusion_matrix(labels, logits)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix)
