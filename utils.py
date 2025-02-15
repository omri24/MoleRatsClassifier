import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix

# KNN
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.train_data = None
        self.train_labels = None

    def fit(self, X_train, y_train):
        self.train_data = X_train
        self.train_labels = y_train

    def predict(self, X_test):
        predictions = []
        for test_sample in X_test:
            similarities = cosine_similarity([test_sample], self.train_data)[0]
            top_k_indices = np.argsort(similarities)[-self.k:][::-1]  # Get k largest similarities
            top_k_labels = [self.train_labels[i] for i in top_k_indices]
            predicted_label = max(set(top_k_labels), key=top_k_labels.count)  # Majority vote
            predictions.append(predicted_label)
        return predictions

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return sum(np.array(predictions) == np.array(y_test)) / len(y_test)


class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def plot_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, class_names=None):
    """
    Plots a confusion matrix given prediction and label tensors.

    Args:
        preds (torch.Tensor): Predicted labels.
        labels (torch.Tensor): Ground truth labels.
        class_names (list, optional): List of class names for labeling axes.
    """
    # Convert tensors to numpy arrays
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def plot_confusion_matrix_percentage(preds: torch.Tensor, labels: torch.Tensor, class_names=None):
    """
    Plots a confusion matrix given prediction and label tensors with percentages.

    Args:
        preds (torch.Tensor): Predicted labels.
        labels (torch.Tensor): Ground truth labels.
        class_names (list, optional): List of class names for labeling axes.
    """
    # Convert tensors to numpy arrays
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  # Convert to percentage

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (%)")
    plt.show()

