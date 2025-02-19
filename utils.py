import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

class DropoutMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, drop_out_param):
        super(DropoutMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(drop_out_param),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(drop_out_param),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        return self.classifier(x)


class BatchNormMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, drop_out_param):
        super(BatchNormMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(drop_out_param),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(drop_out_param),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, output_size)
        )

    def forward(self, x):
        return self.classifier(x)


class MoleRatsLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(MoleRatsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x.unsqueeze(1))  # Add sequence dimension
        return self.fc(h_n[-1])  # Use last hidden state


def train_LSTM(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_LSTM(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(test_loader), accuracy

def save_mlp_weights(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")

def load_mlp_weights(model, file_path):
    model.load_state_dict(torch.load(file_path, weights_only=True))
    print(f"Model weights loaded from {file_path}")

def save_lstm_weights(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"LSTM model weights saved to {file_path}")

def load_lstm_weights(model, file_path):
    model.load_state_dict(torch.load(file_path, weights_only=True))
    print(f"LSTM model weights loaded from {file_path}")


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

# Convert PrettyTable to Markdown
def prettytable_to_markdown(table):
    # Extract column headers
    headers = table.field_names
    # Extract all rows
    rows = table._rows

    # Convert headers to Markdown format
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "|-" + "-|-".join(["-" * len(header) for header in headers]) + "-|\n"

    # Convert rows to Markdown format
    for row in rows:
        md_table += "| " + " | ".join(map(str, row)) + " |\n"

    return md_table