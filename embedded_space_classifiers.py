import re
import os
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from utils import KNNClassifier, FullyConnectedNN, plot_confusion_matrix, plot_confusion_matrix_percentage

# Bulid the training and testing sets
labels = []

path_to_embeddings = "Embeddings"

lst_of_embedding_paths = [os.path.join(path_to_embeddings, item) for item in os.listdir(path_to_embeddings)]

lst_of_embedding_arrys = [np.loadtxt(item, delimiter=",", dtype=np.float32) for item  in lst_of_embedding_paths]

print("For general knowledge: each item in lst_of_embedding_paths must end with 'label{label_number}' (see examples) otherwise it won't work")

t_lst_labels_int = [int(item[re.search("label", item).end():re.search("\.csv", item).start()]) for item in lst_of_embedding_paths]
t_lst_labels_amount = [item.shape[0] for item in lst_of_embedding_arrys]

for idx in range(len(t_lst_labels_int)):
  item = t_lst_labels_int[idx]
  amount = t_lst_labels_amount[idx]
  for i in range(amount):
    labels.append(item)

embeddings_arr = np.concatenate(lst_of_embedding_arrys, axis=0)

lst_arr_for_train = []
lst_arr_for_test = []
lst_labels_for_train = []
lst_labels_for_test = []
mod_param_for_train_test_split = 5  # for 0.8/.02 split, choose: 5
for row in range(embeddings_arr.shape[0]):
  if row % mod_param_for_train_test_split == 0:
    lst_arr_for_test.append(embeddings_arr[row: row + 1, :])
    lst_labels_for_test.append(labels[row])
  else:
    lst_arr_for_train.append(embeddings_arr[row: row + 1, :])
    lst_labels_for_train.append(labels[row])

embeddings_train = np.concatenate(lst_arr_for_train, axis=0)
embeddings_test = np.concatenate(lst_arr_for_test, axis=0)

train_set = torch.from_numpy(embeddings_train)
test_set = torch.from_numpy(embeddings_test)

train_labels = deepcopy(lst_labels_for_train)
test_labels = deepcopy(lst_labels_for_test)


# Define the label alias
"""
--- label alias explained:
    The labels of the data are according to: odd -> BMR with | even -> noise from the relevant recording.
    for example, if we want to classify all noise to one class, we must define label alias to map all even labels to the same class.
    The origial labels match the following BMR: {1: Abe, 3: Phoenix, 5: Abe, 7: Bubba, 9: Bubba, 11: Tulsi, 13: Abe, 15: Little}
"""
label_alias0 = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:11, 13:12, 14:13, "n_labels": 14}
label_alias1 = {1:1, 2:0, 3:2, 4:0, 5:1, 6:0, 7:3, 8:0, 9:3, 10:0, 11:4, 12:0, 13:1, 14:0, 15:5, 16:0, 17:4, 18:0, "n_labels": 6}
label_alias2 = {1:1, 2:5, 3:2, 4:6, 5:1, 6:7, 7:3, 8:8, 9:3, 10:9, 11:4, 12:10, 13:1, 14:0, "n_labels": 11}

# Configurations
label_alias = None
label_alias_to_use = 1
n_labels = -1
run_knn = False
show_svm_report = False
show_confusion_mat = False

exec(f"label_alias = label_alias{label_alias_to_use}")
train_labels = [label_alias[i] for i in train_labels]
test_labels = [label_alias[i] for i in test_labels]
n_labels = label_alias["n_labels"]

# Kmeans
n_clusters = n_labels
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(embeddings_arr)
labels_estimation = kmeans.labels_

count_correct = 0
for i in range(len(labels)):
  if labels[i] == labels_estimation[i]:
    count_correct += 1

accuracy = count_correct / len(labels)
print(f"\nAccuracy of WavML with KMeans: {round(accuracy, 4)}")

# KNN
if run_knn:
    X_train = embeddings_train.copy()
    y_train = deepcopy(train_labels)
    X_test = embeddings_test.copy()
    y_test = deepcopy(test_labels)
    k = 3

    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)

    print(f"Accuracy of WavLM with KNN (k = {k}): {round(accuracy, 4)}")

# SVM (number of classes is detected automatically according to the labels lst)
train_samples = embeddings_train.copy()
test_samples = embeddings_test.copy()
kernel = 'rbf'

svm_model = SVC(kernel=kernel, C=1.0, gamma='scale')
svm_model.fit(train_samples, train_labels)
test_predictions = svm_model.predict(test_samples)

accuracy = accuracy_score(test_labels, test_predictions)
print(f"Accuracy of SVM with {kernel} kernel: {round(accuracy, 4)}")
if show_svm_report:
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, zero_division=0))

# FC-NN (1 hidden layer)
input_size = 512
hidden_size = 256
output_size = n_labels
learning_rate = 0.001
num_epochs = 1500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = train_set.clone().to(device)
y = torch.tensor(train_labels).to(device)

# Define the model, the optimizer, and the loss
model = FullyConnectedNN(input_size, hidden_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Select loss function
use_cross_entropy = True    # Change this to False to use L2 loss

if use_cross_entropy:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()
    y = F.one_hot(y, num_classes=output_size).float()

# Training loop
print("FC-NN training started")
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    if not use_cross_entropy:
        outputs = F.softmax(outputs, dim=1)

    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete")

# Test
test_set = test_set.to(device)
test_labels = torch.tensor(test_labels).to(device)
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_outputs = model(test_set)
    if not use_cross_entropy:
        test_outputs = F.softmax(test_outputs, dim=1)  # Apply softmax for L2 loss

    if use_cross_entropy:
        predicted = torch.argmax(test_outputs, dim=1)
        accuracy = (predicted == test_labels).float().mean()
        print(f"Test Accuracy (cross entropy loss and 1 hidden layer): {round(accuracy.item(), 4)}")
    else:
        mse = criterion(test_outputs, test_labels)
        print(f"Test Mean Squared Error: {mse.item():.4f}")

print(f"\nThe data was classified {n_labels} classes.")
labels_count = {}
for item in train_labels:
  if item not in labels_count.keys():
    labels_count[item] = 0
  labels_count[item] += 1
print(f"The algorithm was trained on a dataset in size: {len(train_labels)}.")
print(f"The labels distribution: {labels_count}.")

if use_cross_entropy and show_confusion_mat:
  if label_alias_to_use == 1:
    class_names = ["Noise", "Abe", "Phoenix", "Bubba", "Tulsi", "Little"]
    plot_confusion_matrix(predicted, test_labels, class_names)
    plot_confusion_matrix_percentage(predicted, test_labels, class_names)

    m_f_label_alias = {0:0, 1:1, 2:2, 3:1, 4:2, 5:2}
    predicted_m_f = [m_f_label_alias[i] for i in predicted.tolist()]
    test_labels_m_f = [m_f_label_alias[i] for i in test_labels.tolist()]
    predicted_m_f = torch.tensor(predicted_m_f)
    test_labels_m_f = torch.tensor(test_labels_m_f)
    class_names_m_f = ["Noise", "Male", "Female"]

    plot_confusion_matrix(predicted_m_f, test_labels_m_f, class_names_m_f)
    plot_confusion_matrix_percentage(predicted_m_f, test_labels_m_f, class_names_m_f)

    is_noise_label_alias = {0:0, 1:1, 2:1, 3:1, 4:1, 5:1}
    predicted_is_noise = [is_noise_label_alias[i] for i in predicted.tolist()]
    test_labels_is_noise = [is_noise_label_alias[i] for i in test_labels.tolist()]
    predicted_is_noise = torch.tensor(predicted_is_noise)
    test_labels_is_noise = torch.tensor(test_labels_is_noise)
    class_names_is_noise = ["Noise", "Mole-rat (any)"]

    plot_confusion_matrix(predicted_is_noise, test_labels_is_noise, class_names_is_noise)
    plot_confusion_matrix_percentage(predicted_is_noise, test_labels_is_noise, class_names_is_noise)