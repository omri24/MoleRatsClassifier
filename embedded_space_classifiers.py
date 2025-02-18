import re
import os
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from copy import deepcopy

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from utils import KNNClassifier, FullyConnectedNN, plot_confusion_matrix, plot_confusion_matrix_percentage, prettytable_to_markdown
from utils import DropoutMLP, BatchNormMLP, MoleRatsLSTM, train_LSTM, evaluate_LSTM

from prettytable import PrettyTable
import time




# Create training and testing sets
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

# Configurations (label alias)
label_alias = None
label_alias_to_use = 1
n_labels = -1
exec(f"label_alias = label_alias{label_alias_to_use}")
train_labels = [label_alias[i] for i in train_labels]
test_labels = [label_alias[i] for i in test_labels]
n_labels = label_alias["n_labels"]

# Configurations (other)
run_kmeams = True
run_knn = True
run_svm = True
show_svm_report = False
show_confusion_mat = False
train_MLP = True
train_dropout_MLP = True
train_batch_norm_MLP = True
train_MoleRatsLSTM =True

# Create table

cpu_or_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if cpu_or_gpu != "cpu":
    hw_type = torch.cuda.get_device_name(0)
else:
    hw_type = "CPU"
results_table = PrettyTable(["Classification method", "Accuracy, range: [0-1]", f"Training time, HW: {hw_type} [sec]"])

# Kmeans
if run_kmeams:
    timer_start = time.time()
    n_clusters = n_labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings_arr)
    labels_estimation = kmeans.labels_

    count_correct = 0
    for i in range(len(labels)):
      if labels[i] == labels_estimation[i]:
        count_correct += 1

    accuracy = round(count_correct / len(labels), 4)
    print(f"\nAccuracy of WavML with KMeans: {accuracy}")
    timer_stop = time.time()
    calc_time = round(timer_stop - timer_start, 4)
    results_table.add_row(["K-means", accuracy, calc_time])

# KNN
if run_knn:
    timer_start = time.time()

    X_train = embeddings_train.copy()
    y_train = deepcopy(train_labels)
    X_test = embeddings_test.copy()
    y_test = deepcopy(test_labels)
    k = 3

    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = round(knn.score(X_test, y_test), 4)

    print(f"Accuracy of WavLM with KNN (k = {k}): {accuracy}")
    timer_stop = time.time()
    calc_time = round(timer_stop - timer_start, 4)
    results_table.add_row([f"KNN (k = {k})", accuracy, calc_time])

# SVM
if run_svm:
    timer_start = time.time()

    train_samples = embeddings_train.copy()
    test_samples = embeddings_test.copy()
    kernel = 'rbf'

    svm_model = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm_model.fit(train_samples, train_labels)
    test_predictions = svm_model.predict(test_samples)

    accuracy = accuracy_score(test_labels, test_predictions)
    accuracy = round(accuracy, 4)
    print(f"Accuracy of SVM with {kernel} kernel: {accuracy}")
    timer_stop = time.time()
    calc_time = round(timer_stop - timer_start, 4)

    results_table.add_row([f"SVM ({kernel} kernel)", accuracy, calc_time])

    if show_svm_report:
        print("\nClassification Report:")
        print(classification_report(test_labels, test_predictions, zero_division=0))

# FC-NN (1 hidden layer)
if train_MLP:

    input_size = 512
    hidden_size = 256
    output_size = n_labels
    learning_rate = 0.001
    num_epochs = 1500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    X_train = train_set.clone().to(device)
    y_train = torch.tensor(train_labels).to(device)
    X_test = test_set.clone().to(device)
    y_test = torch.tensor(test_labels).to(device)

    # Define the model, the optimizer, and the loss
    model = FullyConnectedNN(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Select loss function
    use_cross_entropy = True    # Change this to False to use L2 loss

    if use_cross_entropy:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
        y_train = F.one_hot(y_train, num_classes=output_size).float()

    # Training loop
    timer_start = time.time()
    print("FC-NN training started")

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        if not use_cross_entropy:
            outputs = F.softmax(outputs, dim=1)

        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
          print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete")
    timer_stop = time.time()
    calc_time = round(timer_stop - timer_start, 4)



    # Test
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_outputs = model(X_test)
        if not use_cross_entropy:
            test_outputs = F.softmax(test_outputs, dim=1)  # Apply softmax for L2 loss

        if use_cross_entropy:
            predicted = torch.argmax(test_outputs, dim=1)
            accuracy = (predicted == y_test).float().mean()
            print(f"Test Accuracy (cross entropy loss and 1 hidden layer): {round(accuracy.item(), 4)}")
        else:
            mse = criterion(test_outputs, y_test)
            print(f"Test Mean Squared Error: {mse.item():.4f}")

    if use_cross_entropy:
        accuracy = round(accuracy.item(), 4)
        results_table.add_row([f"FC-NN, 1 hidden layer", accuracy, calc_time])

    print(f"\nThe data was classified {n_labels} classes.")
    labels_count = {}
    for item in train_labels:
      if item not in labels_count.keys():
        labels_count[item] = 0
      labels_count[item] += 1
    print(f"The algorithm was trained on a dataset in size: {len(train_labels)}.")
    print(f"The labels distribution: {labels_count}.")

# FC-NN with drop-out (2 hidden layers)
if train_dropout_MLP:

    # MLP with drop-out
    input_size = 512
    hidden_size1 = 512
    hidden_size2 = 256
    output_size = n_labels
    drop_out_param = 0.3
    learning_rate = 0.001
    num_epochs = 1500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    X_train = train_set.clone().to(device)
    y_train = torch.tensor(train_labels).to(device)
    X_test = test_set.clone().to(device)
    y_test = torch.tensor(test_labels).to(device)

    # Define the model, the optimizer, and the loss
    model = DropoutMLP(input_size, hidden_size1, hidden_size2, output_size, drop_out_param).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Select loss function
    use_cross_entropy = True  # Change this to False to use L2 loss

    if use_cross_entropy:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
        y_train = F.one_hot(y_train, num_classes=output_size).float()

    # Training loop
    timer_start = time.time()
    print("FC-NN training started")
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        if not use_cross_entropy:
            outputs = F.softmax(outputs, dim=1)

        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete")
    timer_stop = time.time()
    calc_time = round(timer_stop - timer_start, 4)

    # Test
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_outputs = model(X_test)
        if not use_cross_entropy:
            test_outputs = F.softmax(test_outputs, dim=1)  # Apply softmax for L2 loss

        if use_cross_entropy:
            predicted = torch.argmax(test_outputs, dim=1)
            accuracy = (predicted == y_test).float().mean()
            print(f"Test Accuracy (cross entropy loss and 2 hidden layers): {round(accuracy.item(), 4)}")
        else:
            mse = criterion(test_outputs, y_test)
            print(f"Test Mean Squared Error: {mse.item():.4f}")

    if use_cross_entropy:
        accuracy = round(accuracy.item(), 4)
        results_table.add_row([f"FC-NN, 2 hidden layers, drop-out", accuracy, calc_time])

    print(f"\nThe data was classified {n_labels} classes.")
    labels_count = {}
    for item in train_labels:
        if item not in labels_count.keys():
            labels_count[item] = 0
        labels_count[item] += 1
    print(f"The algorithm was trained on a dataset in size: {len(train_labels)}.")
    print(f"The labels distribution: {labels_count}.")

# FC-NN with drop-out and batch-norm (3 hidden layers)
if train_batch_norm_MLP:

    input_size = 512
    hidden_size1 = 1024
    hidden_size2 = 512
    hidden_size3 = 256
    output_size = n_labels
    drop_out_param = 0.3
    learning_rate = 0.001
    num_epochs = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    X_train = train_set.clone().to(device)
    y_train = torch.tensor(train_labels).to(device)
    X_test = test_set.clone().to(device)
    y_test = torch.tensor(test_labels).to(device)

    # Define the model, the optimizer, and the loss
    model = BatchNormMLP(input_size, hidden_size1, hidden_size2, hidden_size3 ,output_size, drop_out_param).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Select loss function
    use_cross_entropy = True  # Change this to False to use L2 loss

    if use_cross_entropy:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
        y_train = F.one_hot(y_train, num_classes=output_size).float()

    # Training loop
    timer_start = time.time()
    print("FC-NN training started")
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        if not use_cross_entropy:
            outputs = F.softmax(outputs, dim=1)

        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete")
    timer_stop = time.time()
    calc_time = round(timer_stop - timer_start, 4)

    # Test
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_outputs = model(X_test)
        if not use_cross_entropy:
            test_outputs = F.softmax(test_outputs, dim=1)  # Apply softmax for L2 loss

        if use_cross_entropy:
            predicted = torch.argmax(test_outputs, dim=1)
            accuracy = (predicted == y_test).float().mean()
            print(f"Test Accuracy (cross entropy loss and 3 hidden layers): {round(accuracy.item(), 4)}")
        else:
            mse = criterion(test_outputs, y_test)
            print(f"Test Mean Squared Error: {mse.item():.4f}")

    if use_cross_entropy:
        accuracy = round(accuracy.item(), 4)
        results_table.add_row([f"FC-NN, 3 hidden layers, drop-out, batch-norm", accuracy, calc_time])

    print(f"\nThe data was classified {n_labels} classes.")
    labels_count = {}
    for item in train_labels:
        if item not in labels_count.keys():
            labels_count[item] = 0
        labels_count[item] += 1
    print(f"The algorithm was trained on a dataset in size: {len(train_labels)}.")
    print(f"The labels distribution: {labels_count}.")

# LSTM
if train_MoleRatsLSTM:

    # Hyperparameters
    use_cross_entropy = True
    input_dim = 512
    hidden_dim = 256
    num_layers = 3
    num_classes = n_labels
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    X_train = train_set.clone().to(device)
    y_train = torch.tensor(train_labels).to(device)
    X_test = test_set.clone().to(device)
    y_test = torch.tensor(test_labels).to(device)

    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = MoleRatsLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate
    timer_start = time.time()

    for epoch in range(num_epochs):
        train_loss = train_LSTM(model, train_loader, criterion, optimizer, device)
        test_loss, accuracy = evaluate_LSTM(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:.4f}")

    timer_stop = time.time()
    calc_time = round(timer_stop - timer_start, 4)

    accuracy = round(accuracy, 4)
    results_table.add_row([f"LSTM", accuracy, calc_time])

# Show results
markdown_table = prettytable_to_markdown(results_table)
print(markdown_table)


if use_cross_entropy and show_confusion_mat and (train_MLP or train_dropout_MLP or train_batch_norm_MLP):
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