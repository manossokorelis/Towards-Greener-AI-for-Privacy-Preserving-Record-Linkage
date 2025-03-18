# Importing libraries
import pandas as pd
import pyRAPL
import numpy as np
import time
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import math
pyRAPL.setup()

# Declaring functions
def splittingIntoTrainAndTestSets(dataset, test_size=0.20):
    """
    Splits the dataset into training and testing sets while maintaining class distribution.

    Args:
    dataset (pd.DataFrame): The dataset to be split, where the last column is the target variable.
    test_size (float, optional): Proportion of the dataset to be used as the test set. Defaults to 0.20 (20%).

    Returns:
    tuple: Two pandas DataFrames - (train_set, test_set)
    """
    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=42, stratify=dataset.iloc[:, -1])
    return train_set, test_set

class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with one hidden layer.

    Args:
    input_size (int): Number of input features.
    num_of_neurons_in_hidden_layer (int): Number of neurons in the hidden layer.
    """
    def __init__(self, input_size, num_of_neurons_in_hidden_layer):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, num_of_neurons_in_hidden_layer))
        layers.append(nn.BatchNorm1d(num_of_neurons_in_hidden_layer))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(num_of_neurons_in_hidden_layer, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x) 

def trainingNN(train_set, num_of_neurons_in_hidden_layer, val_size, epochs, batch_size):
    """
    Trains a neural network using the given training dataset.

    Args:
    train_set (pd.DataFrame): The training dataset, where the last column is the target variable.
    num_of_neurons_in_hidden_layer (int): Number of neurons in the hidden layer.
    val_size (float): Proportion of data to be used for validation.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    tuple: Trained PyTorch model and energy measurement result from pyRAPL.
    """
    training_features = train_set.iloc[:, :-1].values
    training_labels = train_set.iloc[:, -1].values
    training_features = scaler.fit_transform(training_features)
    training_features = torch.tensor(training_features, dtype=torch.float32)
    training_labels = torch.tensor(training_labels, dtype=torch.float32).view(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(training_features, training_labels, test_size=val_size, random_state=42, stratify=training_labels)
    model = NeuralNetwork(X_train.shape[1], num_of_neurons_in_hidden_layer) 
    positive_count = (train_set.iloc[:, -1] == 1).sum()
    negative_count = (train_set.iloc[:, -1] == 0).sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / positive_count]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = math.inf
    best_model_state_dict = None
    tnn = pyRAPL.Measurement("trainingNN")
    tnn.begin()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_batches = int(X_train.shape[0]/batch_size) + 1
        model.eval()
        val_epoch_loss = 0
        for i in range (0, X_val.shape[0], batch_size):
            batch_X_val = X_val[i:i + batch_size]
            batch_y_val = y_val[i:i + batch_size]
            val_outputs = model(batch_X_val)
            loss = criterion(val_outputs, batch_y_val)
            val_epoch_loss += loss.item()
        val_batches = int(X_val.shape[0]/batch_size) + 1
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/train_batches:.10f}, Val Loss: {val_epoch_loss/val_batches:.10f}")    
        if val_epoch_loss <= best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_state_dict = model.state_dict()
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    else:
        print("No improvement in validation F1 score, no model state loaded.")
    tnn.end()
    return model, tnn

def evaluateNN(model, test_set, batch_size):
    """
    Evaluates a trained neural network on the provided test dataset and computes confusion matrix metrics.

    Args:
    model (nn.Module): The trained PyTorch model.
    test_set (pd.DataFrame): The test dataset, where the last column is the target variable.
    batch_size (int, optional): Batch size for processing the test set. Defaults to 8192.

    Returns:
    tuple: Confusion matrix values (TN, FP, FN, TP) and energy measurement.
    """
    test_features = test_set.iloc[:, :-1].values.astype("float32")
    test_labels = test_set.iloc[:, -1].values.astype("float32")
    test_features = scaler.transform(test_features)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    enn = pyRAPL.Measurement("evaluateNN")
    enn.begin()
    model.eval()
    total_predictions = []
    with torch.no_grad():
        for i in range(0, test_features.shape[0], batch_size):
            batch_X_test = test_features[i:i + batch_size]
            batch_y_test = test_labels[i:i + batch_size]
            batch_predictions = model(batch_X_test)
            batch_predictions = (batch_predictions > 0.5).int().view(-1)
            total_predictions.append(batch_predictions)
    test_predictions = torch.cat(total_predictions)
    enn.end()
    conf_matrix = confusion_matrix(test_labels.numpy(), test_predictions.numpy())
    return conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1], enn

# Parameters
faults_per_record = 1
flip_probability = 0.01
capacity = 200
error_rate = 0.01
num_of_neurons_in_hidden_layer = 16
epochs = 20
batch_size = 8192 # INCREASING batch_size: FASTER RESULTS & MAYBE LESS ACCURATE
scaler = StandardScaler()

# Importing dataset
dataset = pd.read_csv(f"/home/emmanouil-sokorelis/Thesis/datasets/encoding_pairs/dataset_fpr{faults_per_record}_f{flip_probability}_c{capacity}_er{error_rate}.csv")

# Splitting into train and test sets
train_set, test_set = splittingIntoTrainAndTestSets(dataset=dataset)

# Training neural network and measuring energy with PyRAPL
model, tnn = trainingNN(train_set=train_set, num_of_neurons_in_hidden_layer=num_of_neurons_in_hidden_layer, val_size=0.25, epochs=epochs, batch_size=batch_size)

# Evaluating neural network and measuring energy with PyRAPL
TN, FP, FN, TP, enn = evaluateNN(model=model, test_set=test_set, batch_size=batch_size)

# Results
print(f"Faults Per Record: {faults_per_record}, Capacity: {capacity}, Error Rate: {error_rate}, Flip Probability: {flip_probability}")
print(f"Number of Neurons in Hidden Layer: {num_of_neurons_in_hidden_layer}, Total Epochs: {epochs}")
print(f"Time for Training Neural Network: {tnn.result.duration}")
print(f"CPU Energy Consumed on Training Neural Network: {tnn.result.pkg}")
print(f"RAM Energy Consumed on Training Neural Network: {tnn.result.dram}")
print(f"Time for Evaluating Neural Network: {enn.result.duration}")
print(f"CPU Energy Consumed on Evaluating Neural Network: {enn.result.pkg}")
print(f"RAM Energy Consumed on Evaluating Neural Network: {enn.result.dram}")
print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")


