import torch
import numpy as np
import clearml_poc  # for ClearML integration.
from prediction_model import PredictionModel  # Import the PredictionModel class.
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F


# Function to calculate class inverse weights for handling class imbalance in the dataset.
def get_class_inverse_weights(train_loader: DataLoader):
    # Check if the dataset is a concatenation of multiple datasets.
    if isinstance(train_loader.dataset, ConcatDataset):
        datasets = train_loader.dataset.datasets
    else:
        datasets = [train_loader.dataset]

    # Count the number of occurrences of each class label.
    counter = Counter()
    for dataset in datasets:
        counter = counter + Counter([item[2].item() for item in dataset.data])

    # Handle the case where some classes might be missing in the dataset.
    avg = sum(counter.values()) / len(counter)
    for x in range(4):
        if x not in counter:
            counter[x] = avg

    # Calculate the total count and the inverse weight for each class.
    total = sum(counter.values())
    inverse = [total / (counter[key] * len(counter)) for key in counter.keys()]
    return torch.tensor(inverse, dtype=torch.float32)


# Function to train the model for one epoch.
def train(model: PredictionModel, optimizer, train_loader, device, epoch, params):
    model.train()  # Set the model to training mode.
    sum_loss = 0  # Initialize the sum of losses.
    criterion = torch.nn.CrossEntropyLoss()  # Define the loss function.

    # Iterate through batches of data.
    for sequences, lengths, labels in train_loader:
        sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients before each batch.
        outputs = model(sequences, lengths)  # Forward pass through the model.
        loss = criterion(outputs, labels)  # Calculate the loss.
        loss.backward()  # Backpropagate the loss.
        optimizer.step()  # Update the model parameters.
        sum_loss += loss.item()  # Accumulate the loss.

    # Log the average loss for the epoch using ClearML.
    clearml_poc.add_point_to_graph("Loss", "train " + str(vars(params)), epoch, sum_loss / len(train_loader))


# Function to evaluate the model on a test set.
def test(model: PredictionModel, test_loader, device, epoch, params):
    model.eval()  # Set the model to evaluation mode.
    predictions = []  # List to store predictions.
    intensities = []  # List to store true labels.

    # Define the possible intensity values.
    intensity_values = torch.tensor([x for x in range(0, 4)]).to(device)

    # Iterate through the test data.
    for sequences, lengths, intensity in test_loader:
        intensities.extend(torch.flatten(intensity).tolist())  # Flatten and store true labels.
        sequences, lengths = sequences.to(device), lengths.to(device)
        outputs = F.sigmoid(model(sequences, lengths))  # Forward pass and apply sigmoid activation.
        intensity_predictions = torch.sum(outputs * intensity_values, dim=1).cpu().tolist()  # Calculate predictions.
        predictions.extend(intensity_predictions)

    # Calculate the Pearson correlation between predictions and true labels.
    x = np.asarray(intensities)
    y = np.asarray(predictions)
    pearson_correlation = abs(np.corrcoef(x, y=y)[0][1])

    # Log the Pearson correlation for the epoch using ClearML.
    clearml_poc.add_point_to_graph("Pearson Correlation", "test " + str(vars(params)), epoch, pearson_correlation)
    return pearson_correlation  # Return the Pearson correlation.


# Function to make predictions on new data using the trained model.
def predict(model: PredictionModel, data_loader: DataLoader, device):
    model.eval()  # Set the model to evaluation mode.
    intensity_values = torch.tensor([x for x in range(0, 4)]).to(device)

    # Disable gradient calculation for inference.
    with torch.no_grad():
        for sequences, lengths in data_loader:
            sequences, lengths = sequences.to(device), lengths.to(device)
            outputs = F.sigmoid(model(sequences, lengths))  # Forward pass and apply sigmoid activation.
            intensity_predictions = torch.sum(outputs * intensity_values,
                                              dim=1).cpu().tolist()  # Calculate predictions.
            yield sequences, intensity_predictions  # Yield the sequences and predictions.
