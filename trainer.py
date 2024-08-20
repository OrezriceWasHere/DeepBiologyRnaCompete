import torch
import numpy as np
import clearml_poc
from prediction_model import PredictionModel
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F

def get_class_inverse_weights(train_loader: DataLoader):
    if isinstance(train_loader.dataset, ConcatDataset):
        datasets = train_loader.dataset.datasets
    else:
        datasets = [train_loader.dataset]

    counter = Counter()
    for dataset in datasets:
        counter = counter + Counter([item[2].item() for item in dataset.data])

    avg = sum(counter.values()) / len(counter)
    for x in range(4):
        if x not in counter:
            counter[x] = avg

    total = sum(counter.values())
    inverse = [total / (counter[key] * len(counter)) for key in counter.keys()]
    return torch.tensor(inverse, dtype=torch.float32)


def train(model: PredictionModel, optimizer, train_loader, device, epoch, params):
    model.train()
    sum_loss = 0
    # balance = get_class_inverse_weights(train_loader).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # criterion = torch.nn.HuberLoss()
    for sequences, lengths, labels in train_loader:
        sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)
        # labels = (labels.float() * 0.3333)
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    clearml_poc.add_point_to_graph("Loss", "train " + str(vars(params)), epoch, sum_loss / len(train_loader))


def test(model: PredictionModel, test_loader, device, epoch, params):
    model.eval()
    predictions = []
    intensities = []

    intensity_values = torch.tensor([x for x in range(4)]).to(device)
    for sequences, lengths, intensity in test_loader:
        intensities.extend(torch.flatten(intensity).tolist())
        sequences, lengths = sequences.to(device), lengths.to(device)
        # intensity_predictions = torch.argmax(model(sequences, lengths), dim=-1).cpu().tolist()
        outputs = F.softmax(model(sequences, lengths), dim=-1)
        # outputs = F.sigmoid(model(sequences, lengths))
        # outputs = model(sequences, lengths)
        intensity_predictions = torch.sum(outputs * intensity_values, dim=1).cpu().tolist()
        predictions.extend(intensity_predictions)

    x = np.asarray(intensities)
    y = np.asarray(predictions)
    pearson_correlation = np.corrcoef(x, y=y)[0][1]
    clearml_poc.add_point_to_graph("Pearson Correlation", "test " + str(vars(params)), epoch, pearson_correlation)
    return pearson_correlation
