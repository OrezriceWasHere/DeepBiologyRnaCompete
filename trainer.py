import torch
import numpy as np
import clearml_poc
from prediction_model import PredictionModel
import torch.nn.functional as F


def train(model: PredictionModel, optimizer, train_loader, device, epoch, params):
    model.train()
    sum_loss = 0
    for i, (sequences, labels) in enumerate(train_loader):
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    clearml_poc.add_point_to_graph("Loss", "train " + str(vars(params)), epoch, sum_loss / len(train_loader))


def test(model: PredictionModel, test_loader, device, epoch, params):
    model.eval()
    predictions = []
    intensities = []

    intensity_values = torch.tensor([x for x in range(1, 5)]).to(device)
    for i, (sequences, intensity) in enumerate(test_loader):
        intensities.extend(torch.flatten(intensity).tolist())
        sequences = sequences.to(device)
        outputs = F.softmax(model(sequences), dim=-1)
        intensity_predictions = torch.sum(outputs * intensity_values, dim=1).cpu().tolist()
        predictions.extend(intensity_predictions)

    x = np.asarray(intensities)
    y = np.asarray(predictions)
    pearson_correlation = np.corrcoef(x, y=y)[0][1]
    clearml_poc.add_point_to_graph("Pearson Correlation", "test " + str(vars(params)), epoch, pearson_correlation)
    return pearson_correlation
