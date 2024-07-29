import torch
import numpy as np
import clearml_poc
from prediction_model import PredictionModel
import torch.nn.functional as F


def train(model: PredictionModel, optimizer, train_loader, device, epoch, params):
    model.train()
    sum_loss = 0
    for i, (sequences, lengths, labels) in enumerate(train_loader):
        sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        ce_loss = F.cross_entropy(outputs, labels)
        model_weights_loss = (params.sum_weights_lambda * model.sum_weights()).to(ce_loss.device)
        loss = ce_loss + model_weights_loss
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    clearml_poc.add_point_to_graph("Loss", "train " + str(vars(params)), epoch, sum_loss / len(train_loader))


def test(model: PredictionModel, test_loader, device, epoch, params):
    model.eval()
    predictions = []
    intensities = []

    intensity_values = torch.tensor([x for x in range(1, 5)]).to(device)
    for i, (sequences, lengths, intensity) in enumerate(test_loader):
        intensities.extend(torch.flatten(intensity).tolist())
        sequences, lengths = sequences.to(device), lengths.to(device)
        outputs = F.softmax(model(sequences, lengths), dim=-1)
        intensity_predictions = torch.sum(outputs * intensity_values, dim=1).cpu().tolist()
        predictions.extend(intensity_predictions)

    x = np.asarray(intensities)
    y = np.asarray(predictions)
    pearson_correlation = np.corrcoef(x, y=y)[0][1]
    clearml_poc.add_point_to_graph("Pearson Correlation", "test " + str(vars(params)), epoch, pearson_correlation)
    return pearson_correlation
