import argparse
import json
import torch
from tqdm import trange
import optuna  # Library for hyperparameter optimization
import clearml_poc  # Custom module for ClearML integration
import hyper_parmas  # Custom module for hyperparameters management
import rbpcompetesequencedataset  # Custom dataset class for RNA compete sequences
import rbpselexdataset  # Custom dataset class for SELEX data
from prediction_model import PredictionModel  # Model class for prediction
from hyper_parmas import HyperParams  # Class for managing hyperparameters
from clearml_poc import clearml_init  # Function to initialize ClearML
import trainer  # Custom module for training and testing logic
from sequence_encoder import stack_batch as collate_fn  # Custom collate function for batching data
from torch.utils.data import DataLoader
from pathlib import Path
import uuid


# Initialize datasets and perform the chosen action
def init(sequence_file, htr_selex_files, rna_compete_intensities, action, params, device):
    print(f"action is :{action}")

    # Handle the 'several_datasets' action, which processes multiple datasets
    if action == 'several_datasets':
        datasets = json.load(open("data/dataset_mapping.json"))
        several_datasets(datasets, params)
        return

    # Handle the 'test_all' action, which runs experiments on all datasets
    if action == 'test_all':
        datasets = json.load(open("data/dataset_mapping.json"))
        overall_experiment(datasets, params)
        return

    # Prepare training and testing datasets
    k = params.embedding_char_length
    padded_sequence_max_legnth = params.padding_max_size
    train_dataset = rbpselexdataset.RbpSelexDataset(htr_selex_files, k, padded_sequence_max_legnth)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                               collate_fn=collate_fn)

    test_dataset = rbpcompetesequencedataset.RbpCompeteSequenceDataset(rna_compete_intensities, sequence_file, k,
                                                                       padded_sequence_max_legnth)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)

    # Handle the 'hyperparam_exploration' action, which explores hyperparameters
    if action == 'hyperparam_exploration':
        hyper_parameter_exploration(train_loader, test_loader)

    # Handle the 'default_training' action, which trains the model with default settings
    elif action == 'default_training':
        default_training(train_loader, test_loader, params)


# Converts dataset index to a training dataset
def dataset_index_to_trainset(datasets, dataset_index, params: HyperParams):
    dataset_files = datasets[dataset_index]["train"]
    htr_selex_files = [f'./data/{file}' for file in dataset_files]
    k = params.embedding_char_length
    padded_sequence_max_length = params.padding_max_size
    train_dataset = rbpselexdataset.RbpSelexDataset(htr_selex_files, k, padded_sequence_max_length)

    return train_dataset



def dataset_index_to_testset(datasets, dataset_index, params: HyperParams):
    test_sequences = "./data/" + datasets[dataset_index]["test"]["sequences"]
    test_intensities = "./data/" + datasets[dataset_index]["test"]["intensities"]
    k = params.embedding_char_length
    padded_sequence_max_length = params.padding_max_size

    test_dataset = rbpcompetesequencedataset.RbpCompeteSequenceDataset(test_sequences, test_intensities, k,
                                                                       padded_sequence_max_length)
    return test_dataset


def several_datasets(datasets_mapping, params: HyperParams):
    training_datasets = [10]  # Indexes of datasets to be used for training
    testing_datasets = [10]  # Indexes of datasets to be used for testing
    print("training datasets " + str(training_datasets))
    print("testing datasets " + str(testing_datasets))

    # Prepare training datasets and loaders
    train_datasets = [dataset_index_to_trainset(datasets_mapping, dataset_index, params) for dataset_index in
                      training_datasets]
    train_loaders = [DataLoader(dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn) for dataset
                     in train_datasets]

    # Prepare testing datasets and loaders
    test_datasets = [dataset_index_to_testset(datasets_mapping, dataset_index, params) for dataset_index in
                     testing_datasets]
    test_loaders = [DataLoader(dataset, batch_size=params.batch_size, shuffle=True) for dataset in test_datasets]

    # Perform hyperparameter exploration
    hyper_parameter_exploration(train_loaders, test_loaders)


# Function to explore hyperparameters using Optuna
def hyper_parameter_exploration(train_loader, test_loader):
    # Define a single trial for Optuna
    def optuna_single_trial(trial):
        params = hyper_parmas.optuna_suggest_hyperparams(trial)
        print(f'current params are: {vars(params)}')
        return default_training(train_loader, test_loader, params)

    print("starting optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_single_trial, n_trials=10)

    # Print the best trial's results
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


# Function for default training
def default_training(train_loader: DataLoader | list[DataLoader], test_loader: DataLoader | list[DataLoader], params,
                     model=None, optimizer=None, store_model=True):
    if not model:
        model = PredictionModel(params).to(device)

    # Uncomment to load an existing model (disabled by default)
    # path = "model.pt"
    # if Path(path).exists():
    #     model.load_state_dict(torch.load(path, weights_only=True))

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    if isinstance(train_loader, DataLoader):
        train_loader = [train_loader]

    if isinstance(test_loader, DataLoader):
        test_loader = [test_loader]

    pearson_correlation = 0

    # Train for a set number of epochs
    for epoch in trange(params.epochs):

        # Train on each loader
        for loader in train_loader:
            trainer.train(model, optimizer, loader, device, epoch, params)

        # Test and compute Pearson correlation for each test loader
        pearson_correlations = []
        for loader in test_loader:
            pearson_correlation = trainer.test(model, loader, device, epoch, params)
            pearson_correlations.append(pearson_correlation)

        pearson_correlation = sum(pearson_correlations) / len(pearson_correlations)

        # Optionally store the model
        if store_model:
            write_model(model, params, pearson_correlation)

    return pearson_correlation


# Function to run experiments on all datasets
def overall_experiment(datasets_mapping, params: HyperParams):
    testing_datasets = list(range(10, 38))  # List of datasets to be tested
    sum_pearson = 0
    count = 0

    # Loop through each dataset for training and testing
    for dataset_index in testing_datasets:
        train_dataset = dataset_index_to_trainset(datasets_mapping, dataset_index, params)
        print(f'dataset files: {train_dataset.rbps_files}')
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
        model = PredictionModel(params).to(device)
        path = "model.pt"

        # Uncomment to load an existing model (disabled by default)
        # if Path(path).exists() and len(train_dataset.rbps_files) == 1:
        #     print("loading model")
        #     model.load_state_dict(torch.load(path, weights_only=True))

        test_dataset = dataset_index_to_testset(datasets_mapping, dataset_index, params)
        test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)

        # Perform default training without saving the model
        default_training(train_loader, test_loader, params, model=model, store_model=False)
        pearson_correlation = trainer.test(model, test_loader, device, 1, params)
        sum_pearson += pearson_correlation
        count += 1
        clearml_poc.add_point_to_graph("pearson correlation", "running average", 1, sum_pearson / count)
        print(f"average so far is {sum_pearson / count}")

    print(f"average pearson correlation is {sum_pearson / len(testing_datasets)}")


MODEL_OUTPUT_DIR = "model_output/"
pearson_best_epoch = 0  # Track the best Pearson correlation


# Function to save the model after training
def write_model(model, params, pearson_result):
    output_file = Path(MODEL_OUTPUT_DIR + "model" + ".pt")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    print(f'new pearson result is {pearson_result}. saving model')
    pearson
