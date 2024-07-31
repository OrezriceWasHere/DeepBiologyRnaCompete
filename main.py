import argparse
import json

import torch
from tqdm import trange
import optuna
import clearml_poc
import hyper_parmas
import rbpcompetesequencedataset
import rbpselexdataset
from prediction_model import PredictionModel
from hyper_parmas import HyperParams
from clearml_poc import clearml_init
import trainer
from sequence_encoder import stack_batch as collate_fn
from torch.utils.data import DataLoader


def init(sequence_file, htr_selex_files, rna_compete_intensities, action, params, device):
    print(f"action is :{action}")

    if action == 'several_datasets':
        datasets = json.load(open("dataset_mapping.json"))
        several_datasets(datasets, params)

    k = params.embedding_char_length
    padded_sequence_max_legnth = params.padding_max_size
    train_dataset = rbpselexdataset.RbpSelexDataset(htr_selex_files, k, padded_sequence_max_legnth)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                               collate_fn=collate_fn)

    test_dataset = rbpcompetesequencedataset.RbpCompeteSequenceDataset(rna_compete_intensities, sequence_file, k,
                                                                       padded_sequence_max_legnth)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)

    if action == 'hyperparam_exploration':
        hyper_parameter_exploration(train_loader, test_loader)

    elif action == 'default_training':
        default_training(train_loader, test_loader, params)

def dataset_index_to_train_loader(datasets, dataset_index, params: HyperParams):
    dataset_files = datasets[dataset_index]["train"]
    htr_selex_files = [f'./data/{file}' for file in dataset_files]
    k = params.embedding_char_length
    padded_sequence_max_length = params.padding_max_size
    train_dataset = rbpselexdataset.RbpSelexDataset(htr_selex_files, k, padded_sequence_max_length)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params.batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn)

    return train_loader


def dataset_index_to_test_loader(datasets, dataset_index, params: HyperParams):
    test_sequences = datasets[dataset_index]["test"]["sequences"]
    test_intensities = datasets[dataset_index]["test"]["intensities"]
    k = params.embedding_char_length
    padded_sequence_max_length = params.padding_max_size

    test_dataset = rbpcompetesequencedataset.RbpCompeteSequenceDataset(test_intensities, test_sequences, k,
                                                                       padded_sequence_max_length)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)

    return test_loader


def several_datasets(datasets, params: HyperParams):
    training_datasets = [0, 1, 2, 3]
    testing_datasets = [4]
    train_loaders = [dataset_index_to_train_loader(datasets, dataset_index, params) for dataset_index in training_datasets]
    test_loaders = [dataset_index_to_test_loader(datasets, dataset_index, params) for dataset_index in testing_datasets]
    default_training(train_loaders, test_loaders, params)


def hyper_parameter_exploration(train_loader, test_loader):
    def optuna_single_trial(trial):
        params = hyper_parmas.optuna_suggest_hyperparams(trial)
        print(f'current params are: {vars(params)}')
        return default_training(train_loader, test_loader, params)

    print("starting optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_single_trial, n_trials=30)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def default_training(train_loader: DataLoader | list[DataLoader], test_loader: DataLoader | list[DataLoader],
                     params, model=None, optimizer=None):
    if not model:
        model = PredictionModel(params).to(device)

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    if isinstance(train_loader, DataLoader):
        train_loader = [train_loader]

    if isinstance(test_loader, DataLoader):
        test_loader = [test_loader]

    pearson_correlation = 0

    for epoch in trange(params.epochs):

        for loader in train_loader:
            trainer.train(model, optimizer, loader, device, epoch, params)

        pearson_correlations = []
        for loader in test_loader:
            pearson_correlation = trainer.test(model, loader, device, epoch, params)
            pearson_correlations.append(pearson_correlation)

        pearson_correlation = sum(pearson_correlations) / len(pearson_correlations)

    return pearson_correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action',
                        choices=['hyperparam_exploration',
                                 'default_training',
                                 'several_datasets'],
                        nargs='?',
                        default='default_training',
                        help='action')
    parser.add_argument('rna_compete_sequences',
                        default="./data/RNAcompete_sequences_rc.txt",
                        nargs='?',
                        type=str,
                        help='sequences file')
    parser.add_argument('htr_selex_files',
                        default=['./data/RBP1_1.txt',
                                 './data/RBP1_2.txt',
                                 './data/RBP1_3.txt',
                                 './data/RBP1_4.txt', ],
                        nargs='*',
                        help='htr selex files')
    default_rna_compete_intensities = './data/RBP1.txt'
    args = parser.parse_args()
    params = HyperParams
    clearml_init(args, params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    action = clearml_poc.get_param("action") or args.action

    init(args.rna_compete_sequences, args.htr_selex_files, default_rna_compete_intensities, action, params, device)
