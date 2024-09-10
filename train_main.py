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
from pathlib import Path
import uuid


def init(sequence_file, htr_selex_files, rna_compete_intensities, action, params, device):
    print(f"action is :{action}")

    if action == 'several_datasets':
        datasets = json.load(open("data/dataset_mapping.json"))
        several_datasets(datasets, params)
        return

    if action == 'test_all':
        datasets = json.load(open("data/dataset_mapping.json"))
        overall_experiment(datasets, params)
        return

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
    training_datasets = [10]
    testing_datasets = [10]
    print("training datasets " + str(training_datasets))
    print("testing datasets " + str(testing_datasets))
    train_datasets = [dataset_index_to_trainset(datasets_mapping, dataset_index, params) for dataset_index in
                      training_datasets]
    train_loaders = [DataLoader(dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
                     for dataset in train_datasets]
    test_datasets = [dataset_index_to_testset(datasets_mapping, dataset_index, params) for dataset_index in
                     testing_datasets]
    test_loaders = [DataLoader(dataset, batch_size=params.batch_size, shuffle=True) for dataset in test_datasets]

    hyper_parameter_exploration(train_loaders, test_loaders)


def hyper_parameter_exploration(train_loader, test_loader):
    def optuna_single_trial(trial):
        params = hyper_parmas.optuna_suggest_hyperparams(trial)
        print(f'current params are: {vars(params)}')
        return default_training(train_loader, test_loader, params)

    print("starting optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_single_trial, n_trials=10)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def default_training(train_loader: DataLoader | list[DataLoader], test_loader: DataLoader | list[DataLoader],
                     params, model=None, optimizer=None, store_model=True):
    if not model:
        model = PredictionModel(params).to(device)

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

    for epoch in trange(params.epochs):

        for loader in train_loader:
            trainer.train(model, optimizer, loader, device, epoch, params)

        pearson_correlations = []
        for loader in test_loader:
            pearson_correlation = trainer.test(model, loader, device, epoch, params)
            pearson_correlations.append(pearson_correlation)

        pearson_correlation = sum(pearson_correlations) / len(pearson_correlations)
        if store_model:
            write_model(model, params, pearson_correlation)

    return pearson_correlation


def overall_experiment(datasets_mapping, params: HyperParams):
    testing_datasets = list(range(38, 76))
    sum_pearson = 0
    count = 0
    for dataset_index in testing_datasets:
        train_dataset = dataset_index_to_trainset(datasets_mapping, dataset_index, params)
        print(f'dataset files: {train_dataset.rbps_files}')
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
        model = PredictionModel(params).to(device)
        path = "model.pt"
        if Path(path).exists() and len(train_dataset.rbps_files) < 4:
            print("loading model")
            model.load_state_dict(torch.load(path, weights_only=True))

        test_dataset = dataset_index_to_testset(datasets_mapping, dataset_index, params)
        test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)
        default_training(train_loader, test_loader, params, model=model, store_model=False)
        pearson_correlation = trainer.test(model, test_loader, device, 1, params)
        sum_pearson += pearson_correlation
        count += 1
        clearml_poc.add_point_to_graph("pearson correlation", "running average", 1, sum_pearson / count)
        print(f"average so far is {sum_pearson / count}")

    print(f"average pearson correlation is {sum_pearson / len(testing_datasets)}")


MODEL_OUTPUT_DIR = "model_output/"
pearson_best_epoch = 0


def write_model(model, params, pearson_result):

    output_file = Path(MODEL_OUTPUT_DIR + "model" + ".pt")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    print(f'new pearson result is {pearson_result}. saving model')
    pearson_best_epoch = pearson_result
    torch.save(model.state_dict(), str(output_file))
    clearml_poc.upload_model_to_clearml(str(output_file), params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action',
                        choices=['hyperparam_exploration',
                                 'test_all',
                                 'default_training',
                                 'several_datasets'],
                        nargs='?',
                        default='several_datasets',
                        help='action')
    parser.add_argument('rna_compete_sequences',
                        default="./data/RNAcompete_sequences_rc.txt",
                        nargs='?',
                        type=str,
                        help='sequences file')
    parser.add_argument('rna_intensities',
                        default='./data/RBP11.txt',
                        nargs='?',
                        help='rna compete intensities')
    parser.add_argument('htr_selex_files',
                        default=[
                            './data/RBP11_1.txt',
                            './data/RBP11_2.txt',
                            './data/RBP11_3.txt',
                            './data/RBP11_4.txt',
                        ],
                        nargs='*',
                        help='htr selex files')
    args = parser.parse_args()
    params = HyperParams
    clearml_init(args, params)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    action = clearml_poc.get_param("action") or args.action
    action = "test_all"
    init(args.rna_compete_sequences, args.htr_selex_files, args.rna_intensities, action, params, device)
