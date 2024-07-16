import argparse
import torch
from tqdm import trange
import clearml_poc
import rbpcompetesequencedataset
import rbpselexdataset
from prediction_model import PredictionModel
from hyper_parmas import HyperParams
from clearml_poc import clearml_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    data = torch.stack(data, dim=0)
    target = torch.stack(target, dim=0)

    return data, target


def main(sequence_file, htr_selex_files, rna_compete_intensities, params):

    model = PredictionModel(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    train_dataset = rbpselexdataset.RbpSelexDataset(htr_selex_files)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset = rbpcompetesequencedataset.RbpCompeteSequenceDataset(rna_compete_intensities, sequence_file)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    for epoch in trange(params.epochs):
        # train(model, optimizer, train_loader, epoch)
        test(model, test_loader, epoch)

    # max_len = 41
    # input = torch.randint(params.one_hot_size, (params.batch_size, max_len)).to(device)
    # input = torch.nn.functional.one_hot(input).float()
    # input = input.permute(0, 2, 1)
    #
    # model = PredictionModel(params).to(device)
    # y = model(input)
    # pass


def train(model, optimizer, train_loader, epoch):

    sum_loss = 0
    for i, (sequences, labels) in enumerate(train_loader):
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    clearml_poc.add_point_to_graph("Loss", "train", epoch, sum_loss / len(train_loader))


def test(model, test_loader, epoch):

    predictions = []
    intensities = []

    intensity_values = torch.tensor([x for x in range(5)]).to(device)
    for i, (sequences, intensity) in enumerate(test_loader):
        intensities.extend(torch.flatten(intensity).tolist())
        sequences = sequences.to(device)
        intensity_predictions = model(sequences) * intensity_values
        predictions.extend(intensity_predictions)

    pearson_correlation = torch.corrcoef(predictions, intensities)[0][1].item()
    clearml_poc.add_point_to_graph("Pearson Correlation", "test", epoch, pearson_correlation)



# Regression
# Pearson correlation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('rna_compete_sequences',
                        default="./data/RNAcompete_sequences.txt",
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
    clearml_init(params)

    main(args.rna_compete_sequences, args.htr_selex_files, default_rna_compete_intensities, params)
