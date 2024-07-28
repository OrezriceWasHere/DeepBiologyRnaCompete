import torch
from torch.utils.data import Dataset, DataLoader
import sequence_encoder


class RbpCompeteSequenceDataset(Dataset):
    def __init__(self, rbp_file, sequence_file):
        self.rbp_file = rbp_file
        self.sequence_file = sequence_file
        self.sequences, self.sequences_length = self._load_sequences(sequence_file)
        self.data = self._load_data()

    def _load_sequences(self, sequence_file):
        sequences = []
        sequences_length = []
        with open(sequence_file, 'r') as f:
            for line in f:
                sequences_length.append(len(line))
                sequences.append(sequence_encoder.encode_rna(line.strip()).float())
                # sequences.append(torch.Tensor([float(line.strip())]).squeeze(-1))

        return sequences, sequences_length

    def _load_data(self):
        with open(self.rbp_file, 'r') as f:
            # rbps = [rna_sequence_encoder.encode_rna(_).float() for _ in f]
            rbps = [torch.Tensor([float(_)]).squeeze(-1) for _ in f]
            data = list(zip(self.sequences, rbps, self.sequences_length))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, size = self.data[idx]
        return x, y, size

if __name__ == '__main__':
    rbp_file = 'c:/Users/Tomer/Downloads/!htr-selex/test/RBP9.txt'  # Directory containing RBP files
    sequence_file = 'c:/Users/Tomer/Downloads/!htr-selex/test/RNAcompete_sequences.txt'  # Path to the RNAcompete sequences file

    dataset = RbpCompeteSequenceDataset(rbp_file, sequence_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)
