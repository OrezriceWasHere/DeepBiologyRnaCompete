import torch
from torch.utils.data import Dataset, DataLoader
import sequence_encoder


class RbpCompeteSequenceDataset(Dataset):
    def __init__(self, rbp_file, sequence_file, k, padded_sequence_max_legnth):
        self.rbp_file = rbp_file
        self.sequence_file = sequence_file
        self.possible_encodings = sequence_encoder.all_possible_encodings(k, ['A', 'C', 'G', 'T'])
        self.padded_sequence_max_legnth = padded_sequence_max_legnth
        self.k = k
        self.sequences, self.sequences_length = self._load_sequences(sequence_file)
        self.data = self._load_data()

    def _load_sequences(self, sequence_file):
        sequences = []
        sequences_length = []
        with open(sequence_file, 'r') as f:
            for line in f:
                encoded_sequence, sequence_length = sequence_encoder.encode_embedding(line.strip(), self.possible_encodings, self.k, self.padded_sequence_max_legnth)
                # sequences.append(encoded_sequence.float())
                sequences.append(encoded_sequence)

                sequences_length.append(sequence_length)

        return sequences, sequences_length

    def _load_data(self):
        with open(self.rbp_file, 'r') as f:
            # rbps = [rna_sequence_encoder.encode_rna(_).float() for _ in f]
            rbps = [torch.Tensor([float(_)]).squeeze(-1) for _ in f]
            data = list(zip(self.sequences, self.sequences_length, rbps))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, size, y = self.data[idx]
        return x, size, y

if __name__ == '__main__':
    rbp_file = 'c:/Users/Tomer/Downloads/!htr-selex/test/RBP9.txt'  # Directory containing RBP files
    sequence_file = 'c:/Users/Tomer/Downloads/!htr-selex/test/RNAcompete_sequences.txt'  # Path to the RNAcompete sequences file

    dataset = RbpCompeteSequenceDataset(rbp_file, sequence_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)
