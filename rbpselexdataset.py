import os

import torch
from torch.utils.data import Dataset, DataLoader

import rna_sequence_encoder


class RbpSelexDataset(Dataset):
    def __init__(self, rbps_files):
        self.rbps_files = rbps_files
        self.data = []

        for file in self.rbps_files:
            with open(file, 'r') as f:
                for line in f:
                    sequence, _ = line.strip().split(',')
                    label = int(file[:-4].split('_')[1])  # Extract the label
                    encoded_sequence = rna_sequence_encoder.encode_rna(sequence).float()
                    tensor_label = torch.Tensor([label]).long().squeeze(-1) - 1  # Subtract 1 to make the labels 0-based
                    self.data.append((encoded_sequence, tensor_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y


if __name__ == '__main__':
    file1 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP18_3.txt'
    file2 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP13_3.txt'
    file3 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP11_4.txt'
    list = [file1, file2, file3]
    dataset = RbpSelexDataset(list)
    dataloader = DataLoader(dataset, batch_size=64)

    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)