import os

import torch
from torch.utils.data import Dataset, DataLoader

import sequence_encoder
import sequence_generator

RBP_DEF_FILE_NAME = './RBP_EXTRA.txt'


class RbpSelexDataset(Dataset):
    def __init__(self, rbps_files, embedding_size, padded_sequence_max_legnth):
        self.rbps_files = rbps_files
        self.data = []
        self.possible_encodings = sequence_encoder.all_possible_encodings(embedding_size, ['A', 'C', 'G', 'T'])
        self.k = embedding_size
        self.padded_sequence_max_legnth = padded_sequence_max_legnth

        # rbps_files.append(RBP_DEF_FILE_NAME)

        for file in self.rbps_files:
            with open(file, 'r') as f:
                for line in f:
                    sequence, _ = line.strip().split(',')
                    label = int(file[:-4].split('_')[1])
                    encoded_sequence, sequence_length = sequence_encoder.encode_embedding(sequence,
                                                                                          self.possible_encodings,
                                                                                          self.k,
                                                                                          self.padded_sequence_max_legnth)
                    # Extract the label
                    # encoded_sequence, sequence_length = sequence_encoder.encode_dna(sequence)
                    # encoded_sequence = encoded_sequence.long()
                    tensor_label = torch.Tensor([label]).long().squeeze(-1) - 1  # Subtract 1 to make the labels 0-based
                    self.data.append((encoded_sequence, sequence_length, tensor_label))

        # if len(rbps_files) == 1:
        #     lines = sequence_generator.generate_rbp_list(num_lines=len(self.data))
        #     for line in lines:
        #         sequence = line
        #         label = 1
        #         encoded_sequence, sequence_length = sequence_encoder.encode_embedding(sequence,
        #                                                                               self.possible_encodings,
        #                                                                               self.k,
        #                                                                               self.padded_sequence_max_legnth)
        #         # Extract the label
        #         # encoded_sequence, sequence_length = sequence_encoder.encode_dna(sequence)
        #         # encoded_sequence = encoded_sequence.long()
        #         tensor_label = torch.Tensor([label]).long().squeeze(-1) - 1
        #         self.data.append((encoded_sequence, sequence_length, tensor_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, length, y = self.data[idx]
        return x, length, y


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
