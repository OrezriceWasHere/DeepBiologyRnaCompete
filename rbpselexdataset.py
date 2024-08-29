import os

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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

        for file in tqdm(self.rbps_files):
            with open(file, 'r', encoding='utf-8') as f:
                detected_faulty_in_line = False
                label = int(file[:-4].split('_')[-1])

                for lindex, line in enumerate(f):

                    if len(line) > 55:
                        print(f"There is something wrong with line reading. read string with size {len(line)} in file {file} at line #{lindex}")
                        detected_faulty_in_line = True
                        continue
                    sequence, _ = line.strip().split(',')
                    encoded_sequence, sequence_length = sequence_encoder.encode_embedding(sequence,
                                                                                          self.possible_encodings,
                                                                                          self.k,
                                                                                          self.padded_sequence_max_legnth)
                    # Extract the label
                    # encoded_sequence, sequence_length = sequence_encoder.encode_dna(sequence)
                    # encoded_sequence = encoded_sequence.long()
                    tensor_label = torch.Tensor([label]).long().squeeze(-1) - 1  # Subtract 1 to make the labels 0-based
                    self.data.append((encoded_sequence, sequence_length, tensor_label))
                if detected_faulty_in_line:
                    print(f"read {lindex} lines from file {file}")

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
