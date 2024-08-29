import os

import torch
from torch.utils.data import Dataset, DataLoader

import sequence_encoder  # Custom module for encoding DNA sequences.
import sequence_generator  # Custom module for generating sequence files.

# Define a constant for a default RBP (RNA-binding protein) file name.
RBP_DEF_FILE_NAME = './RBP1_1.txt'

# Custom dataset class for loading and processing RBPSelex.
class RbpSelexDataset(Dataset):
    def __init__(self, rbps_files):
        self.rbps_files = rbps_files  # List of RBP files provided as input.
        self.data = []  # List to store the encoded sequences and labels.

        # If only one file is provided, generate an additional RBP file using sequence_generator.
        if len(rbps_files) == 1:
            sequence_generator.generate_rbp_file(RBP_DEF_FILE_NAME)
            rbps_files.append(RBP_DEF_FILE_NAME)  # Add the generated file to the list.

        # Iterate over each file in the provided list.
        for file in self.rbps_files:
            with open(file, 'r') as f:
                for line in f:
                    # Split each line to separate the RNA sequence from the rest of the data.
                    sequence, _ = line.strip().split(',')
                    # Extract the label from the file name.
                    label = int(file[:-4].split('_')[1])
                    # Encode the DNA sequence using the custom encoder and convert it to a float tensor.
                    encoded_sequence = sequence_encoder.encode_dna(sequence).float()
                    # Convert the label to a long tensor and make it 0-based (by subtracting 1).
                    tensor_label = torch.Tensor([label]).long().squeeze(-1) - 1
                    # Append the encoded sequence and label as a tuple to the data list.
                    self.data.append((encoded_sequence, tensor_label))

    def __len__(self):
        # Return the number of samples in the dataset.
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the encoded sequence and label at the specified index.
        x, y = self.data[idx]
        return x, y


# Main execution block for testing the dataset and dataloader.
if __name__ == '__main__':
    # TEST
    file1 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP18_3.txt'
    file2 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP13_3.txt'
    file3 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP11_4.txt'
    list = [file1, file2, file3]  # Create a list of file paths.

    # Initialize the custom dataset with the list of files.
    dataset = RbpSelexDataset(list)
    # Create a DataLoader to batch and shuffle the data, setting the batch size to 64.
    dataloader = DataLoader(dataset, batch_size=64)

    # Iterate through the batches of data and print the sequences and labels.
    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)
