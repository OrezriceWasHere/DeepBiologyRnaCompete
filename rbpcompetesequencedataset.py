import torch
from torch.utils.data import Dataset, DataLoader
import sequence_encoder  # Custom module for encoding RNA sequences.

# Custom dataset class for handling compete sequence data and associated scores.
class RbpCompeteSequenceDataset(Dataset):
    def __init__(self, rbp_file, sequence_file):
        self.rbp_file = rbp_file  # Path to the file containing RBP scores.
        self.sequence_file = sequence_file  # Path to the file containing RNA compete sequences.
        # Load the sequences and their lengths.
        self.sequences, self.sequences_length = self._load_sequences(sequence_file)
        # Load the combined data (sequences, RBP scores, and sequence lengths).
        self.data = self._load_data()

    def _load_sequences(self, sequence_file):
        # This method loads and encodes RNA sequences from the provided file.
        sequences = []  # List to store encoded RNA sequences.
        sequences_length = []  # List to store the length of each RNA sequence.
        with open(sequence_file, 'r') as f:
            for line in f:
                sequences_length.append(len(line))  # Store the length of the sequence.
                # Encode the RNA sequence using the custom encoder and convert it to a float tensor.
                sequences.append(sequence_encoder.encode_rna(line.strip()).float())

        return sequences, sequences_length

    def _load_data(self):
        # This method loads the RBP scores and combines them with the sequences and their lengths.
        with open(self.rbp_file, 'r') as f:
            # Convert each RBP score to a tensor and combine it with the corresponding sequence and its length.
            rbps = [torch.Tensor([float(_)]).squeeze(-1) for _ in f]
            data = list(zip(self.sequences, rbps, self.sequences_length))
        return data

    def __len__(self):
        # Return the number of data samples in the dataset.
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the sequence, RBP score, and sequence length at the specified index.
        x, y, size = self.data[idx]
        return x, y, size

# Main execution block for testing the dataset and DataLoader.
if __name__ == '__main__':
    # TEST.
    rbp_file = 'c:/Users/Tomer/Downloads/!htr-selex/test/RBP9.txt'
    # Path to the RNAcompete sequences file.
    sequence_file = 'c:/Users/Tomer/Downloads/!htr-selex/test/RNAcompete_sequences.txt'

    # Initialize the custom dataset with the RBP and sequence files.
    dataset = RbpCompeteSequenceDataset(rbp_file, sequence_file)
    # Create a DataLoader to batch, shuffle, and load the data, with a batch size of 4.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate through the batches of data and print the sequences and labels.
    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)
