import torch
from torch.utils.data import Dataset, DataLoader
import sequence_encoder  # Import the sequence_encoder module for encoding sequences.

# Dataset for RBP (RNA-binding protein) Compete Sequences without intensity labels
class RbpCompeteSequenceNoIntensityDataset(Dataset):
    def __init__(self, sequence_file, k, padded_sequence_max_legnth):
        # Initialize with the sequence file, Embedding (encoding) size (k), and maximum padded sequence length
        self.sequence_file = sequence_file
        self.possible_encodings = sequence_encoder.all_possible_encodings(k, ['A', 'C', 'G', 'T'])
        self.padded_sequence_max_legnth = padded_sequence_max_legnth
        self.k = k
        self.sequences, self.sequences_length = self._load_sequences(sequence_file)

    # Load and encode sequences from the sequence file
    def _load_sequences(self, sequence_file):
        sequences = []
        sequences_length = []
        with open(sequence_file, 'r') as f:
            for line in f:
                # Encode each sequence and get its length
                encoded_sequence, sequence_length = sequence_encoder.encode_embedding(
                    line.strip(), self.possible_encodings, self.k, self.padded_sequence_max_legnth)
                sequences.append(encoded_sequence)
                sequences_length.append(sequence_length)
        return sequences, sequences_length

    # Return the number of sequences in the dataset
    def __len__(self):
        return len(self.sequences)

    # Get a sequence and its length by index
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences_length[idx]

# Dataset for RBP Compete Sequences with intensity labels
class RbpCompeteSequenceDataset(Dataset):
    def __init__(self, rbp_file, sequence_file, k, padded_sequence_max_legnth):
        # Initialize with the RBP file, sequence file, encoding size (k), and maximum padded sequence length
        self.rbp_file = rbp_file
        self.sequence_file = sequence_file
        self.possible_encodings = sequence_encoder.all_possible_encodings(k, ['A', 'C', 'G', 'T'])
        self.padded_sequence_max_legnth = padded_sequence_max_legnth
        self.k = k
        self.sequences, self.sequences_length = self._load_sequences(sequence_file)
        self.data = self._load_data()

    # Load and encode sequences from the sequence file
    def _load_sequences(self, sequence_file):
        sequences = []
        sequences_length = []
        with open(sequence_file, 'r') as f:
            for line in f:
                # Encode each sequence and get its length
                encoded_sequence, sequence_length = sequence_encoder.encode_embedding(
                    line.strip(), self.possible_encodings, self.k, self.padded_sequence_max_legnth)
                sequences.append(encoded_sequence)
                sequences_length.append(sequence_length)
        return sequences, sequences_length

    # Load RBP data and combine with sequences
    def _load_data(self):
        with open(self.rbp_file, 'r') as f:
            # Load RBP intensities from the file and pair them with sequences
            rbps = [torch.Tensor([float(_)]).squeeze(-1) for _ in f]
            data = list(zip(self.sequences, self.sequences_length, rbps))
        return data

    # Return the number of data points in the dataset
    def __len__(self):
        return len(self.data)

    # Get a data point (sequence, length, and RBP intensity) by index
    def __getitem__(self, idx):
        x, size, y = self.data[idx]
        return x, size, y

# If the script is run directly, this block will execute
if __name__ == '__main__':
    # Specify the paths to the RBP and RNAcompete sequence files
    rbp_file = 'c:/Users/Tomer/Downloads/!htr-selex/test/RBP9.txt'
    sequence_file = 'c:/Users/Tomer/Downloads/!htr-selex/test/RNAcompete_sequences.txt'

    # Create an instance of the RbpCompeteSequenceDataset
    dataset = RbpCompeteSequenceDataset(rbp_file, sequence_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Load data in batches of 4

    # Iterate through the DataLoader and print sequences and their corresponding labels (RBP intensities)
    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)
