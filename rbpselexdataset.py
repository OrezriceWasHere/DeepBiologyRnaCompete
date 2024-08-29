import os  # Import the os module for interacting with the operating system.
import torch  # Import PyTorch for tensor operations and neural network handling.
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader for custom dataset handling.
from tqdm import tqdm  # Import tqdm for creating progress bars.
import sequence_encoder  # Import the sequence_encoder module for encoding sequences.
import sequence_generator  # Import sequence_generator, possibly for generating sequences (though not used here).

# Default file name for RBP (RNA-binding protein) data
RBP_DEF_FILE_NAME = './RBP_EXTRA.txt'


# Custom Dataset for handling SELEX (Systematic Evolution of Ligands by EXponential enrichment) data
class RbpSelexDataset(Dataset):
    def __init__(self, rbps_files, embedding_size, padded_sequence_max_legnth):
        # Initialize with the list of RBP files, embedding size, and maximum padded sequence length
        self.rbps_files = rbps_files
        self.data = []
        self.possible_encodings = sequence_encoder.all_possible_encodings(embedding_size, ['A', 'C', 'G', 'T'])
        self.k = embedding_size
        self.padded_sequence_max_legnth = padded_sequence_max_legnth

        # Iterate through each file and load the data
        for file in tqdm(self.rbps_files):
            with open(file, 'r', encoding='utf-8') as f:
                detected_faulty_in_line = False
                label = int(file[:-4].split('_')[-1])  # Extract label from the file name

                for lindex, line in enumerate(f):
                    # Check for lines that are too long, indicating a possible error
                    if len(line) > 55:
                        print(
                            f"There is something wrong with line reading. read string with size {len(line)} in file {file} at line #{lindex}")
                        detected_faulty_in_line = True
                        continue

                    # Extract the sequence and encode it
                    sequence, _ = line.strip().split(',')
                    encoded_sequence, sequence_length = sequence_encoder.encode_embedding(
                        sequence, self.possible_encodings, self.k, self.padded_sequence_max_legnth)

                    # Convert the label to a tensor,
                    tensor_label = torch.Tensor([label]).long().squeeze(-1) - 1
                    self.data.append((encoded_sequence, sequence_length, tensor_label))

                # Print a message if any faulty lines were detected
                if detected_faulty_in_line:
                    print(f"read {lindex} lines from file {file}")

    # Return the number of data points in the dataset
    def __len__(self):
        return len(self.data)

    # Get a data point (sequence, length, and label) by index
    def __getitem__(self, idx):
        x, length, y = self.data[idx]
        return x, length, y


# Main execution block
if __name__ == '__main__':
    # Example files containing RBP data
    file1 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP18_3.txt'
    file2 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP13_3.txt'
    file3 = 'c:/Users/Tomer/Downloads/!htr-selex/RBP11_4.txt'

    # Create a list of files
    file_list = [file1, file2, file3]

    # Create an instance of RbpSelexDataset with the file list
    dataset = RbpSelexDataset(file_list, embedding_size=8, padded_sequence_max_legnth=100)

    # Load the dataset in batches of 64
    dataloader = DataLoader(dataset, batch_size=64)

    # Iterate through the DataLoader and print sequences and their corresponding labels
    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)
