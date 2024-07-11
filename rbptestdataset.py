import os
from torch.utils.data import Dataset, DataLoader


class RBPSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_file):
        self.data_dir = data_dir
        self.sequence_file = sequence_file
        self.files = [f for f in os.listdir(data_dir) if f.startswith('RBP') and f.endswith('.txt')]
        self.sequences = self._load_sequences(sequence_file)
        self.data = self._load_data()

    def _load_sequences(self, sequence_file):
        sequences = []
        with open(sequence_file, 'r') as f:
            for line in f:
                sequences.append(line.strip())
        return sequences

    def _load_data(self):
        data = []
        for file in self.files:
            for seq in self.sequences:
                with open(os.path.join(self.data_dir, file), 'r') as f:
                    for line in f:
                        data.append((seq, line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

if __name__ == '__main__':
    data_dir = 'c:/Users/Tomer/Downloads/!htr-selex/test'  # Directory containing RBP files
    sequence_file = os.path.join(data_dir, 'RNAcompete_sequences.txt')  # Path to the RNAcompete sequences file

    dataset = RBPSequenceDataset(data_dir, sequence_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)
