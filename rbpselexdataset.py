import os
from torch.utils.data import Dataset, DataLoader


class RbpSelexDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
        self.data = []

        for file in self.files:
            with open(file, 'r') as f:
                for line in f:
                    sequence, _ = line.strip().split(',')
                    label = int(file[:-4].split('_')[1])  # Extract the label
                    self.data.append((sequence, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y


if __name__ == '__main__':
    directory = 'c:/Users/Tomer/Downloads/!htr-selex/'  # Change to your directory containing text files
    dataset = RbpSelexDataset(directory)
    dataloader = DataLoader(dataset, batch_size=64)

    for batch in dataloader:
        sequences, labels = batch
        print('Sequences:', sequences)
        print('Labels:', labels)