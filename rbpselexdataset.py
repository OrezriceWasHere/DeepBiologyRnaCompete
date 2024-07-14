import os
from torch.utils.data import Dataset, DataLoader


class RbpSelexDataset(Dataset):
    def __init__(self, rbps_files):
        self.rbps_files = rbps_files
        self.data = []

        for file in self.rbps_files:
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