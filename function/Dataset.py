
from torch.utils.data import Dataset
import csv
import torch

class MyDataset(Dataset):
    def __init__(self, data_file):
        super(MyDataset, self).__init__()

        # Check the file extension
        if data_file.endswith('.csv'):
            self._load_csv(data_file)
        elif data_file.endswith('.arff'):
            self._load_arff(data_file)
        else:
            raise ValueError("Unsupported file format.")

    def _load_csv(self, data_file):
        # Load data from CSV file
        with open(data_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            data = list(csv_reader)

        # Extract features and labels from the loaded data
        self.features = []
        self.labels = []
        for line in data:
            self.features.append([float(f) for f in line[:-1]])
            self.labels.append(float(line[-1]))
        self.features = torch.tensor(self.features)
        self.labels = torch.tensor(self.labels)

    def _load_arff(self, data_file):
        # Load data from ARFF file
        with open(data_file, 'r') as f:
            lines = f.readlines()

        data_start = False
        data_lines = []
        for line in lines:
            if not data_start:
                if line.startswith('@data'):
                    data_start = True
            else:
                if not line.startswith('@'):
                    data_lines.append(line.strip())

        # Process data lines to extract features and labels
        data = [line.split(',') for line in data_lines]
        self.features = torch.tensor([list(map(float, row[:-1])) for row in data], dtype=torch.float32)
        self.labels = torch.tensor([float(row[-1]) for row in data], dtype=torch.float32)

    def split_dataset(self, num_splits):
        # Calculate the number of samples in each split
        split_size = len(self.features) // num_splits

        # Split the dataset into num_splits parts
        split_features = [[] for _ in range(num_splits)]
        split_labels = [[] for _ in range(num_splits)]
        for i in range(len(self.features)):
            split_index = i % num_splits

            split_features[split_index].append(self.features[i])
            split_labels[split_index].append(self.labels[i])

        return split_features, split_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
