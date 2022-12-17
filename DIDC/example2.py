from otdd.pytorch.distance import DatasetDistance
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self):
        # embedding space
        self.embddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.4], [0.1, 0.1, 0.1], [0.2, 0.2, 0.5]])
        self.targets = torch.tensor([0, 1, 0, 1])
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        emd = self.embddings[idx]
        label = self.targets[idx]
        return emd, label

data = CustomDataset()
wiki_loader = DataLoader(data,batch_size=64)

data = CustomDataset()
imdb_loader = DataLoader(data,batch_size=64)

dist = DatasetDistance(wiki_loader, imdb_loader,
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          p = 2, entreg = 1e-1)

d = dist.distance()
print(f'OTDD(MNIST,USPS)={d:8.2f}')
