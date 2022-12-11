This repo is largely based on otdd by Alvarez-Melis el al. (https://github.com/microsoft/otdd)

# Optimal Transport Dataset Distance (OTDD)

## Getting Started

### Installation

**Note**: It is highly recommended that the following be done inside a virtual environment


#### Via Conda (recommended)

If you use [ana|mini]conda , you can simply do:

```
conda env create -f environment.yaml python=3.8
conda activate otdd
conda install .
```

(you might need to install pytorch separately if you need a custom install)

#### Via pip

First install dependencies. Start by install pytorch with desired configuration using the instructions provided in the [pytorch website](https://pytorch.org/get-started/locally/). Then do:
```
pip install -r requirements.txt
```
Finally, install this package:
```
pip install .
```

## Usage Examples

A vanilla example for OTDD:

```python
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance


# Load datasets
loaders_src = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=2000)[0]
loaders_tgt = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=2000)[0]

# Instantiate distance
dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device='cpu')

d = dist.distance(maxsamples = 1000)
print(f'OTDD(src,tgt)={d}')

```

Toy example For Demographic Inference:

```python
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self):
        # embedding space
        self.embddings = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.4], [0.1, 0.1, 0.1], [0.2, 0.2, 0.5]])
        self.targets = [0, 1, 0, 1]
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
```


## Acknowledgements

This repo relies on the [geomloss](https://www.kernel-operations.io/geomloss/) and [POT](https://pythonot.github.io/) packages for internal EMD and Sinkhorn algorithm implementation. We are grateful to the authors and maintainers of those projects.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
