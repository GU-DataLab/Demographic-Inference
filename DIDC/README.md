This repo is based on Optimal Transport Dataset Distance ([otdd](https://github.com/microsoft/otdd)) by Alvarez-Melis el al.

# User-level OTDD

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

A vanilla example for OTDD (example.py)

Toy example For Demographic Inference (example2.py)


## Acknowledgements

This repo relies on the [geomloss](https://www.kernel-operations.io/geomloss/) and [POT](https://pythonot.github.io/) packages for internal EMD and Sinkhorn algorithm implementation. We are grateful to the authors and maintainers of those projects.
