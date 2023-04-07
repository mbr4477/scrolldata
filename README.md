# scrolldata

An **unofficial** package for loading data from the [Vesuvius Challenge](https://scrollprize.org/).

## Installation
```bash
python -m pip install git+https://github.com/mbr4477/scrolldata.git
```

## Setup
Create a `scrolldata.cfg` file in the working directory
and include the url, username, and password required for data download.

```ini
[data]
url=<url>
username=<username>
password=<password>
```

## Usage

### Downloading Data
You can easily download scroll data from the command line into `.npy` files:

```bash
python -m scrolldata load FRAG1_54KEV_SURFACE --downsample 4 --out frag1.npy
```

This will also create `frag1_ink_labels.npy` and `frag1_mask.npy`. If labels and mask are not available, these files will not be created.

For more help, run `python -m scrolldata load -h`.

### Script Usage

```python
from scrolldata import Scroll, VesuviusData

scroll = Scroll(
    VesuviusData.FRAG1_54KEV_SURFACE, 
    downsampling=4,
    numpy_cache=True # cache downloaded files as NPY instead of TIFF
)

# Pulls the metadata
scroll.init()
print(scroll)

# Loads the data as numpy
all_data = scroll.load(to_end=True)
```

## Creating a Patch Dataset
The command line tool can also generate a data set of randomly sampled patches to the working directory:

```bash
python -m scrolldata patches FRAG1_54KEV_SURFACE --cache ./cache --downsampling 4 --size 128 --holdout 0.4,0.4,0.2,0.2 --num 512 --train 0.7 --seed 0 --show --export
```

For more information use `python -m scrolldata patches -h`.

You can then load this as a PyTorch data set:

```python
from scrolldata.torch import PatchDataset

trainset = PatchDataset("./train")

x = trainset[0]
print(x["inputs"].shape, x["targets"].shape)
```

## Tests
To run tests, install `pytest`.

```bash
python -m pip install pytest
python -m pytest .
```