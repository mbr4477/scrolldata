# scrolldata

An **unofficial** package for loading data from the [Vesuvius Challenge](https://scrollprize.org/).

## Kaggle Demo

Browse the demo Kaggle notebook: [Loading Fragment Data](https://www.kaggle.com/code/matthewbrussell/loading-fragment-data).

## Installation
```bash
python -m pip install git+https://github.com/mbr4477/scrolldata.git
```

## Setup
Create a `*.cfg` config file in the working directory
and include the url, username, and password required for data download.

```ini
[data]
url=<url>
username=<username>
password=<password>
```

## Usage

### Downloading Data
You can easily download scroll data from the command line into `.tif` or `.npy` files:

```bash
python -m scrolldata download FRAG1_54KEV_SURFACE --config my_config.cfg --downsample 4 --cache ./frag1_data [--numpy]
```

For more help, run `python -m scrolldata download -h`.

### Script Usage

To load the data from remote and cache:

```python
from scrolldata import Scroll, VesuviusData

scroll = Scroll.from_remote(
    "path/to/auth.cfg",
    VesuviusData.FRAG1_54KEV_SURFACE,
    data_dir="./FRAG1_54KEV_SURFACE/surface_volume,
    mask_labels_dir="./FRAG1_54KEV_SURFACE",
    downsampling=8,
    numpy=True,
)

# Loads the data as numpy (not recommended)
all_data = scroll.load(to_end=True)
```

To load the data from a local directory (e.g., for Kaggle notebook):

```python
from scrolldata import Scroll

scroll = Scroll.from_remote(
    data_dir="./FRAG1_54KEV_SURFACE/surface_volume,
    mask_labels_dir="./FRAG1_54KEV_SURFACE",
    downsampling=8,
    numpy=False,
)

# Loads the data as numpy (not recommended)
all_data = scroll.load(to_end=True)
```


## Creating a Patch Dataset
The command line tool can also generate a data set of randomly sampled patches to the working directory:

```bash
python -m scrolldata patches FRAG1_54KEV_SURFACE --cache ./cache --downsampling 4 --size 128 --holdout 0.4,0.4,0.2,0.2 --num 512 --train 0.7 --seed 0 --show --export ./out -c config.cfg
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