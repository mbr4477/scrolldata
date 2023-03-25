# scrolldata

A package for loading data from the [Vesuvius Challenge](https://scrollprize.org/).

## Installation
```bash
python -m pip install https://github.com/mbr4477/scrolldata.git
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

### Command Line Tool
You can easily download scroll data from the command line into `.npy` files:

```bash
python -m scrolldata FRAG1_54KEV_SURFACE --downsample 4 --out frag1.npy
```

This will also create `frag1_ink_labels.npy` and `frag1_mask.npy`. If labels and mask are not available, these files will not be created.

For more help, run `python -m scrolldata -h`.

### Script Usage

```python
from scrolldata import Scroll, VesuviusData

scroll = Scroll(
    VesuviusData.FRAG1_54KEV_SURFACE, 
    downsampling=4
)

# Pulls the metadata
scroll.init()
print(scroll)

# Loads the data as numpy
all_data = scroll.load(to_end=True)
```