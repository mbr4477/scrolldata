import argparse
from .scroll import Scroll, VesuviusData
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "data",
    type=str,
    choices=[e.name for e in VesuviusData],
    help="the scroll to load",
)
parser.add_argument("--start", "-s", type=int, help="first slice index", default=0)
parser.add_argument("--end", "-e", type=int, help="last slice index")
parser.add_argument("--downsampling", "-d", type=int, help="downsampling factor")
parser.add_argument("--out", "-o", type=str, help="output numpy file")
parser.add_argument("--cache", type=str, help="path to cache directory")
parser.add_argument("--numpy", action="store_true", help="cache as numpy", default=False)
args = parser.parse_args()

data = Scroll(VesuviusData[args.data], args.cache, args.downsampling, args.numpy)
data.init()
x = data.load(start_slice=args.start, end_slice=args.end, to_end=args.end is None)
if args.out:
    np.save(args.out, x)
    if data.has_labels:
        parts = args.out.rsplit(".", 1)
        np.save(f"{parts[0]}_mask.{parts[1]}", data.mask)
        np.save(f"{parts[0]}_ink_labels.{parts[1]}", data.ink_labels)
