import argparse
import logging

import numpy as np

from ._make_patch_dataset import make_patch_dataset
from ._scroll import LOGGER_NAME, Scroll, VesuviusData

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# Load command
load_parser = subparsers.add_parser(
    "load", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
load_parser.set_defaults(which="load")
load_parser.add_argument(
    "data",
    type=str,
    choices=[e.name for e in VesuviusData],
    help="the scroll to load",
)
load_parser.add_argument("--start", "-s", type=int, help="first slice index", default=0)
load_parser.add_argument("--end", "-e", type=int, help="last slice index")
load_parser.add_argument("--downsampling", "-d", type=int, help="downsampling factor")
load_parser.add_argument("--out", "-o", type=str, help="output numpy file")
load_parser.add_argument("--cache", type=str, help="path to cache directory")
load_parser.add_argument(
    "--numpy", action="store_true", help="cache as numpy", default=False
)
load_parser.add_argument("--verbose", "-v", action="store_true", help="show all logs")

# Patches command
patches_parser = subparsers.add_parser(
    "patches", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
patches_parser.set_defaults(which="patches")
patches_parser.add_argument(
    "data",
    type=str,
    choices=[e.name for e in VesuviusData],
    help="the scroll to load",
)
patches_parser.add_argument(
    "--downsampling", "-d", type=int, help="downsampling factor", default=1
)
patches_parser.add_argument(
    "--cache", type=str, help="path to cache directory", default="."
)
patches_parser.add_argument(
    "--size", type=int, help="the size of the square patches", required=True
)
patches_parser.add_argument(
    "--holdout",
    type=str,
    help="the comma-separated x,y,w,h of the holdout region like '0.4,0.4,0.2,0.2'",
    required=True,
)
patches_parser.add_argument(
    "--num", type=int, help="number of patches to sample", required=True
)
patches_parser.add_argument(
    "--export",
    action="store_true",
    help="export the patches to the working directory",
    default=False,
)
patches_parser.add_argument(
    "--show", action="store_true", help="show the patches", default=False
)
patches_parser.add_argument(
    "--train",
    type=float,
    help="fraction of patches to use for training data",
    default=0.7,
)
patches_parser.add_argument(
    "--seed", type=int, help="random seed for patch sampling", default=0
)
patches_parser.add_argument(
    "--verbose", "-v", action="store_true", help="show all logs"
)

args = parser.parse_args()

if args.verbose:
    logging.basicConfig()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

if args.which == "load":
    data = Scroll(VesuviusData[args.data], args.cache, args.downsampling, args.numpy)
    data.init()
    x = data.load(start_slice=args.start, end_slice=args.end, to_end=args.end is None)
    if args.out:
        np.save(args.out, x)
        if data.has_labels:
            parts = args.out.rsplit(".", 1)
            np.save(f"{parts[0]}_mask.{parts[1]}", data.mask)
            np.save(f"{parts[0]}_ink_labels.{parts[1]}", data.ink_labels)
elif args.which == "patches":
    make_patch_dataset(
        VesuviusData[args.data],
        args.downsampling,
        args.size,
        args.cache,
        args.num,
        tuple([float(x) for x in args.holdout.split(",")]),
        args.export,
        args.show,
        args.seed,
        train_frac=0.7,
    )
