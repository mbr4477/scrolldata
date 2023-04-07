import argparse
import logging

from tqdm import tqdm

from ._make_patch_dataset import make_patch_dataset
from ._scroll import LOGGER_NAME, Scroll, VesuviusData

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# Load command
load_parser = subparsers.add_parser(
    "download", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
load_parser.set_defaults(which="download")
load_parser.add_argument(
    "--config",
    "-c",
    type=str,
    help="path to config file with data set url and auth",
    required=True,
)
load_parser.add_argument(
    "data",
    type=str,
    choices=[e.name for e in VesuviusData],
    help="the scroll to load",
)
load_parser.add_argument("--start", "-s", type=int, help="first slice index", default=0)
load_parser.add_argument("--end", "-e", type=int, help="last slice index")
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
    "--config",
    "-c",
    type=str,
    help="path to config file with data set url and auth",
    required=True,
)
patches_parser.add_argument(
    "--numpy", action="store_true", help="cache as numpy", default=False
)
patches_parser.add_argument(
    "--num", type=int, help="number of patches to sample", required=True
)
patches_parser.add_argument(
    "--export",
    type=str,
    help="export the patches to this directory",
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

if args.which == "download":
    data = Scroll.from_remote(
        args.config,
        VesuviusData[args.data],
        args.cache,
        downsampling=1,
        numpy=args.numpy,
    )
    assert data.metadata is not None
    for i in tqdm(
        range(
            args.start,
            args.end + 1 if args.end is not None else data.metadata.num_slices,
        )
    ):
        tmp = data.load(start_slice=i, num_slices=1)
        del tmp

elif args.which == "patches":
    scroll = Scroll.from_remote(
        args.config,
        VesuviusData[args.data],
        args.cache,
        downsampling=args.downsampling,
        numpy=args.numpy,
    )
    make_patch_dataset(
        scroll,
        args.size,
        args.num,
        tuple([float(x) for x in args.holdout.split(",")]),
        args.export,
        args.show,
        args.seed,
        train_frac=0.7,
    )
