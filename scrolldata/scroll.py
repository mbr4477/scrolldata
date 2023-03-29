from typing import Optional
import requests
from PIL import Image
import io
from os import path, makedirs
import logging
import numpy as np
from enum import Enum
import configparser
import os

logging.basicConfig()
logger = logging.getLogger("scrolldata")
logger.setLevel(
    logging.DEBUG if os.environ.get("DEBUG_SCROLL") == "1" else logging.WARNING
)

_CFG_FILE = "scrolldata.cfg"


class VesuviusData(Enum):
    """Routes to Vesuvius Challenge data."""

    FRAG1_54KEV = "/fragments/Frag1.volpkg/volumes/20230205142449"
    FRAG1_54KEV_SURFACE = (
        "/fragments/Frag1.volpkg/working/54keV_exposed_surface/surface_volume"
    )
    FRAG1_88KEV = "/fragments/Frag1.volpkg/volumes/20230213100222"
    FRAG2_54KEV = "/fragments/Frag2.volpkg/volumes/20230216174557"
    FRAG2_54KEV_SURFACE = (
        "/fragments/Frag2.volpkg/working/54keV_exposed_surface/surface_volume"
    )
    FRAG2_88KEV = "/fragments/Frag2.volpkg/volumes/20230226143835"
    FRAG3_54KEV = "/fragments/Frag3.volpkg/volumes/20230215142309"
    FRAG3_54KEV_SURFACE = (
        "/fragments/Frag3.volpkg/working/54keV_exposed_surface/surface_volume"
    )
    FRAG3_88KEV = "/fragments/Frag3.volpkg/volumes/20230212182547"
    SCROLL1_54KEV = "/full-scrolls/Scroll1.volpkg/volumes/20230205180739"
    SCROLL2_54KEV = "/full-scrolls/Scroll2.volpkg/volumes/20230210143520"
    SCROLL2_88KEV = "/full-scrolls/Scroll2.volpkg/volumes/20230212125146"


class Scroll:
    """A utility class for loading Vesuvius Challenge Data.

    Basic Usage:

    ```python
        from scrolldata import Scroll, VesuviusData

        scroll = Scroll(VesuviusData.FRAG1_54KEV_SURFACE, downsampling=4)
        scroll.init()
        all_data = scroll.load(to_end=True)
    ```

    Please create a `scrolldata.cfg` file in the working directory
    with the data download information accessible after accepting
    the Vesuvius Challenge data license at https://scrollprize.org:

    ```ini
        [data]
        url=<root url>
        username=<username>
        password=<password>
    ```
    """

    def __init__(
        self,
        data: VesuviusData,
        dir: Optional[str] = None,
        downsampling: Optional[int] = None,
        numpy_cache: bool = False,
    ):
        """
        Args:
            data: The VesuviusData to load.
            dir: The cache parent directory.
            downsampling: The downsampling factor to apply.
            numpy_cache: If True, cache data as NPY arrays
                instead of raw TIFF files.
        """
        super().__init__()
        self._slice_height_mm: Optional[float] = None
        self._num_slices: Optional[int] = None
        self._name: Optional[str] = None
        self._min_value: Optional[float] = None
        self._max_value: Optional[float] = None
        self._parent_dir = dir if dir is not None else "."
        self._root_dir: Optional[str] = None
        self._filename_width = 4
        self._mask: Optional[np.ndarray] = None
        self._ink_labels: Optional[np.ndarray] = None
        self._downsampling = downsampling if downsampling is not None else 1
        self._numpy_cache = numpy_cache
        self._uuid: Optional[str] = None

        if not path.exists(_CFG_FILE):
            raise RuntimeError("Unable to find scrollprize.cfg in working directory")

        config = configparser.ConfigParser()
        config.read(_CFG_FILE)
        self._auth = (config["data"]["username"], config["data"]["password"])
        self._volume_url = config["data"]["url"] + str(data.value)

    def __repr__(self) -> str:
        out = [
            "Scroll(",
            f"  url={self._volume_url}",
            f"  name={self.name}",
            f"  num_slices={self.num_slices}",
            f"  slice_height_mm={self.slice_height_mm:5f}",
            f"  min_value={self._min_value}",
            f"  max_value={self._max_value}",
            f"  cache_dir={self.cache_root_dir}",
            ")",
        ]
        return "\n".join(out)

    @property
    def uuid(self) -> str:
        """Get the UUID.

        Returns:
            The UUID.
        """
        assert self._uuid is not None
        return self._uuid

    @property
    def slice_height_mm(self) -> float:
        """Get the slice height in mm.

        Returns:
            The slice height in mm.
        """
        assert self._slice_height_mm is not None
        return self._slice_height_mm

    @property
    def num_slices(self) -> int:
        """Get the number of slices in the scroll.

        Returns:
            The total number of slices.
        """
        assert self._num_slices is not None
        return self._num_slices

    @property
    def name(self) -> str:
        """Get the name of the scroll data.

        Returns:
            The name.
        """
        assert self._name is not None
        return self._name

    @property
    def depth_mm(self) -> float:
        """Get the total depth in mm.

        Returns:
            The depth in mm.
        """
        return self.num_slices * self.slice_height_mm

    @property
    def cache_root_dir(self) -> str:
        """Get the cache root directory.

        Returns:
            The cache root directory path.
        """
        assert self._root_dir is not None
        return self._root_dir

    @property
    def ink_labels(self) -> np.ndarray:
        """Get the ink labels if available.

        Returns:
            The numpy array of ink labels.
        """
        assert self._ink_labels is not None, "No ink labels available"
        return self._ink_labels

    @property
    def mask(self) -> np.ndarray:
        """Get the fragment mask if available.

        Returns:
            The numpy array of the fragment mask.
        """
        assert self._mask is not None, "No fragment mask available"
        return self._mask

    @property
    def has_labels(self) -> bool:
        """Check if the scroll has labels.

        Returns:
            True, if the scroll has a mask and ink labels.
        """
        return self._mask is not None and self._ink_labels is not None

    def _load_image(self, url: str) -> np.ndarray:
        """Load an image, downloading if necessary.

        Args:
            url: The image url.

        Returns:
            The numpy array of image data.
        """
        filename = path.basename(url)
        if self._numpy_cache:
            filename = filename.rsplit(".", 1)[0] + ".npy"

        cache_path = path.join(self.cache_root_dir, filename)
        if path.exists(cache_path):
            logger.info(f"Loading cached {filename} ...")
            if self._numpy_cache:
                out = np.load(cache_path)
            else:
                im = Image.open(cache_path)
                out = np.array(im)
                del im
        else:
            logger.info(f"Downloading {path.basename(url)} ...")
            res = requests.get(url, auth=self._auth)
            assert res.status_code == 200, res.status_code
            im = Image.open(io.BytesIO(res.content))
            out = np.array(im)
            if self._numpy_cache:
                np.save(cache_path, np.array(im))
            else:
                im.save(cache_path)
            del im
        resized = out[:: self._downsampling, :: self._downsampling]
        return resized

    def init(self):
        """Initialize the scroll data."""
        logger.info("Initializing scroll metadata ...")
        self._populate_metadata()
        if not path.exists(self.cache_root_dir):
            makedirs(self.cache_root_dir)

        # Load the fragment data if available
        if "fragments" in self._volume_url and "working" in self._volume_url:
            root_fragment_url = self._volume_url.rsplit("/", maxsplit=1)[0]

            ink_labels_url = f"{root_fragment_url}/inklabels.png"
            self._ink_labels = self._load_image(ink_labels_url)

            mask_url = f"{root_fragment_url}/mask.png"
            self._mask = self._load_image(mask_url)

    def _populate_metadata(self):
        """Populate the metadata from the volume."""
        metadata_url = f"{self._volume_url}/meta.json"
        res = requests.get(metadata_url, auth=self._auth)
        content = res.json()
        self._num_slices = content["slices"]
        self._slice_height_mm = content["voxelsize"] * 1e-3
        self._name = content["name"]
        self._min_value = content["min"]
        self._max_value = content["max"]
        self._root_dir = path.join(self._parent_dir, content["uuid"])
        self._filename_width = int(np.log10(self._num_slices)) + 1
        self._uuid = content["uuid"]

    def load(
        self,
        start_depth_mm: Optional[float] = None,
        end_depth_mm: Optional[float] = None,
        size_mm: Optional[float] = None,
        start_slice: Optional[int] = None,
        end_slice: Optional[int] = None,
        num_slices: Optional[int] = None,
        to_end: bool = False,
    ) -> np.ndarray:
        """Load scroll data as numpy array.

        Must provide one of:
            `start_depth_mm`
            `start_slice`

        Must provide one of:
            `end_depth_mm`
            `size_mm`
            `end_slice`
            `num_slices`
            `to_end`

        Args:
            start_depth_mm: The starting depth in millimeters.
            end_depth_mm: The end depth in millimeters.
            size_mm: The size of the volume to extract, in millimeters.
            start_slice: The starting slice index.
            end_slice: The ending slice index (inclusive).
            num_slices: The number of slices to extract.
            to_end: If True, extract all slices from the provided start to the end.

        Returns:
            The (N, H, W) array of data.
        """
        start_index = 0

        if start_depth_mm is not None:
            start_index = int(start_depth_mm / self.slice_height_mm)
        elif start_slice is not None:
            start_index = start_slice
        else:
            logger.warning(
                "Must provide one of start_depth_mm or start_slice. Defaulting to start_slice = 0"
            )

        end_index = start_index

        if end_depth_mm is not None:
            end_index = int(end_depth_mm / self.slice_height_mm)
        elif size_mm is not None:
            offset = int(size_mm / self.slice_height_mm)
            end_index = start_index + offset
        elif end_slice is not None:
            end_index = end_slice + 1
        elif num_slices is not None:
            end_index = start_index + num_slices
        elif to_end:
            end_index = self.num_slices
        else:
            raise RuntimeError(
                "Must provide one of end_depth_mm, size_mm, end_slice, or num_slices. To download all data explicitly pass to_end=True"
            )

        slices = []
        for i in range(start_index, end_index):
            filename = "{slice:0{width}d}.tif".format(
                slice=i, width=self._filename_width
            )
            slice_url = f"{self._volume_url}/{filename}"
            slices.append(self._load_image(slice_url))

        return np.stack(slices)
