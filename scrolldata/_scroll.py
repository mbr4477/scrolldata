import configparser
import glob
import io
import json
import logging
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from os import makedirs, path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from typing_extensions import Self

from . import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


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


@dataclass
class VesuviusMetadata:
    """Vesuvius scroll metadata."""

    name: str
    num_slices: int
    slice_height_mm: float
    uuid: str
    min_value: float
    max_value: float
    depth_mm: float


class Resolver(ABC):
    def get_num_slices(self) -> int:
        """Get the number of slices.

        Returns:
            The number of slices.
        """
        ...

    def resolve_slice(self, slice: int, width: Optional[int] = None) -> np.ndarray:
        """Resolve the specified image slice.

        Args:
            slice: The slice index.
            width: The filename width. If the index is 12
                and the width is 5, this implies "00012"

        Returns:
            The npy array of the slice.
        """
        ...

    def resolve_mask(self) -> Optional[np.ndarray]:
        """Get the mask, if one exists.

        Returns:
            The numpy mask array or None.
        """
        ...

    def resolve_ink_labels(self) -> Optional[np.ndarray]:
        """Get the ink labels, if they exist.

        Returns:
            The numpy array of labels or None.
        """
        ...

    def resolve_metadata(self) -> Optional[VesuviusMetadata]:
        """Resolve the metadata.

        Returns:
            The metadata or None.
        """
        ...


class RemoteResolver(Resolver):
    """Resolve the images via the online server if necessary."""

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        data: VesuviusData,
        data_dir: str,
        mask_labels_dir: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        numpy: bool = False,
        downsampling: int = 1,
    ):
        """
        Args:
            url: The official data set base url
            username: The username.
            password: The password.
            data: The remote data to link to.
            data_dir: The local folder for caching data.
            mask_labels_dir: Optional directory to store mask/labels,
                defaults to data_dir.
            metadata_dir: Optional directory to store metadata json,
                defaults to mask_labels_dir.
            numpy: True if the data_dir should use npy files.
            downsampling: The downsampling factor.
        """
        super().__init__()
        self._url = url
        self._auth = (username, password)
        self._volume_path = str(data.value)
        self._data_dir = data_dir
        self._mask_labels_dir = data_dir if mask_labels_dir is None else mask_labels_dir
        self._metadata_dir = (
            self._mask_labels_dir if metadata_dir is None else metadata_dir
        )
        self._numpy = numpy
        self._downsampling = downsampling

    def _load_local(self, path: str) -> np.ndarray:
        """Load the local data via numpy or pillow.

        Args:
            path: The path to the local file.

        Returns:
            The numpy array.
        """
        if path.endswith(".npy"):
            out = np.load(path)
        else:
            im = Image.open(path)
            out = np.array(im)
            del im
        return out[:: self._downsampling, :: self._downsampling]

    def _load_remote(self, url: str, save_local: Optional[str] = None) -> np.ndarray:
        """Load the image from remote, optionally saving a cached copy.

        Args:
            url: The url of the remote image file.
            save_local: Optional, the local directory to cache the downloaded file.
        """
        res = requests.get(url, auth=self._auth)
        assert res.status_code == 200, f"GET {url} = {res.status_code}"
        im = Image.open(io.BytesIO(res.content))
        out = np.array(im)

        if save_local is not None:
            local_path = path.join(save_local, url.rsplit("/", 1)[-1])
            if not path.exists(save_local):
                makedirs(save_local)
            if self._numpy:
                no_ext = local_path.rsplit(".", 1)[0]
                np.save(f"{no_ext}.npy", np.array(im))
            else:
                im.save(local_path)
        del im
        return out[:: self._downsampling, :: self._downsampling]

    def get_num_slices(self) -> int:
        metadata = self.resolve_metadata
        assert metadata is not None
        return metadata.num_slices

    def resolve_slice(self, slice: int, width: Optional[int] = None) -> np.ndarray:
        if width is None:
            # Try to infer from the number of slices
            metadata = self.resolve_metadata()
            if metadata is not None:
                width = int(np.log10(metadata.num_slices)) + 1
        slice_id = "{slice:0{width}d}".format(slice=slice, width=width)
        ext = "npy" if self._numpy else "tif"
        local_path = path.join(self._data_dir, f"{slice_id}.{ext}")
        if path.exists(local_path):
            out = self._load_local(local_path)
        else:
            remote_url = self._url + self._volume_path + f"/{slice_id}.tif"
            out = self._load_remote(remote_url, self._data_dir)
        return out

    def resolve_ink_labels(self) -> Optional[np.ndarray]:
        if "fragments" in self._volume_path and "working" in self._volume_path:
            ext = "npy" if self._numpy else "png"
            local_path = path.join(self._mask_labels_dir, f"inklabels.{ext}")
            if path.exists(local_path):
                out = self._load_local(local_path)
            else:
                root_fragment_url = self._volume_path.rsplit("/", maxsplit=1)[0]
                remote_url = f"{self._url}{root_fragment_url}/inklabels.png"
                out = self._load_remote(remote_url, self._mask_labels_dir)
            return out
        return None

    def resolve_mask(self) -> Optional[np.ndarray]:
        if "fragments" in self._volume_path and "working" in self._volume_path:
            ext = "npy" if self._numpy else "png"
            local_path = path.join(self._mask_labels_dir, f"mask.{ext}")
            if path.exists(local_path):
                out = self._load_local(local_path)
            else:
                root_fragment_url = self._volume_path.rsplit("/", maxsplit=1)[0]
                remote_url = f"{self._url}{root_fragment_url}/mask.png"
                out = self._load_remote(remote_url, self._mask_labels_dir)
            return out
        return None

    def resolve_metadata(self) -> Optional[VesuviusMetadata]:
        local_path = path.join(self._metadata_dir, "meta.json")
        if path.exists(local_path):
            with open(local_path, "r") as meta_file:
                content = json.loads(meta_file.read())
        else:
            metadata_url = f"{self._url}{self._volume_path}/meta.json"
            res = requests.get(metadata_url, auth=self._auth)
            content = res.json()

            if not path.exists(self._metadata_dir):
                makedirs(self._metadata_dir)
            with open(local_path, "w") as meta_file:
                meta_file.write(json.dumps(content, indent=2))

        return VesuviusMetadata(
            content["name"],
            content["slices"],
            content["voxelsize"] * 1e-3,
            content["uuid"],
            content["min"],
            content["max"],
            content["voxelsize"] * content["slices"] * 1e-3,
        )


class LocalResolver(Resolver):
    """Resolve from a local directory."""

    def __init__(
        self,
        data_dir: str,
        mask_labels_dir: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        numpy: bool = False,
        downsampling: int = 1,
    ):
        """
        Args:
            data_dir: The local folder for caching data.
            mask_labels_dir: Optional directory to store mask/labels,
                defaults to data_dir.
            metadata_dir: Optional directory to store metadata json,
                defaults to mask_labels_dir.
            numpy: True if the data_dir should use npy files.
            downsampling: The downsampling factor.
        """
        super().__init__()
        self._data_dir = data_dir
        self._mask_labels_dir = data_dir if mask_labels_dir is None else mask_labels_dir
        self._metadata_dir = (
            self._mask_labels_dir if metadata_dir is None else metadata_dir
        )
        self._numpy = numpy
        self._downsampling = downsampling

    def _load_local(self, path: str) -> np.ndarray:
        """Load the local data via numpy or pillow.

        Args:
            path: The path to the local file.

        Returns:
            The numpy array.
        """
        if path.endswith(".npy"):
            out = np.load(path)
        else:
            im = Image.open(path)
            out = np.array(im)
            del im
        return out[:: self._downsampling, :: self._downsampling]

    def get_num_slices(self) -> int:
        metadata = self.resolve_metadata()
        if metadata is None:
            # Infer from data directory
            ext = "npy" if self._numpy else "tif"
            data_files = glob.glob(path.join(self._data_dir, f"*.{ext}"))
            return len(data_files)
        else:
            return metadata.num_slices

    def resolve_slice(self, slice: int, width: Optional[int] = None) -> np.ndarray:
        ext = "npy" if self._numpy else "tif"
        if width is None:
            # Try to infer from local data directory
            data_files = glob.glob(path.join(self._data_dir, f"*.{ext}"))
            assert len(data_files) > 0, f"No local {ext} data in {self._data_dir}!"
            width = len(path.basename(data_files[0]).rsplit(".", 1)[0])
        slice_id = "{slice:0{width}d}".format(slice=slice, width=width)
        local_path = path.join(self._data_dir, f"{slice_id}.{ext}")
        return self._load_local(local_path)

    def resolve_ink_labels(self) -> Optional[np.ndarray]:
        ext = "npy" if self._numpy else "png"
        local_path = path.join(self._mask_labels_dir, f"inklabels.{ext}")
        if path.exists(local_path):
            return self._load_local(local_path)
        else:
            return None

    def resolve_mask(self) -> Optional[np.ndarray]:
        ext = "npy" if self._numpy else "png"
        local_path = path.join(self._mask_labels_dir, f"mask.{ext}")
        if path.exists(local_path):
            return self._load_local(local_path)
        else:
            return None

    def resolve_metadata(self) -> Optional[VesuviusMetadata]:
        local_path = path.join(self._metadata_dir, "meta.json")
        if path.exists(local_path):
            with open(local_path, "r") as meta_file:
                content = json.loads(meta_file.read())
            return VesuviusMetadata(
                content["name"],
                content["slices"],
                content["voxelsize"] * 1e-3,
                content["uuid"],
                content["min"],
                content["max"],
                content["voxelsize"] * content["slices"] * 1e-3,
            )
        return None


class Scroll:
    """A utility class for loading Vesuvius Challenge Data."""

    def __init__(
        self,
        resolver: Resolver,
        mask: Optional[np.ndarray] = None,
        ink_labels: Optional[np.ndarray] = None,
        metadata: Optional[VesuviusMetadata] = None,
    ):
        """
        Args:
            resolver: The slice resolver.
            downsampling: The downsampling factor to apply.
            mask: Optional fragment mask.
            ink_labels: Optional ink labels.
            metadata: The VesuviusMetadata for this scroll.
        """
        super().__init__()
        self.metadata = metadata
        self._resolver = resolver
        self._mask: Optional[np.ndarray] = mask
        self._ink_labels: Optional[np.ndarray] = ink_labels

    @staticmethod
    def from_remote(
        config_file: str,
        data: VesuviusData,
        data_dir: str,
        mask_labels_dir: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        downsampling: int = 1,
        numpy: bool = False,
    ) -> "Scroll":
        """Load the scroll from remote.

        Please create a config file
        with the data download information accessible after accepting
        the Vesuvius Challenge data license at https://scrollprize.org:

        ```ini
            [data]
            url=<root url>
            username=<username>
            password=<password>
        ```

        Args:
            config_file: Path to the config file with url and login info.
            data: The scroll/fragment to load.
            data_dir: The local directory to store cached slices.
            mask_labels_dir: Optional local directory to store cached
                mask and inklabels. Defaults to data_dir.
            metadata_dir: Optional directory to store metadata json,
                defaults to mask_labels_dir.
            downsampling: The downsampling factor.
            numpy: True if data is stored as numpy locally.
        """
        if not path.exists(config_file):
            raise RuntimeError(f"Unable to find {config_file}")

        config = configparser.ConfigParser()
        config.read(config_file)

        resolver = RemoteResolver(
            config["data"]["url"],
            config["data"]["username"],
            config["data"]["password"],
            data,
            data_dir,
            mask_labels_dir,
            metadata_dir,
            numpy,
            downsampling,
        )
        return Scroll(
            resolver,
            resolver.resolve_mask(),
            resolver.resolve_ink_labels(),
            resolver.resolve_metadata(),
        )

    @staticmethod
    def from_local(
        data_dir: str,
        mask_labels_dir: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        downsampling: int = 1,
        numpy: bool = False,
    ) -> "Scroll":
        """Load the scroll from local directories.

        Args:
            data_dir: The local directory to store cached slices.
            mask_labels_dir: Optional local directory to store cached
                mask and inklabels. Defaults to data_dir.
            metadata_dir: Optional directory to store metadata json,
                defaults to mask_labels_dir.
            downsampling: The downsampling factor.
            numpy: True if data is stored as numpy locally.
        """
        resolver = LocalResolver(
            data_dir, mask_labels_dir, metadata_dir, numpy, downsampling
        )
        return Scroll(
            resolver,
            resolver.resolve_mask(),
            resolver.resolve_ink_labels(),
            resolver.resolve_metadata(),
        )

    def __len__(self) -> int:
        return self._resolver.get_num_slices()

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

        If using `mm` values, the scroll *must* have metadata!

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

        if start_depth_mm is not None and self.metadata is not None:
            start_index = int(start_depth_mm / self.metadata.slice_height_mm)
        elif start_slice is not None:
            start_index = start_slice
        else:
            logger.warning(
                "Must provide one of start_depth_mm or start_slice. Defaulting to start_slice = 0"
            )

        end_index = start_index

        if end_depth_mm is not None and self.metadata is not None:
            end_index = int(end_depth_mm / self.metadata.slice_height_mm)
        elif size_mm is not None and self.metadata is not None:
            offset = int(size_mm / self.metadata.slice_height_mm)
            end_index = start_index + offset
        elif end_slice is not None:
            end_index = end_slice + 1
        elif num_slices is not None:
            end_index = start_index + num_slices
        elif to_end and self.metadata is not None:
            end_index = self.metadata.num_slices
        else:
            raise RuntimeError(
                "Must provide one of end_depth_mm, size_mm, end_slice, or num_slices. To download all data explicitly pass to_end=True"
            )

        slices = []
        for i in range(start_index, end_index):
            slices.append(self._resolver.resolve_slice(i))

        return np.stack(slices)
