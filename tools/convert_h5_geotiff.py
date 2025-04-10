import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import rasterio
import rasterio.transform
from numpy.typing import NDArray


def shrink_slice(hand: NDArray[np.floating]):
    ibegin = 0
    for i in range(hand.shape[0]):
        if np.any(np.isfinite(hand[i, :])):
            ibegin = i
            break
    iend = hand.shape[0]
    for i in reversed(range(hand.shape[0])):
        if np.any(np.isfinite(hand[i, :])):
            iend = i + 1
            break

    jbegin = 0
    for j in range(hand.shape[1]):
        if np.any(np.isfinite(hand[:, j])):
            jbegin = j
            break
    jend = hand.shape[1]
    for j in reversed(range(hand.shape[1])):
        if np.any(np.isfinite(hand[:, j])):
            jend = j + 1
            break
    return slice(ibegin, iend), slice(jbegin, jend)


def convert(h5file: Path, h5var: str | None, geotiff: Path, shrink: bool = False):
    with h5py.File(h5file, "r") as f:
        lat = np.array(f["lat"][:], np.float64)  # type: ignore
        lon = np.array(f["lon"][:], np.float64)  # type: ignore
        if h5var is None:
            vars = set(str(x) for x in f.keys()) - {"lat", "lon", "crs"}
            if len(vars) != 1:
                raise ValueError(f"Multiple variables found: {vars}")
            h5var = vars.pop()
        data = np.array(f[h5var][:, :], np.float32)  # type: ignore
    if shrink:
        islice, jslice = shrink_slice(data)
        lat = lat[islice]
        lon = lon[jslice]
        data = data[islice, jslice]
    dlat = math.fabs((lat[-1] - lat[0]).item()) / (lat.size - 1)
    dlon = math.fabs((lon[-1] - lon[0]).item()) / (lon.size - 1)

    transform = rasterio.transform.from_origin(
        lon.min().item() - 0.5 * dlon, lat.max().item() + 0.5 * dlat, dlon, dlat
    )
    with rasterio.open(
        geotiff,
        "w",
        driver="GTiff",
        height=lat.size,
        width=lon.size,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        bigtiff="yes",
    ) as dst:
        dst.write(data, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5file", type=Path)
    parser.add_argument("geotiff", type=Path)
    parser.add_argument("-v", "--var", type=str)
    parser.add_argument("-s", "--shrink", action="store_true", default=False)
    args = parser.parse_args()
    var = str(args.var) if args.var is not None else None
    convert(args.h5file, var, args.geotiff, args.shrink)
