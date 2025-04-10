import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import rasterio
import rasterio.crs
import rasterio.transform


def resample_raster(desraster: Path, desgrid: Path):
    # read destination grid information
    with h5py.File(desgrid, "r") as f:
        lat = np.array(f["lat"][:], np.float64)
        lon = np.array(f["lon"][:], np.float64)
    dlat = math.fabs((lat[-1] - lat[0]).item()) / (lat.size - 1)
    dlon = math.fabs((lon[-1] - lon[0]).item()) / (lon.size - 1)

    des_crs = rasterio.crs.CRS.from_epsg(4326)
    des_transform = rasterio.transform.from_origin(
        lon.min().item() - 0.5 * dlon, lat.max().item() + 0.5 * dlat, dlon, dlat
    )

    data = np.zeros((lat.size, lon.size), np.float32)

    # write destination data
    with rasterio.open(
        desraster,
        "w",
        driver="GTiff",
        height=lat.size,
        width=lon.size,
        count=1,
        dtype=np.float32,
        crs=des_crs,
        transform=des_transform,
        compress="deflate",
    ) as f:
        f.write(data, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("desraster", type=Path)
    parser.add_argument("grid", type=Path)
    args = parser.parse_args()

    resample_raster(args.desraster, args.grid)
