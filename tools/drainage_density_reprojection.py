import argparse
from pathlib import Path

import numpy as np
import rasterio
import rasterio.crs
import rasterio.transform
import rasterio.warp


def resample_raster(srcraster: Path, desgrid: Path, desraster: Path):
    with rasterio.open(desgrid, "r") as f:
        des_crs = f.crs
        des_transform = f.transform
        des_width = f.width
        des_height = f.height

    # read source data
    with rasterio.open(srcraster, "r") as f:
        src_data = f.read(1)
        src_transform = f.transform
        src_crs = f.crs
        src_nodata = f.nodata
    src_nodata = -1.0
    src_data[src_data < 0.0] = src_nodata

    # reprojection
    dst_data = np.full((des_height, des_width), np.nan, np.float32)
    rasterio.warp.reproject(
        src_data,
        dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=des_transform,
        dst_crs=des_crs,
        dst_nodata=np.nan,
        resampling=rasterio.enums.Resampling.bilinear,
    )
    dst_data *= 1e-3  # convert km/km2 to m/m2

    # write destination data
    with rasterio.open(
        desraster,
        "w",
        driver="GTiff",
        height=des_height,
        width=des_width,
        count=1,
        dtype=np.float32,
        nodata=np.nan,
        crs=des_crs,
        transform=des_transform,
        compress="deflate",
    ) as f:
        f.write(dst_data, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srcraster", type=Path)
    parser.add_argument("desgrid", type=Path)
    parser.add_argument("desraster", type=Path)
    args = parser.parse_args()

    resample_raster(args.srcraster, args.desgrid, args.desraster)
