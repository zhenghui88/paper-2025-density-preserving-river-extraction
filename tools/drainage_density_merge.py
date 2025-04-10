import argparse
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import rasterio
from numpy.typing import NDArray


def latlon_slice(
    latlon: NDArray[np.floating], latlon_limit: tuple[float, float] | None
):
    assert latlon.ndim == 1, "Latitude/longitude must be 1D array"
    if latlon_limit is None:
        return slice(None)
    else:
        index = set()
        for ii, ll in enumerate(latlon):
            if latlon_limit[0] <= ll <= latlon_limit[1]:
                index.add(ii)
        index_min = min(index)
        index_max = max(index)
        return slice(
            index_min - 1 if index_min > 1 else None,
            index_max + 2 if index_max < latlon.size - 2 else None,
        )


def merge_drainage_density(
    modeled_file: Path, surveyed_file: Path, surveymask_file: Path, output_file: Path
):
    with (
        h5py.File(modeled_file, "r") as modeled,
        h5py.File(surveyed_file, "r") as surveyed,
    ):
        lat = cast(h5py.Dataset, modeled["lat"][...])
        lon = cast(h5py.Dataset, modeled["lon"][...])
        with rasterio.open(surveymask_file, "r") as src:
            surveymask = np.array(src.read(1), np.uint8)
        modeled_dd = cast(h5py.Dataset, modeled["den"][...])
        surveyed_dd = cast(h5py.Dataset, surveyed["den"][...])
        nanmask = np.logical_or(
            np.isnan(surveyed_dd),
            np.logical_and(np.logical_not(surveymask), surveyed_dd == 0),
        )
        latslice = latlon_slice(lat, (21.733, 25.365))
        lonslice = latlon_slice(lon, (118.835, 122.267))
        nanmask[latslice, lonslice] = True
        merged = np.where(nanmask, modeled_dd, surveyed_dd)
    with h5py.File(output_file, "w") as fo:
        latds = fo.create_dataset("lat", dtype=np.float64, compression="gzip", data=lat)
        latds.attrs.update(
            units="degrees_north".encode("ascii"),
            standard_name="latitude".encode("ascii"),
        )
        londs = fo.create_dataset("lon", dtype=np.float64, compression="gzip", data=lon)
        londs.attrs.update(
            units="degrees_east".encode("ascii"),
            standard_name="longitude".encode("ascii"),
        )
        crsds = fo.create_dataset("crs", dtype=np.uint8)
        crsds.attrs.update(
            grid_mapping_name="latitude_longitude".encode("ascii"),
            crs_wkt="""
GEOGCRS["WGS 84",
    ENSEMBLE["World Geodetic System 1984 ensemble",
        MEMBER["World Geodetic System 1984 (Transit)"],
        MEMBER["World Geodetic System 1984 (G730)"],
        MEMBER["World Geodetic System 1984 (G873)"],
        MEMBER["World Geodetic System 1984 (G1150)"],
        MEMBER["World Geodetic System 1984 (G1674)"],
        MEMBER["World Geodetic System 1984 (G1762)"],
        MEMBER["World Geodetic System 1984 (G2139)"],
        ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],
        ENSEMBLEACCURACY[2.0]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["degree",0.0174532925199433]],
    CS[ellipsoidal,2],
        AXIS["geodetic latitude (Lat)",north,
            ORDER[1],
            ANGLEUNIT["degree",0.0174532925199433]],
        AXIS["geodetic longitude (Lon)",east,
            ORDER[2],
            ANGLEUNIT["degree",0.0174532925199433]],
    USAGE[
        SCOPE["Horizontal component of 3D system."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["EPSG",4326]]""".encode("ascii"),
        )

        # copy den
        den = np.array(merged, np.float32)  # type: ignore
        dends = fo.require_dataset(
            "den",
            shape=den.shape,
            dtype=np.float32,
            compression="gzip",
            chunks=(4096, 4096) if den.size > 1e8 else None,
            data=den,
        )
        dends.dims[0].attach_scale(fo["lat"])
        dends.dims[1].attach_scale(fo["lon"])
        dends.attrs.update(
            long_name="drainage density".encode("ascii"),
            _FillValue=np.nan,
            units="m m-2".encode("ascii"),
            grid_mapping="crs".encode("ascii"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modeled", type=Path)
    parser.add_argument("surveyed", type=Path)
    parser.add_argument("survey_mask", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    merge_drainage_density(args.modeled, args.surveyed, args.survey_mask, args.output)
