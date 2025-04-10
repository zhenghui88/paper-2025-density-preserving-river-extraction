import argparse
from pathlib import Path

import h5py
import numba
import numpy as np


@numba.njit
def mask_d8_outlet(d8, den):
    for ii in range(1, d8.shape[0] - 1):
        for jj in range(1, d8.shape[1] - 1):
            if ~np.isfinite(den[ii, jj]):
                continue
            if d8[ii, jj] == 1 and np.isnan(den[ii, jj + 1]):
                d8[ii, jj] = 0
            elif d8[ii, jj] == 2 and np.isnan(den[ii + 1, jj + 1]):
                d8[ii, jj] = 0
            elif d8[ii, jj] == 4 and np.isnan(den[ii + 1, jj]):
                d8[ii, jj] = 0
            elif d8[ii, jj] == 8 and np.isnan(den[ii + 1, jj - 1]):
                d8[ii, jj] = 0
            elif d8[ii, jj] == 16 and np.isnan(den[ii, jj - 1]):
                d8[ii, jj] = 0
            elif d8[ii, jj] == 32 and np.isnan(den[ii - 1, jj - 1]):
                d8[ii, jj] = 0
            elif d8[ii, jj] == 64 and np.isnan(den[ii - 1, jj]):
                d8[ii, jj] = 0
            elif d8[ii, jj] == 128 and np.isnan(den[ii - 1, jj + 1]):
                d8[ii, jj] = 0


def main(d8_file: Path, den_file: Path):
    with h5py.File(d8_file, "a") as fd8, h5py.File(den_file, "r") as fupl:
        den = fupl["den"][...]
        d8 = fd8["dir"][...]
        upa = fd8["upa"][...]
        d8[~np.isfinite(den)] = 0
        mask_d8_outlet(d8, den)
        upa[~np.isfinite(den)] = np.nan
        fd8["dir"][:] = d8
        fd8["upa"][:] = upa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask flow direction")
    parser.add_argument("d8_file", type=str, help="Path to the D8 file")
    parser.add_argument("den_file", type=str, help="Path to the drainage density file")
    args = parser.parse_args()
    main(args.d8_file, args.den_file)
