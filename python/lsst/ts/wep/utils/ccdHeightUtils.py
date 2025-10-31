# This file is part of ts_wep.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["LsstCamHeightMapBuilder"]

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from lsst.ts.wep.utils.ioUtils import resolveRelativeConfigPath
from scipy.spatial import Delaunay
from sklearn.neighbors import KNeighborsRegressor

mapDir = "policy:heightMaps/"


class LsstCamHeightMapBuilder:
    """Class to convert height map provided by camera team to format for WEP.

    Intended use:
        from lsst.ts.wep.utils import LsstCamHeightMapBuilder
        builder = LsstCamHeightMapBuilder()
        builder.processFits()
        builder.interpolateMap()
    The resulting interpolated height map is committed to git and loaded
    directly by WEP at runtime.
    """

    def __init__(
        self,
        rawFile: str = mapDir + "LSST_FP_cold_b_measurement_4col_bysurface.fits",
        procFile: str = mapDir + "LsstCam_focal_plane_heights.fits",
        interpFile: str = mapDir + "LsstCam_focal_plane_heights_interpolated.fits",
        interpGrid: np.ndarray = np.linspace(-320, +320, 1000),
    ) -> None:
        """Initialize LsstCamHeightMapBuilder.

        Parameters
        ----------
        rawFile : str
            Path to raw input FITS file from camera team.
        procFile : str
            File in which to save the table of processed focal plane heights as
            an Astropy table. Raw values are in this table, but the default
            values have been adjusted to set the median of the science sensors
            within 317mm of center to zero, and the CWFS offsets have been
            removed.
        interpFile : str
            Path at which to save the interpolated focal plane height map.
            This is also saved as an Astropy Table.
        interpGrid : np.ndarray
            1D array of x and y coordinates (in mm) for interpolation grid.
        """
        self.rawFile = Path(resolveRelativeConfigPath(rawFile))
        self.procFile = Path(resolveRelativeConfigPath(procFile))
        self.interpFile = Path(resolveRelativeConfigPath(interpFile))
        self.interpGrid = interpGrid

    def processFits(self, overWrite: bool = False) -> None:
        """Process FITS file containing focal plane height measurements.

        Reads focal plane height data from format provided by camera team
        and saves it in an Astropy table.

        Raw values are retained in this table, but the default values have been
        adjusted to set the median of the science sensors within 317mm of
        center to zero, and the CWFS offsets have been removed.

        Parameters
        ----------
        overWrite : bool, optional
            Whether to overwrite existing output file. (Default is False)
        """
        if self.procFile.exists() and not overWrite:
            print(f"File {self.procFile} already exists. Skipping processFits.")
            return

        # Extract data from input fits file and save in a table
        rows = []
        for hdu in fits.open(self.rawFile):
            if isinstance(hdu, fits.BinTableHDU):
                extName = hdu.header["EXTNAME"]
                if extName[0] == "R":
                    table = Table(hdu.data)
                    detector = extName[:3] + "_" + extName[3:]
                    detector = detector.replace("WFS", "SW")
                    for x, y, z_mod, z_meas in zip(
                        table["X_CCS"],
                        table["Y_CCS"],
                        table["Z_CCS_MODEL"],
                        table["Z_CCS_MEASURED"],
                    ):
                        rows.append([x, y, detector, z_mod, z_meas])
        data = Table(
            rows=rows,
            names=["x_ccs", "y_ccs", "detector", "z_model_raw", "z_measured_raw"],
            units=2 * ["mm"] + [None] + 2 * ["mm"],
        )

        # Set zero-point to median of science sensors within 317mm of center
        mask = np.sqrt(data["x_ccs"] ** 2 + data["y_ccs"] ** 2) < 317
        for i in range(len(data)):
            if data[i]["detector"][-3] != "S" or data[i]["detector"][-2] == "W":
                mask[i] = False
        data["z_model"] = data["z_model_raw"] - np.median(data[mask]["z_model_raw"])
        data["z_measured"] = data["z_measured_raw"] - np.median(
            data[mask]["z_measured_raw"]
        )

        # Remove CWFS offsets
        for i in range(len(data)):
            if data[i]["detector"][-3:] == "SW1":
                data[i]["z_model"] += 1.5
                data[i]["z_measured"] += 1.5
            elif data[i]["detector"][-3:] == "SW0":
                data[i]["z_model"] -= 1.5
                data[i]["z_measured"] -= 1.5

        print(f"Saving {self.procFile}")
        data.write(self.procFile, format="fits", overwrite=overWrite)

    def interpolateMap(
        self, overWrite: bool = False, mapType: str = "measured"
    ) -> np.ndarray:
        """Interpolate focal plane height map onto regular grid.

        Parameters
        ----------
        overWrite : bool, optional
            Whether to overwrite existing output file.
            (Default is False)
        mapType : str, optional
            Type of map to interpolate ("measured" or "model").
            (Default is "measured")
        """
        if self.interpFile.exists() and not overWrite:
            print(f"File {self.interpFile} already exists. Skipping interpolateMap.")
            return

        # Load the processed data table
        if not self.procFile.exists():
            raise FileNotFoundError(
                f"procFile {self.procFile} does not exist. "
                "Please run processFits first."
            )
        table = Table.read(self.procFile)

        # Create grids for interpolation
        X, Y = np.meshgrid(self.interpGrid, self.interpGrid)
        x = X.ravel()
        y = Y.ravel()
        z = np.zeros_like(x)

        # Loop over every detector
        for detector in np.unique(table["detector"]):
            # Get data for this detector
            data = table[table["detector"] == detector]

            # Determine which points are inside this chip
            mask = (x >= data["x_ccs"].min()) & (x <= data["x_ccs"].max())
            mask &= (y >= data["y_ccs"].min()) & (y <= data["y_ccs"].max())

            # Interpolate
            knn = KNeighborsRegressor(n_neighbors=3, weights="distance")
            points = np.column_stack((data["x_ccs"], data["y_ccs"]))
            values = data[f"z_{mapType}"]
            knn.fit(points, values)
            z[mask] = knn.predict(np.column_stack((x, y))[mask])

        # Set anything outside original convex hull to NaN
        hull_points = np.column_stack([table["x_ccs"], table["y_ccs"]])
        test_points = np.column_stack([x, y])
        deln = Delaunay(hull_points)
        inside = deln.find_simplex(test_points) >= 0
        z[~inside] = np.nan

        # Save in an Astropy table
        data = Table(
            rows=np.column_stack((x, y, z)),
            names=["x", "y", "z"],
            units=3 * ["mm"],
        )
        print(f"Saving {self.interpFile}")
        data.write(self.interpFile, format="fits", overwrite=overWrite)
