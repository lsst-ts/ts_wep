# This file is part of ts-wep.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# mypy: ignore-errors
from lsst.daf.butler import Butler, DatasetType, Timespan
from lsst.ip.isr import IntrinsicZernikes

butler = Butler("./", instrument="LSSTCam", collections="LSSTCam/aos/intrinsic", writeable=True)

intrinsic_zernikes_type = DatasetType(
    "intrinsicZernikes",
    ("instrument", "physical_filter", "detector"),
    "IsrCalib",
    universe=butler.dimensions,
    isCalibration=True,
)
butler.registry.registerDatasetType(intrinsic_zernikes_type)

ds_src = butler.query_datasets("intrinsic_aberrations_temp")
ds_dest = butler.query_datasets("intrinsicZernikes", explain=False)
for dataset in ds_src:
    if dataset.dataId in [dsd.dataId for dsd in ds_dest]:
        continue
    data = butler.get(dataset)
    izk = IntrinsicZernikes(table=data)
    ref = butler.put(izk, "intrinsicZernikes", dataId=dataset.dataId, run="LSSTCam/aos/intrinsic")
    butler.registry.certify(
        "LSSTCam/calib",
        [ref],
        timespan=Timespan(None, None),
    )
