#!/usr/bin/env python

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

import argparse

from lsst.ts.wep.CreatePhosimDonutTemplates import CreatePhosimDonutTemplates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate donut templates for AOS using Phosim."
    )
    parser.add_argument(
        "--numOfProc",
        type=int,
        default=1,
        help="Number of processor to run PhoSim. (default: 1)",
    )
    parser.add_argument(
        "--detectorList",
        type=str,
        default="",
        help="""
        Specify detectors.
        By default will generate templates for all detectors.
        (Example Input: "R22_S00 R22_S01 R22_S02")
        """,
    )
    parser.add_argument(
        "--templateSize",
        type=int,
        default=240,
        help="Size of each side of the template in pixels. (default: 240)",
    )
    parser.add_argument(
        "--intraVisitId",
        type=int,
        default=9006002,
        help="Visit ID for phosim intrafocal images. (default: 9006002)",
    )
    parser.add_argument(
        "--extraVisitId",
        type=int,
        default=9006001,
        help="Visit ID for phosim extrafocal images. (default: 9006001)",
    )
    args = parser.parse_args()

    # Run tasks
    phosimDonuts = CreatePhosimDonutTemplates()
    phosimDonuts.createWorkDirectories()
    decListPhosim, decListFlats = phosimDonuts.createDetectorLists(
        detectorStr=args.detectorList
    )
    phosimDonuts.generateDefocalImages(decListPhosim, args.numOfProc)
    phosimDonuts.repackagePhosimImages()
    phosimDonuts.ingestImages()
    phosimDonuts.makeFlats(decListFlats)
    phosimDonuts.ingestCalibs()
    phosimDonuts.runISR()
    phosimDonuts.cutOutIntraExtraTemplates(
        args.templateSize, args.intraVisitId, args.extraVisitId
    )
    phosimDonuts.cleanUpWorkDirs()
