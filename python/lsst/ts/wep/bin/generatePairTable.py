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


from argparse import ArgumentParser

from lsst.daf.butler import Butler
from lsst.ts.wep.task.pairTask import ExposurePairer, ExposurePairerConfig


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--butler-config",
        help="Location of the butler/registry config file.",
        required=True,
        metavar="TEXT",
    )
    parser.add_argument(
        "-c", "--collection", help="Collection name.", required=True, metavar="NAME"
    )
    parser.add_argument(
        "-o",
        type=str,
        help="Output filename.",
        required=True,
        metavar="FILENAME",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        help="Name of the instrument.",
        required=True,
        metavar="INST",
    )
    parser.add_argument(
        "-d",
        "--data-query",
        type=str,
        help="User data selection expression.",
        required=True,
        metavar="QUERY",
    )
    parser.add_argument(
        "--timeThreshold",
        type=float,
        help="Maximum time difference between paired intra- and extra-focal exposures (s)",
        default=300.0,
        metavar="SEC",
    )
    parser.add_argument(
        "--pointingThreshold",
        type=float,
        help="Maximum pointing difference between paired intra- and extra-focal exposures (arcsec)",
        default=60.0,
        metavar="ARCSEC",
    )
    parser.add_argument(
        "--rotationThreshold",
        type=float,
        help="Maximum rotator angle difference between paired intra- and extra-focal exposures (deg)",
        default=1.0,
        metavar="DEG",
    )
    parser.add_argument(
        "--overrideSeparation",
        type=float,
        help="Expected intra-focal to focal separation (mm)",
        default=None,
        metavar="MM",
    )
    parser.add_argument(
        "--groupingThreshold",
        type=float,
        help="Threshold for assigning visit to intra/extra/focal group as a fraction of the expected"
        " intra-focal to focal separation.",
        default=0.1,
        metavar="FRAC",
    )
    parser.add_argument(
        "--forceUniquePairs",
        action="store_true",
        help="If True, force each extra exposure to be paired with a unique intra exposure.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    butler = Butler.from_config(args.butler_config, collections=args.collection)

    doOverrideSeparation = False
    overrideSeparation = 1.5
    if args.overrideSeparation is not None:
        overrideSeparation = args.overrideSeparation
        doOverrideSeparation = True

    config = ExposurePairerConfig(
        timeThreshold=args.timeThreshold,
        pointingThreshold=args.pointingThreshold,
        rotationThreshold=args.rotationThreshold,
        doOverrideSeparation=doOverrideSeparation,
        overrideSeparation=overrideSeparation,
        groupingThreshold=args.groupingThreshold,
        forceUniquePairs=args.forceUniquePairs,
    )
    pairer = ExposurePairer(config=config)

    visitInfos = {
        v.dataId["exposure"]: butler.get("raw.visitInfo", dataId=v.dataId)
        for v in butler.registry.queryDatasets("raw", where=args.data_query)
    }

    tables = pairer.makeTables(visitInfos)

    print()
    print("Found pairs")
    tables["pairTable"].pprint_all()
    print()
    print("Writing pairs to file: ", args.o)
    tables["pairTable"].write(args.o, format="ascii.ecsv", overwrite=True)

    if args.verbose:
        if len(tables["unusedExtraTable"]) > 0:
            print()
            print("Unused extra exposures")
            tables["unusedExtraTable"].pprint_all()
        if len(tables["unusedIntraTable"]) > 0:
            print()
            print("Unused intra exposures")
            tables["unusedIntraTable"].pprint_all()
        if len(tables["focalTable"]) > 0:
            print()
            print("Focal exposures")
            tables["focalTable"].pprint_all()
