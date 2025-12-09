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


import logging
import time
from argparse import ArgumentParser

from astropy.table import Table

from lsst.daf.butler import Butler, CollectionType, DatasetType, DimensionUniverse


def main() -> None:
    tz = time.strftime("%z")
    logging.basicConfig(
        format="%(levelname)s %(asctime)s.%(msecs)03d" + tz + " - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--butler-config",
        help="Location of the butler/registry config file.",
        required=True,
        metavar="TEXT",
    )
    parser.add_argument(
        "-o",
        "--output-collection",
        type=str,
        help="Name of the output collection to ingest the injection catalog into.",
        required=True,
        metavar="COLL",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        help="Name of the instrument.",
        required=True,
        metavar="INST",
    )
    parser.add_argument(
        "pairs",
        type=str,
        help="Path to the pair table to ingest.",
    )
    args = parser.parse_args()

    logger.info(f"Reading pair table from {args.pairs}")
    table = Table.read(args.pairs)

    logger.info(f"Writing pair table to butler collection {args.output_collection}")
    butler = Butler.from_config(args.butler_config, writeable=True)

    # Register if needed
    logger.info("Registering donutVisitPairTable dataset type")
    donutVisitPairTableDatasetType = DatasetType(
        "donutVisitPairTable",
        tuple(["instrument"]),
        "AstropyTable",
        universe=DimensionUniverse(),
    )
    butler.registry.registerDatasetType(donutVisitPairTableDatasetType)
    logger.info("Registering donutVisitPairTable collection")
    butler.registry.registerCollection(name=args.output_collection, type=CollectionType.RUN)

    logger.info("Ingesting pair table")
    butler.put(
        table,
        donutVisitPairTableDatasetType,
        dict(instrument=args.instrument),
        run=args.output_collection,
    )
    logger.info("Done")
