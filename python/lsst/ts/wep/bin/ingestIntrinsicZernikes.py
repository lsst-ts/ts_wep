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

from lsst.daf.butler import Butler, CollectionType, DatasetType, Timespan
from lsst.ip.isr import IntrinsicZernikes


def main() -> None:
    tz = time.strftime("%z")
    logging.basicConfig(
        format="%(levelname)s %(asctime)s.%(msecs)03d" + tz + " - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = ArgumentParser(
        description=(
            "Convert intrinsic_aberrations_temp astropy tables into "
            "lsst.ip.isr.IntrinsicZernikes calibrations, ingest them into a "
            "RUN collection, and (optionally) certify them into a "
            "CALIBRATION collection."
        )
    )
    parser.add_argument(
        "-b",
        "--butler-config",
        help="Location of the butler/registry config file (e.g. /repo/main).",
        required=True,
        metavar="REPO",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="LSSTCam",
        help="Instrument name. (default: LSSTCam)",
        metavar="INST",
    )
    parser.add_argument(
        "--input-collection",
        type=str,
        default="LSSTCam/aos/intrinsic",
        help=(
            "Collection containing the source intrinsic_aberrations_temp "
            "tables. (default: LSSTCam/aos/intrinsic)"
        ),
        metavar="COLL",
    )
    parser.add_argument(
        "--input-dataset-type",
        type=str,
        required=True,
        help="Dataset type to read from. (default: intrinsic_aberrations_temp)",
        metavar="TYPE",
    )
    parser.add_argument(
        "--output-dataset-type",
        type=str,
        default="intrinsicZernikes",
        help="Dataset type to write to. (default: intrinsicZernikes)",
        metavar="TYPE",
    )
    parser.add_argument(
        "--output-run",
        type=str,
        required=True,
        help=(
            "RUN collection to write the IntrinsicZernikes calibrations into. "
            "e.g. LSSTCam/calib/DM-55048/intrinsicZernikes.v1.0/run.<date>"
        ),
        metavar="COLL",
    )
    parser.add_argument(
        "--certify-into",
        type=str,
        default=None,
        help=(
            "If set, certify the ingested calibrations into this CALIBRATION "
            "collection with an unbounded timespan. e.g. "
            "LSSTCam/calib/DM-55048/intrinsicZernikes.v1.0"
        ),
        metavar="COLL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve everything and log the plan, but do not write or certify.",
    )
    args = parser.parse_args()

    butler = Butler.from_config(args.butler_config, writeable=not args.dry_run)

    # Register the output dataset type if it does not exist yet. Marking it
    # is_calibration=True allows it to live in CALIBRATION collections, but
    # does not require certification on its own.
    outputDatasetType = DatasetType(
        args.output_dataset_type,
        ("instrument", "detector", "physical_filter"),
        "IsrCalib",
        universe=butler.dimensions,
        isCalibration=True,
    )
    if not args.dry_run:
        logger.info(f"Registering dataset type {args.output_dataset_type!r} (if absent)")
        butler.registry.registerDatasetType(outputDatasetType)

    # Find every source table in the input collection.
    inputRefs = sorted(
        butler.registry.queryDatasets(
            args.input_dataset_type,
            collections=args.input_collection,
            instrument=args.instrument,
        ),
        key=lambda r: (r.dataId["detector"], r.dataId["physical_filter"]),
    )
    if not inputRefs:
        raise RuntimeError(
            f"No {args.input_dataset_type!r} datasets found in collection "
            f"{args.input_collection!r} for instrument {args.instrument!r}"
        )
    logger.info(f"Found {len(inputRefs)} source tables in {args.input_collection!r}")

    if not args.dry_run:
        logger.info(f"Registering output RUN collection {args.output_run!r}")
        butler.registry.registerCollection(args.output_run, CollectionType.RUN)

    # Convert each table to an IntrinsicZernikes calibration and ingest it
    # into the output RUN collection.
    outputRefs = []
    for ref in inputRefs:
        table = butler.get(ref)
        calib = IntrinsicZernikes(table=table)

        # Carry useful provenance into the calibration metadata.
        calib.updateMetadata(
            INSTRUME=args.instrument,
            DETECTOR=int(ref.dataId["detector"]),
            FILTER=str(ref.dataId["physical_filter"]),
        )

        logger.info(
            f"Ingesting detector={ref.dataId['detector']:>3d} filter={ref.dataId['physical_filter']!r}"
        )
        if args.dry_run:
            continue
        outputRef = butler.put(
            calib,
            outputDatasetType,
            dataId=ref.dataId,
            run=args.output_run,
        )
        outputRefs.append(outputRef)

    if args.certify_into is None:
        logger.info("Done. No certification requested.")
        return

    if args.dry_run:
        logger.info(f"[dry-run] Would certify {len(inputRefs)} refs into {args.certify_into!r}")
        return

    logger.info(f"Registering CALIBRATION collection {args.certify_into!r}")
    butler.registry.registerCollection(args.certify_into, CollectionType.CALIBRATION)

    logger.info(f"Certifying {len(outputRefs)} refs into {args.certify_into!r} with unbounded timespan")
    butler.registry.certify(args.certify_into, outputRefs, Timespan(None, None))
    logger.info("Done.")
