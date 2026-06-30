"""Demo script for DonutBlitzMonolithTask running on real butler data.

Loads corner wavefront sensor raws matching a query, runs ISR and donut
detection in parallel via DonutBlitzMonolithTask.run(), and reports timing.
Supports multiple visits in a single query — the script loops over them,
reusing the same task instance (and persistent worker pool) across visits.

Usage
-----
    set -a && source .env
    python3 devel/blitz_demo.py "day_obs=20260216 and seq_num=185" -b main
    python3 devel/blitz_demo.py "day_obs=20260216 and seq_num=185" -b embargo -j 8
    python3 devel/blitz_demo.py "day_obs=20260216 and seq_num in (185, 186)" -b embargo -j 8

The query is passed to butler.query_datasets("raw", where=...).
Omitting the detector constraint is fine — the script filters to corner
sensors automatically.
"""

import argparse
import logging
import time
from contextlib import contextmanager
from functools import lru_cache

from lsst.daf.butler import Butler, DatasetNotFoundError
from lsst.ts.wep.task.donutBlitzMonolith import CORNER_SENSOR_NAMES, DonutBlitzMonolithTask, DonutBlitzMonolithTaskConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("blitz_demo")

CORNER_DET_IDS = {191, 192, 195, 196, 199, 200, 203, 204}


@contextmanager
def timer(label):
    t0 = time.time()
    try:
        yield
    finally:
        log.info("%s: %.3f s", label, time.time() - t0)


@lru_cache(maxsize=8)
def _get_det_calibs(butler, detector):
    """Fetch per-detector calibrations (cached across sensors if reused)."""
    ptc = butler.get("ptc", instrument="LSSTCam", detector=detector)
    linearizer = butler.get("linearizer", instrument="LSSTCam", detector=detector)
    crosstalk = butler.get("crosstalk", instrument="LSSTCam", detector=detector)
    return ptc, linearizer, crosstalk


@lru_cache(maxsize=8)
def _get_flat(butler, detector, physical_filter):
    return butler.get("flat", instrument="LSSTCam", detector=detector, physical_filter=physical_filter)


def load_inputs(butler, datasets):
    """Load all ISR inputs for the given raw datasets."""
    raws, ptcs, flats, linearizers, crosstalkCalib = [], [], [], [], []

    camera = butler.get("camera", instrument="LSSTCam")

    for dataset in datasets:
        det_id = dataset.dataId["detector"]
        raw = butler.get("raw", dataId=dataset.dataId)
        physical_filter = raw.info.getFilter().physicalLabel

        try:
            ptc, linearizer, crosstalk = _get_det_calibs(butler, det_id)
            flat = _get_flat(butler, det_id, physical_filter)
        except (DatasetNotFoundError, ValueError) as exc:
            log.warning("Skipping detector %d: %s", det_id, exc)
            continue

        raws.append(raw)
        ptcs.append(ptc)
        flats.append(flat)
        linearizers.append(linearizer)
        crosstalkCalib.append(crosstalk)

    return raws, camera, ptcs, flats, linearizers, crosstalkCalib


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="Butler where-clause, e.g. \"day_obs=20260216 and seq_num=185\"")
    parser.add_argument("-b", "--butler", default="main", help="Butler name/path (default: main)")
    parser.add_argument("-j", type=int, default=1, help="Number of parallel ISR processes (default: 1)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger("lsst").setLevel(logging.DEBUG)

    with timer("Butler init"):
        butler = Butler(args.butler, collections="LSSTCam/defaults", instrument="LSSTCam")

    with timer("Dataset query"):
        all_datasets = list(butler.query_datasets("raw", where=args.query))

    # Filter to corner sensors only, grouped by visit
    corner_datasets = [d for d in all_datasets if d.dataId["detector"] in CORNER_DET_IDS]
    visits_map: dict = {}
    for d in corner_datasets:
        visits_map.setdefault(d.dataId["exposure"], []).append(d)

    log.info(
        "Found %d corner sensor raws across %d visit(s) (of %d total matching datasets)",
        len(corner_datasets), len(visits_map), len(all_datasets),
    )

    if len(visits_map) == 0:
        log.error("No corner sensor raws found for query: %s", args.query)
        return

    config = DonutBlitzMonolithTaskConfig()
    task = DonutBlitzMonolithTask(config=config)

    for visit, datasets in sorted(visits_map.items()):
        log.info("--- Exposure %d (%d sensors) ---", visit, len(datasets))

        found_names = set()
        with timer(f"Loading ISR inputs (exposure {visit})"):
            raws, camera, ptcs, flats, linearizers, crosstalkCalib = load_inputs(butler, datasets)
            found_names = {exp.getDetector().getName() for exp in raws}

        missing = CORNER_SENSOR_NAMES - found_names
        if missing:
            log.warning("Missing corner sensors (will fail): %s", sorted(missing))

        log.info("Running DonutBlitzMonolithTask.run() with %d core(s)", args.j)
        with timer(f"DonutBlitzMonolithTask.run() (exposure {visit})"):
            result = task.run(
                raws=raws,
                camera=camera,
                ptc=ptcs,
                flat=flats,
                linearizer=linearizers,
                crosstalk=crosstalkCalib,
                numCores=args.j,
            )

        log.info("Exposure %d done. Total donuts: %d", visit, len(result.donuts))


if __name__ == "__main__":
    main()
