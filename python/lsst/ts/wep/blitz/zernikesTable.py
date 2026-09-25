# This file is part of ts_wep.
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

"""The per-corner ``zernikes`` output table, for compatibility with the
non-blitz corner pipeline.

`CalcZernikesTask` emits one ``zernikes`` table per corner, dimensioned
``(visit, detector, instrument)`` and keyed on the extra-focal detector.  This
rebuilds that shape from blitz's per-donut catalog so the same consumers read
blitz output unchanged.

It is a deliberate *subset* of that schema rather than a clone.  Two
families of column are absent:

- The entropy / frac_bad_pix / max_power_grad quality metrics, which blitz
  does not compute.
- The total-OPD ``Z{j}`` and ``Z{j}_intrinsic`` columns.  Intrinsic
  Zernikes are a function of field position, so a joint fit spanning N
  donuts has N *different* intrinsics and no meaningful average; the OPD
  total inherits that.  The deviation is what is actually fit, so it is
  the only Zernike reported.  ``meta["opd_columns"]`` and
  ``meta["intrinsic_columns"]`` are still written, as empty lists, so
  that a consumer configured to read either one trips its own "no
  columns" guard instead of silently averaging nonsense.
"""

__all__ = ["build_zernikes_tables"]

import astropy.units as u
import numpy as np
from astropy.table import QTable

from lsst.ts.wep.task.calcZernikesTask import blurClipZkTable, pos2f_dtype

from .lsstCam import _LSSTCAM
from .utils import CORNER_BY_DET_NAME, CORNER_DEFOCAL_BY_DET_NAME, CORNER_PAIRS

# Label prefix per `wfEstimationMode`. A row is only a "pair" in paired
# mode; in the others it is one donut or a joint fit, and calling those
# "pair" would misdescribe them. Safe to vary because every reader in
# ts_wep tests `label == "average"` or `label != "average"` -- nothing
# matches on the prefix. Keep these short: the column is <U12, so the
# prefix plus the index must fit.
_LABEL_PREFIX = {
    "paired": "pair",
    "unpaired": "donut",
    "full_detector": "group",
    "full_corner": "group",
}

# The two sides of focus, as the column-name prefixes the schema uses.
_SIDES = ("intra", "extra")

# The structured (x, y) columns are fed bare arrays rather than
# Quantities: a QTable cannot unit-check a structured dtype, so passing
# one warns ("Units from inserted quantities will be ignored") even
# though the value lands correctly. The unit comes from the column
# definition in `_init_table`, so the values must already be in degrees
# (field) and pixels (centroid). Scalar Quantity columns are different --
# those *are* converted on insert, so the Zernikes below are handed
# `* u.nm` and would be converted if given µm.


def _empty_side_values() -> dict:
    """Column values for a side with no single donut to describe it.

    Used both where a side has no donuts at all (an unpaired row, or a
    one-sided corner) and where it has several (a joint fit), since in neither
    case is there one field position or centroid that the row can honestly
    report.
    """
    return {
        "field": np.array((np.nan, np.nan), dtype=pos2f_dtype),
        "centroid": np.array((np.nan, np.nan), dtype=pos2f_dtype),
        "mag": np.nan,
        "sn": np.nan,
        "donut_id": "",
    }


def _side_values(rows: QTable) -> dict:
    """Column values describing one side of focus of one fit group.

    Populated only when the side has exactly one donut; zero or several both
    yield `_empty_side_values`.  See the module docstring: a joint fit has no
    single field point, so reporting one donut's as the row's would imply a
    precision the fit does not have.
    """
    if len(rows) != 1:
        return _empty_side_values()

    row = rows[0]
    # The schema's field angles are DVCS, which is CCS with x and y swapped
    # (sitcomtn-003); blitz stores CCS. `DonutStamp.calcFieldXY` is the
    # non-blitz source of these and returns DVCS, as confirmed by
    # CalcZernikesTask unpacking it into (ccs_y, ccs_x).
    thx_ccs = row["thx_ccs"].to_value(u.deg)
    thy_ccs = row["thy_ccs"].to_value(u.deg)
    return {
        "field": np.array((thy_ccs, thx_ccs), dtype=pos2f_dtype),
        "centroid": np.array(
            (row["x_det"].to_value(u.pix), row["y_det"].to_value(u.pix)),
            dtype=pos2f_dtype,
        ),
        "mag": float(row["photo_mag"].to_value(u.mag)),
        "sn": float(row["snr"]),
        "donut_id": str(row["donut_id"]),
    }


def _init_table(noll_indices) -> QTable:
    """An empty table of the output schema, with its leading average row.

    Follows `CalcZernikesTask.initZkTable`, minus the columns this output does
    not carry.  The ``average`` row comes first and is all-NaN until a combine
    task fills it; exactly one leading average row is required, since
    `blurClipZkTable` indexes the per-row fwhm list with ``useIdx - 1`` and the
    fit-failure NaN-out uses ``failIdx + 1``.
    """
    dtype: list[tuple] = [("label", "<U12"), ("used", np.bool_)]
    for side in _SIDES:
        dtype.append((f"{side}_field", pos2f_dtype))
        dtype.append((f"{side}_centroid", pos2f_dtype))
    for side in _SIDES:
        dtype.append((f"{side}_mag", "<f4"))
        dtype.append((f"{side}_sn", "<f4"))
        dtype.append((f"{side}_donut_id", "<U21"))
    for j in noll_indices:
        dtype.append((f"Z{j}_deviation", "<f4"))

    table = QTable(dtype=dtype)
    for side in _SIDES:
        table[f"{side}_field"].unit = u.deg
        table[f"{side}_centroid"].unit = u.pixel
    for j in noll_indices:
        table[f"Z{j}_deviation"].unit = u.nm

    empty = {side: _empty_side_values() for side in _SIDES}
    table.add_row(
        {
            "label": "average",
            "used": True,
            **{f"Z{j}_deviation": np.nan * u.nm for j in noll_indices},
            **{f"{side}_{key}": value for side in _SIDES for key, value in empty[side].items()},
        }
    )
    return table


def _table_metadata(noll_indices, det_names_by_side, visit_id, cam_name, band, visit_info) -> dict:
    """Metadata for one corner's table, following ``createZkTableMetadata``.

    ``opd_columns`` and ``intrinsic_columns`` are deliberately empty; see the
    module docstring.  All three keys are present regardless, because
    `CombineZernikesSigmaClipTask` selects one of them by config.
    """
    meta: dict = {}
    dfc_dist = _LSSTCAM.defocal_offset * 1e3
    for side in _SIDES:
        det_name = det_names_by_side[side]
        if det_name is None:
            meta[side] = {}
            continue
        entry = {
            "det_name": det_name,
            "visit": visit_id,
            "dfc_dist": dfc_dist,
            "band": band,
        }
        if visit_info is not None:
            entry["boresight_rot_angle_rad"] = visit_info.boresightRotAngle.asRadians()
            entry["boresight_par_angle_rad"] = visit_info.boresightParAngle.asRadians()
            az_alt = visit_info.boresightAzAlt
            entry["boresight_alt_rad"] = az_alt.getLatitude().asRadians()
            entry["boresight_az_rad"] = az_alt.getLongitude().asRadians()
            ra_dec = visit_info.boresightRaDec
            entry["boresight_ra_rad"] = ra_dec.getLongitude().asRadians()
            entry["boresight_dec_rad"] = ra_dec.getLatitude().asRadians()
            entry["mjd"] = visit_info.date.toAstropy().tai.mjd
        meta[side] = entry

    noll_list = [int(j) for j in noll_indices]
    meta["cam_name"] = cam_name
    meta["noll_indices"] = noll_list
    meta["opd_columns"] = []
    meta["intrinsic_columns"] = []
    meta["deviation_columns"] = [f"Z{j}_deviation" for j in noll_list]
    return meta


def _groups_by_corner(catalog: QTable) -> dict[str, dict[str, QTable]]:
    """Fit groups from the catalog, bucketed by corner.

    One entry per ``group_id``, so a joint fit spanning N donuts appears
    once rather than N times: the catalog replicates every ``group_*``
    value onto each member row, and carrying that through would inflate
    the donut count and hand sigma clipping N identical samples.

    Rows with an empty ``group_id`` are skipped -- no fit consumed them,
    so they have no Zernikes to report.  A group is assigned to the corner
    of its first recognized detector, which is what makes
    ``full_detector`` work: its groups span a single detector rather than
    a corner pair.
    """
    by_corner: dict[str, dict[str, QTable]] = {corner: {} for corner in CORNER_PAIRS}
    if len(catalog) == 0:
        return by_corner

    group_ids = np.asarray(catalog["group_id"], dtype=str)
    for group_id in dict.fromkeys(group_ids):  # preserves first-seen order
        if not group_id:
            continue
        rows = catalog[group_ids == group_id]
        corner = None
        for det_name in np.asarray(rows["det_name"], dtype=str):
            corner = CORNER_BY_DET_NAME.get(str(det_name))
            if corner is not None:
                break
        if corner is None:
            continue
        by_corner[corner][group_id] = rows
    return by_corner


def _extra_focal_det_id(rows: QTable) -> int | None:
    """The SW0 (extra-focal) detector id for the corner these rows belong to.

    The corner's two detector ids are always adjacent with SW0 first, so
    either side answers for the pair: an intra-focal row gives it away as
    ``det_id - 1``.  That matters because a corner whose SW0 detector
    contributed no donuts still needs a key to be written under.
    """
    for det_name, det_id in zip(
        np.asarray(rows["det_name"], dtype=str), np.asarray(rows["det_id"], dtype=int)
    ):
        side = CORNER_DEFOCAL_BY_DET_NAME.get(str(det_name))
        if side == "extra":
            return int(det_id)
        if side == "intra":
            return int(det_id) - 1
    return None


def build_zernikes_tables(
    catalog: QTable,
    noll_indices,
    wf_mode: str,
    combine_zernikes,
    visit_id: int,
    cam_name: str = "",
    visit_info=None,
    do_blur_clip: bool = True,
    blur_clip_min_rows: int = 3,
    log=None,
) -> dict[int, QTable]:
    """Build one ``zernikes`` table per corner from the per-donut catalog.

    Parameters
    ----------
    catalog : `astropy.table.QTable`
        The per-donut catalog from
        `lsst.ts.wep.blitz.catalogBuilder._build_donut_catalog`.
    noll_indices : sequence of `int`
        The Noll indices that were fitted, from the fitting subtask's
        config, so the columns cannot drift from the fit.
    wf_mode : `str`
        The ``wfEstimationMode`` the fits ran under.  Selects the row label
        prefix; the row *content* rules do not branch on it.
    combine_zernikes : `lsst.pipe.base.Task`
        The combine subtask, run per corner to fill the average row.
    visit_id : `int`
        The visit these tables are for.
    cam_name : `str`, optional
        Camera name for ``meta["cam_name"]``.
    visit_info : `lsst.afw.image.VisitInfo`, optional
        Read for the boresight angles and mjd in ``meta``.
    do_blur_clip : `bool`, optional
        Sigma clip donuts with outlier blur.
    blur_clip_min_rows : `int`, optional
        Minimum data rows for blur clipping to run.  Below this there is
        nothing to clip against, and running it anyway would replace the
        configured combine's average with an unweighted mean for no gain.
    log : optional
        Logger for the per-corner summary.

    Returns
    -------
    dict [`int`, `astropy.table.QTable`]
        One table per corner that contributed at least one fit, keyed by
        the corner's extra-focal (SW0) detector id.  Corners absent from
        the catalog are omitted entirely rather than written empty.
    """
    noll_indices = [int(j) for j in noll_indices]
    prefix = _LABEL_PREFIX.get(wf_mode, "group")
    deviation_columns = [f"Z{j}_deviation" for j in noll_indices]
    band = str(catalog["band"][0]) if len(catalog) > 0 else ""

    tables: dict[int, QTable] = {}
    for corner, groups in _groups_by_corner(catalog).items():
        if not groups:
            continue

        sw0, sw1 = CORNER_PAIRS[corner]
        det_names_by_side = {"extra": None, "intra": None}
        det_id = None
        table = _init_table(noll_indices)
        fwhm: list[float] = []
        fit_success: list[bool] = []

        for i, rows in enumerate(groups.values()):
            if det_id is None:
                det_id = _extra_focal_det_id(rows)

            det_names = np.asarray(rows["det_name"], dtype=str)
            sides = np.array([CORNER_DEFOCAL_BY_DET_NAME.get(str(n), "") for n in det_names])
            for side in _SIDES:
                side_rows = rows[sides == side]
                if len(side_rows) > 0:
                    det_names_by_side[side] = sw1 if side == "intra" else sw0

            row: dict = {"label": f"{prefix}{i + 1}", "used": False}
            for side in _SIDES:
                for key, value in _side_values(rows[sides == side]).items():
                    row[f"{side}_{key}"] = value

            # Every member row of a group carries the same replicated fit, so
            # the first answers for all of them.
            zk_deviation = np.asarray(rows["zk_deviation_ccs"][0].to_value(u.micron), dtype=float)
            for j, col in zip(noll_indices, deviation_columns):
                value = zk_deviation[j] if j < len(zk_deviation) else np.nan
                row[col] = value * 1e3 * u.nm
            table.add_row(row)

            fwhm.append(float(rows["group_fwhm"][0].to_value(u.arcsec)))
            fit_success.append(bool(rows["group_fit_success"][0]))

        if det_id is None:
            continue

        table.meta = _table_metadata(noll_indices, det_names_by_side, visit_id, cam_name, band, visit_info)
        table.meta["estimatorInfo"] = {"fwhm": fwhm, "fit_success": fit_success}

        tables[det_id] = _finalize_table(
            table,
            deviation_columns,
            combine_zernikes,
            do_blur_clip=do_blur_clip,
            blur_clip_min_rows=blur_clip_min_rows,
        )
        if log is not None:
            log.info(
                "zernikes[%d] (%s): %d rows, %d successful fits",
                det_id,
                corner,
                len(groups),
                int(np.sum(fit_success)),
            )

    return tables


def _finalize_table(
    table: QTable,
    deviation_columns,
    combine_zernikes,
    do_blur_clip: bool,
    blur_clip_min_rows: int,
) -> QTable:
    """NaN out failed fits, fill the average row, then blur clip.

    Mirrors the tail of `CalcZernikesTask.run`.
    """
    fit_success = np.asarray(table.meta["estimatorInfo"]["fit_success"], dtype=bool)
    n_rows = len(fit_success)

    if not fit_success.any():
        # Nothing to average. The average row stays NaN and is marked unused,
        # but the table is still emitted: consumers read it like any other, and
        # a missing average row would be an IndexError for them.
        table["used"][table["label"] == "average"] = False
        table.meta["estimatorInfo"]["blur_clipped"] = [False] * n_rows
        return table

    # +1 to skip the average row.
    for idx in np.where(~fit_success)[0] + 1:
        for col in deviation_columns:
            table[col][idx] = np.nan

    table = combine_zernikes.run(table).combinedTable

    if do_blur_clip and n_rows >= blur_clip_min_rows:
        table = blurClipZkTable(table)
    else:
        # Written even when skipped, so the key is present and uniformly shaped
        # in every mode. Below the threshold mad_std cannot flag anything, and
        # blurClipZkTable would additionally overwrite the configured combine's
        # average with an unweighted mean for no gain.
        table.meta["estimatorInfo"]["blur_clipped"] = [False] * n_rows

    return table
