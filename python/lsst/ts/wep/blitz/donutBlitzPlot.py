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

"""Diagnostic plots regenerated from the ``donutBlitzCornerResults`` table."""

__all__ = [
    "DonutBlitzPlotConnections",
    "DonutBlitzPlotConfig",
    "DonutBlitzPlotTask",
]

from dataclasses import dataclass, replace
from typing import Any

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    QuantumContext,
)

from .utils import (
    _CUTOUT_STAGE_KEYS,
    _MAX_NEARBY,
    CORNER_BY_DET_NAME,
    CORNER_DEFOCAL_BY_DET_NAME,
    CORNER_PAIRS,
    _resolve_color_log_enabled,
)

# Stand-in for "this record has no Zernikes": rows the fitter never produced
# deviations for (unfitted surplus donuts, blank layout padding). Noll-indexed
# like the real thing, just empty, so `_draw_bar` needs no special case.
_NO_ZK = np.zeros(0)

# Colors below are drawn from the colorblind-safe Okabe-Ito palette.
_COLOR_APERTURE = "#56B4E9"
_COLOR_BKG_ANNULUS = "#E69F00"
_COLOR_REJECTED = "#D55E00"
_COLOR_PHOTO_REFCAT = "#56B4E9"
_COLOR_ASTROM_REFCAT = "#E69F00"
_COLOR_CMAP_NEG = "#0072B2"
_COLOR_CMAP_MID = "#56B4E9"
_COLOR_CMAP_POS = "#D55E00"
_COLOR_COMA = "#F0E442"
_COLOR_ASTIGMATISM = "#E69F00"
_COLOR_TREFOIL = "#009E73"
_COLOR_QUADRAFOIL = "#0072B2"
_COLOR_PENTAFOIL = "#CC79A7"
_COLOR_HEXAFOIL = "#D55E00"
# The focal-plane plot's selection funnel, from widest to narrowest stage.
# `_COLOR_PHOTO_REFCAT` is reused for the refcat overlay: it already means
# exactly that on the donut stamps, so the two plots agree. The rest stay
# within the Okabe-Ito palette.
_COLOR_DETECTION = "#0072B2"
_COLOR_SELECTION = "#CC79A7"
_COLOR_ACCEPTED = "#009E73"
# Deliberately *not* `_COLOR_PHOTO_REFCAT`, which the refcat overlay already
# uses: a link drawn in that color is invisible among the hundreds of refcat
# circles it crosses. Yellow is the palette's most distinct remaining entry and
# reads against both the grey image and the other overlays.
_COLOR_PAIR_LINK = "#F0E442"
_COLOR_FIELD_LIMIT = "#E69F00"

# The funnel's marker style per stage, as
# ``(kind, color, marker, size, alpha, linewidth)``, widest stage first so the
# narrow ones draw on top. Size grows along the funnel so the stages nest as
# visibly concentric rings rather than competing at one scale.
_FP_FUNNEL_STYLES = (
    ("refcat", _COLOR_PHOTO_REFCAT, "o", 15.0, 0.55, 0.9),
    ("detection", _COLOR_DETECTION, "o", 46.0, 0.85, 0.7),
    ("selection", _COLOR_SELECTION, "o", 78.0, 0.95, 0.9),
)

# How a donut that survived the quality cut is marked, as
# ``(marker, size, linewidth)``.
#
# A diamond rather than another circle, and larger than the widest funnel
# stage.
_FP_ACCEPTED_MARKER = ("D", 165.0, 1.4)

# Annotation text on a donut stamp: the per-donut stats block and the refcat
# overlay labels. Small because a donut plot packs one row per detector and
# every stamp carries its own text.
_STAMP_TEXT_FONTSIZE = 3.5

# How much of a stage error to show in a detector's stats panel. The panel is
# one narrow column of monospace text, so the full message (up to
# `dataStructures._ERROR_MAX_CHARS`) would overrun its neighbours; the
# untrimmed string is in the catalog's det_meta for anyone who needs it.
_PANEL_ERROR_CHARS = 40


@dataclass(frozen=True)
class _DonutLayout:
    """Figure geometry for the donut diagnostic plot, in inches and columns.

    One record per plot rather than a flat block of module constants, because
    both plots have a ``row_h`` and they are unrelated numbers -- see
    `_WfLayout`.  Grid *indices* derived from these (the accepted/rejected
    column offsets) stay local to the method that builds the GridSpec: they
    are not tunable, they are consequences.
    """

    stamps_per_row: int
    rejected_per_row: int
    stamp_col_w: float
    stats_col_w: float
    row_h: float
    legend_h: float
    spacer_w: float
    suptitle_h: float

    @property
    def n_cols(self) -> int:
        """Grid columns: stats, accepted stamps, spacer, rejected stamps."""
        return 1 + self.stamps_per_row + 1 + self.rejected_per_row


@dataclass(frozen=True)
class _WfLayout:
    """Figure geometry for the wavefront diagnostic plot, in inches.

    ``row_h`` here is the height of one *donut* row within a corner block, of
    which a corner stacks ``max_rows``; the donut plot's ``row_h`` is the
    height of a whole *detector* section.  Same name, different unit of work.
    """

    cell: float
    row_h: float
    hpad: float


# One row per detector, so the row height is set small enough that all eight
# corner sensors fit a printable page.
_DONUT_LAYOUT = _DonutLayout(
    stamps_per_row=8,
    rejected_per_row=2,
    stamp_col_w=1.8 * 1.05,
    stats_col_w=2.8,
    row_h=1.7 * 1.05,
    legend_h=0.35,
    spacer_w=0.15,
    suptitle_h=0.55,
)

# Four corner blocks in a 2x2, each of which is a square grid of unit cells,
# so `cell` and `row_h` are equal by intent and not by coincidence.
_WF_LAYOUT = _WfLayout(cell=1.0, row_h=1.0, hpad=0.08)


@dataclass(frozen=True)
class _FpLayout:
    """Figure geometry for the focal-plane selection plot, in inches.

    A third sibling of `_DonutLayout` and `_WfLayout`.  Square because the plot
    is a 2x2 of corners whose cells are each two detectors of 2:1 aspect.

    Attributes
    ----------
    fig_side : float
        Side of the square figure.
    margin : float
        Outer border on all four sides.
    suptitle_h : float
        Extra height reserved at the top for the suptitle.
    legend_h : float
        Extra height reserved at the bottom for the legend.
    gap_intra : float
        Seam between the two detectors of one corner.  The wavefront sensor's
        two halves are physically 3.27 mm apart, so this is what decides
        whether the vignetting arc reads as continuous across the seam.
    gap_inter : float
        Gap between adjacent corners.  One value for both directions, so the
        focal plane reads as evenly spaced whichever way each corner falls.
    max_sources : int
        Per detector, per kind, the most markers drawn.  The *dataset* is
        uncapped; this only bounds the ink.  A galactic-bulge field can put
        ~10^4 reference sources on one detector, and eight panels of that is a
        slow figure made of solid circles.  Brightest-first, so the cap drops
        the faint tail rather than an arbitrary slice.
    """

    fig_side: float
    margin: float
    suptitle_h: float
    legend_h: float
    gap_intra: float
    gap_inter: float
    max_sources: int


_FP_LAYOUT = _FpLayout(
    fig_side=13.0,
    margin=0.26,
    suptitle_h=0.52,
    legend_h=0.39,
    gap_intra=0.44,
    gap_inter=0.30,
    max_sources=2000,
)

# Aspect forced on each panel: the sensors are 4072x2000, which is 2.036:1, but
# they are *drawn* 2:1 and so stretched by 1.8% along their long axis.
# Deliberate to let the four corner blocks tile a square with no residual
# slack, which is what makes the gaps below exactly settable.
_FP_PANEL_ASPECT = 2.0

# How the 2x2 of corners is built, as
# ``(corner, grid row, grid col, first detector, second detector)``.
#
# The detector *order* within a corner is tabulated rather than derived from
# intra/extra.
#
# Which *way* a corner's two detectors stack is not here: it follows from their
# displayed aspect, which follows from `n_quarter`, which is read off the data.
# See `_focal_plane_axes_rects`.
_FP_MOSAIC = (
    ("R00", 0, 0, "R00_SW1", "R00_SW0"),
    ("R40", 0, 1, "R40_SW1", "R40_SW0"),
    ("R04", 1, 0, "R04_SW0", "R04_SW1"),
    ("R44", 1, 1, "R44_SW0", "R44_SW1"),
)

# The Noll range every Zernike bar chart spans, regardless of the configured
# `nollIndices`, so plots stay comparable across configs.
_ZK_BAR_MIN, _ZK_BAR_MAX = 4, 28

# Zernike families shaded behind the bars, as
# ``(first Noll indices, color, alpha, indices spanned)``. An (n, |m|) doublet
# occupies two adjacent Noll indices and so spans 2; the m=0 spherical terms
# are single and span 1.
#
# Data rather than the run of `axvspan` calls this replaces, because that run
# wrote two of the seven bands as raw axis coordinates (19.5-21.5 is j=20,
# 26.5-28.5 is j=27) and the other five as `j - 0.5, j + 1.5` -- the same thing
# spelled two ways, which is how one gets edited out of step with the other.
_ZK_FAMILY_BANDS = (
    ((4, 11, 22), "#000000", 0.15, 1),  # spherical, m=0
    ((7, 16), _COLOR_COMA, 0.35, 2),  # coma, m=1
    ((5, 12, 23), _COLOR_ASTIGMATISM, 0.25, 2),  # astigmatism, m=2
    ((9, 18), _COLOR_TREFOIL, 0.25, 2),  # trefoil, m=3
    ((14, 25), _COLOR_QUADRAFOIL, 0.25, 2),  # quadrafoil, m=4
    ((20,), _COLOR_PENTAFOIL, 0.25, 2),  # pentafoil, m=5
    ((27,), _COLOR_HEXAFOIL, 0.25, 2),  # hexafoil, m=6
)


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    """A ``#rrggbb`` string as an RGB triple in 0-1."""
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def _diverging_cmap(name: str, stops: tuple, colors: tuple):
    """A `LinearSegmentedColormap` from normalized stops and hex colors."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(zip(stops, [_hex_to_rgb(c) for c in colors])))


# Diverging colormaps for the WF image panels, built once rather than per call.
#
# `_CMAP_DONUT` is deliberately *asymmetric*: its white point sits at 0.5 but a
# sky-blue anchor sits at 0.45, just below it, which lifts the near-zero
# negatives clear of the background so a faint donut edge stays visible.
# `_CMAP_DONUT_SYM` is the plain symmetric version, used for residuals where an
# asymmetry would read as structure that is not there.
_CMAP_DONUT = _diverging_cmap(
    "bwr_donut",
    (0.0, 0.45, 0.5, 1.0),
    (_COLOR_CMAP_NEG, _COLOR_CMAP_MID, "#FFFFFF", _COLOR_CMAP_POS),
)
_CMAP_DONUT_SYM = _diverging_cmap(
    "bwr_donut_sym",
    (0.0, 0.5, 1.0),
    (_COLOR_CMAP_NEG, "#FFFFFF", _COLOR_CMAP_POS),
)


@dataclass(frozen=True)
class _StampStyle:
    """What every donut stamp needs from the catalog's visit-level meta.

    The eight values `_drawDonutStamp` used to capture as closure variables.
    Visit-level by nature: the aperture and annulus fractions are config, the
    obscuration an instrument constant, and which stamp column exists a
    property of how the catalog was written.  Only the *radius* is per donut,
    and that rides each row.

    Module-private and built per call, like `cutDonutStamps._ExposureContext`
    and for the same reason: it is a bundle of arguments for one call tree, not
    a data product.

    Attributes
    ----------
    stamp_col : str
        Which column holds the pixels -- see `from_catalog`.
    px_scale : float
        Multiplier taking unbinned pixel quantities (aperture radii, refcat
        offsets, text offsets) into the drawn image's pixels.  1.0 for an
        unbinned ``stamp``.
    obscuration : float
        Central obscuration as a fraction of the donut radius.
    aperture_margin_frac, bkg_inner_disc_frac : float
        The photometry aperture's fractional margin, and the filled disc inside
        the central obscuration.
    bkg_annulus_inner_frac, bkg_annulus_outer_frac : float
        Inner and outer edges of the background annulus outside the donut.
        Note `bkg_inner_disc_frac` above is a *different* inner radius -- the
        two are easy to confuse, which is why they keep their meta spellings.
    """

    stamp_col: str
    px_scale: float
    obscuration: float
    aperture_margin_frac: float
    bkg_inner_disc_frac: float
    bkg_annulus_inner_frac: float
    bkg_annulus_outer_frac: float

    @classmethod
    def from_catalog(cls, catalog: QTable) -> "_StampStyle":
        """Resolve the stamp column and read the visit-level scalars.

        The unbinned ``stamp`` column is optional (see corner mode's
        ``saveStamps``).  Without it the binned ``wf_img`` is drawn instead,
        which is why `px_scale` exists: every other quantity here is in
        unbinned pixels and has to scale by 1/binning to match.

        Raises
        ------
        RuntimeError
            If the catalog carries neither column, so there is nothing to draw.
        """
        has_stamp = "stamp" in catalog.colnames
        if not has_stamp and "wf_img" not in catalog.colnames:
            raise RuntimeError(
                "Catalog has neither a 'stamp' nor a 'wf_img' column, so there is "
                "nothing to draw. Re-run with saveStamps or saveWfImages enabled "
                "if you want these plots."
            )
        meta = catalog.meta
        return cls(
            stamp_col="stamp" if has_stamp else "wf_img",
            px_scale=1.0 if has_stamp else 1.0 / meta.get("binning", 1),
            obscuration=meta["obscuration"],
            aperture_margin_frac=meta["aperture_margin_frac"],
            bkg_inner_disc_frac=meta["bkg_inner_disc_frac"],
            bkg_annulus_inner_frac=meta["bkg_annulus_inner_frac"],
            bkg_annulus_outer_frac=meta["bkg_annulus_outer_frac"],
        )

    def circle_radii(self, donut_radius: float) -> list[tuple[float, str, str]]:
        """The five aperture/annulus circles, as ``(radius, color, style)``.

        ``donut_radius`` is the row's own measured radius, already scaled into
        the drawn image's pixels.
        """
        inside = donut_radius * self.obscuration
        return [
            (inside * self.bkg_inner_disc_frac, _COLOR_BKG_ANNULUS, "--"),
            (inside * (1 - self.aperture_margin_frac), _COLOR_APERTURE, "-"),
            (donut_radius * (1 + self.aperture_margin_frac), _COLOR_APERTURE, "-"),
            (donut_radius * self.bkg_annulus_inner_frac, _COLOR_BKG_ANNULUS, "--"),
            (donut_radius * self.bkg_annulus_outer_frac, _COLOR_BKG_ANNULUS, "--"),
        ]


@dataclass(frozen=True)
class _WfFitInfo:
    """One fit's scalars, as bare floats in the units the labels state.

    Stripped of their Quantities at construction because every use is a format
    string carrying its own suffix ("t=%.1fs", "blur=%.2farcsec").  A record
    that no fit produced (an unfitted surplus donut) still carries one of
    these, filled with the not-fitted values, so the drawing code needs no
    special case: see `not_fitted`.
    """

    elapsed: float
    nfev: int
    fwhm: float

    @classmethod
    def not_fitted(cls) -> "_WfFitInfo":
        """The stand-in for a donut no fit consumed.

        ``nfev=0`` is what the bar label tests to print "x0" rather than
        "ok"/"fail", so it is load-bearing, not merely an empty default.
        """
        return cls(elapsed=float("nan"), nfev=0, fwhm=float("nan"))


@dataclass(frozen=True)
class _WfDonut:
    """One donut's images and identity within a `_WfGroup`.

    ``model_img`` is None when the fitter produced none, which the drawing
    code distinguishes from an all-NaN array: None means "draw nothing here",
    and it is also what makes the residual panel and Zernike bar drop out.
    """

    donut_id: int
    det_name: str
    # Layout only (intra left, extra right); see `_pair_up`.
    defocal: str
    img: np.ndarray
    model_img: np.ndarray | None
    blend_frac: float


@dataclass(frozen=True)
class _WfGroup:
    """One plot row's worth of fit output, rebuilt from the flat catalog.

    This is the shape `_saveWfDiagnosticPlot`'s drawing code was written
    against -- one fit, its donuts, and the Zernike deviations it produced.
    `_wf_groups_from_catalog` inverts `_buildCatalog` to recover it.

    Both a real fit and an unfitted surplus donut are represented as one of
    these; the latter has ``success=False``, a `_WfFitInfo.not_fitted`, one
    donut with no model, and no Zernikes.
    """

    det_names: list[str]
    success: bool
    fit_info: _WfFitInfo
    donuts: list[_WfDonut]
    # Noll-indexed deviations in µm, element j being Noll j, truncated at the
    # highest fitted index -- so this may be shorter than the drawn range, or
    # empty (`_NO_ZK`) for a record with no fit.
    zk_dev: np.ndarray

    def exploded(self) -> list["_WfGroup"]:
        """One single-donut copy of this group per donut it holds.

        For modes whose groups do not pair intra with extra: the group's
        scalars are shared by every donut in it, so each copy keeps them.
        """
        return [replace(self, donuts=[donut]) for donut in self.donuts]


@dataclass(frozen=True)
class _RowHalf:
    """One side of focus on one WF plot row: four panels' worth of inputs.

    A row is an intra half and an extra half, each drawing image, model,
    residual and Zernike bar.  This is what the drawing code used to unpack
    into two parallel sets of ``intra_*``/``extra_*`` locals before passing
    nine positional arguments per side; `from_group` builds one instead.

    A blank half -- padding, or a side whose group has no donut of that defocal
    type -- is a `_RowHalf` with ``img=None``, not a `None`.  That is what lets
    the drawing code ask one question (``half.img is None``) instead of
    threading `None` checks through every field.

    Attributes
    ----------
    img, model : np.ndarray or None
        The binned donut and the fitted model.  ``img`` None means a blank
        half; ``model`` None means there was no fit, which drops the residual
        panel and the Zernike bar.
    det_hdr : str
        Column header, non-empty only on a corner's first row.
    donut_id : int or None
        Annotated on the image panel.
    fwhm, blend_frac : float
        Annotated on the model and residual panels; NaN to omit.
    zk_dev : np.ndarray
        Noll-indexed deviations for the bar chart, `_NO_ZK` if unfitted.
    bar_label : str
        The fit's timing/status inset on the bar.
    """

    img: np.ndarray | None
    model: np.ndarray | None
    det_hdr: str
    donut_id: int | None
    fwhm: float
    blend_frac: float
    zk_dev: np.ndarray
    bar_label: str

    @classmethod
    def from_group(cls, group: "_WfGroup | None", defocal: str, det_hdr: str) -> "_RowHalf":
        """Pick the ``defocal`` donut out of ``group``, flat for drawing.

        ``group`` is None for a padded layout row.  A paired group holds both
        defocal types and each half picks out its own; an exploded or unfitted
        group holds a single donut, matching only the side it belongs to -- so
        a miss is normal and yields a blank half, not an error.
        """
        fit_info = group.fit_info if group is not None else _WfFitInfo.not_fitted()
        success = group.success if group is not None else False
        # nfev == 0 means the fit never iterated, which reads as "x0" rather
        # than as a failure: see `_WfFitInfo.not_fitted`.
        status = "x0" if fit_info.nfev == 0 else ("ok" if success else "fail")
        donut = next((d for d in group.donuts if d.defocal == defocal), None) if group is not None else None
        return cls(
            img=donut.img if donut else None,
            model=donut.model_img if donut else None,
            det_hdr=det_hdr,
            donut_id=donut.donut_id if donut else None,
            fwhm=fit_info.fwhm,
            blend_frac=donut.blend_frac if donut else float("nan"),
            zk_dev=group.zk_dev if group is not None else _NO_ZK,
            bar_label=f"t={fit_info.elapsed:.1f}s {status} nfev={fit_info.nfev}",
        )


def _meta_value(meta: dict, key: str, unit: u.UnitBase) -> float:
    """Return one ``meta`` scalar as a bare float in ``unit``.

    The catalog's meta values are Quantities (see `_build_donut_catalog`), but
    every use here is a format string that already carries its own unit suffix
    ("run=%.1fs"), so they are stripped at the read the same way the column
    Quantities are.  A missing key degrades to NaN, which the callers already
    render as "N/A" rather than raising mid-plot.
    """
    value = meta.get(key)
    return np.nan if value is None else value.to_value(unit)


def _det_id_by_name(catalog: QTable) -> dict[str, int]:
    """Map detector name to detector id, read off the catalog rows.

    Parameters
    ----------
    catalog : QTable
        Per-donut table carrying ``det_name`` and ``det_id`` columns.

    Returns
    -------
    dict [`str`, `int`]
        Detector name -> detector id, for the detectors present in ``catalog``.
    """
    names = np.asarray(catalog["det_name"], dtype=str)
    ids = np.asarray(catalog["det_id"], dtype=int)
    return {str(n): int(i) for n, i in zip(names, ids)}


def _donut_rows_by_detector(catalog: QTable) -> list[tuple[str, QTable, QTable]]:
    """Split the catalog per detector into accepted and rejected rows.

    Splits on ``candidate`` rather than on whether a fit consumed the donut:
    this plot is about donut *selection*, so a candidate no fit used still
    belongs in the accepted panel, having passed every cut the plot reports.

    Returns
    -------
    list of tuple
        ``(det_name, accepted, rejected)`` per detector with at least one row,
        sorted by detector name.  Detectors absent from the catalog are absent
        here -- the caller sizes the figure from this list.
    """
    det_name_col = np.asarray(catalog["det_name"], dtype=str)
    dets_with_data = []
    for det_name in sorted(set(det_name_col.tolist())):
        det_rows = catalog[det_name_col == det_name]
        accepted = det_rows[det_rows["candidate"]]
        rejected = det_rows[~det_rows["candidate"]]
        if len(accepted) > 0 or len(rejected) > 0:
            dets_with_data.append((det_name, accepted, rejected))
    return dets_with_data


def _detector_stats_lines(det_name: str, det_id: int, n_donuts: int, det_stats: dict) -> list[str]:
    """The monospace stats block for one detector's panel, one line per entry.

    Returns strings rather than drawing, so the formatting is testable without
    a figure.

    Parameters
    ----------
    det_name, det_id : str, int
        Detector identity, for the header line.
    n_donuts : int
        Accepted donut count.
    det_stats : dict
        One ``catalog.meta["det_meta"]`` entry, or ``{}`` for a detector with
        no entry -- every read below tolerates the absence, reporting NaN or
        omitting the line.
    """
    scatter_val = _meta_value(det_stats, "astrom_scatter", u.arcsec)
    scatter_str = f'{scatter_val:.3f}"' if np.isfinite(scatter_val) else "N/A"
    lines = [f"{det_name} ({det_id})", f"donuts: {n_donuts}"]
    # One line per cutout stage, driven off the shared key list so this panel
    # cannot fall behind the log lines reporting the same stages. The label
    # column is padded to the longest label rather than to a hard-coded width,
    # since this is monospace text.
    width = max(len(label) for label in _CUTOUT_STAGE_KEYS) + 2
    for label, key in _CUTOUT_STAGE_KEYS.items():
        line = f"{label + ':':<{width}}{_meta_value(det_stats, key, u.s):.3f}s"
        # Scatter belongs to the WCS refit, so it hangs off that stage.
        lines.append(f"{line}  ({scatter_str})" if label == "astrom" else line)
    if det_stats.get("wcs_refit_error"):
        lines.append(f"WCS ERR: {det_stats['wcs_refit_error'][:_PANEL_ERROR_CHARS]}")
    if det_stats.get("cat_select_error"):
        lines.append(f"CAT ERR: {det_stats['cat_select_error'][:_PANEL_ERROR_CHARS]}")
    return lines


def _donut_annotation(row, rejected: bool) -> str:
    """The three-line stats caption drawn above one donut stamp.

    A ``?`` in place of a number means the value is non-finite, which is how a
    measurement the cutout never made reads -- distinct from a real measurement
    that happens to be zero.  Returns the string rather than drawing it, so it
    is testable without a figure.

    ``rejected`` is unused in the text itself: the rejection *flags* come off
    the row's own ``rejected_*`` columns, which an accepted row simply has none
    of.  It stays in the signature because the caller's colour choice pairs
    with this caption, and a future change here is likely to want it.
    """
    values = [
        ("snr", row["snr"], 0),
        ("if", row["inner_frac"], 3),
        ("of", row["outer_frac"], 3),
        ("osm", row["outer_sector_minmax_frac"], 3),
    ]
    parts = {
        name: (f"{name}={value:.{places}f}" if np.isfinite(value) else f"{name}=?")
        for name, value, places in values
    }
    flags = [
        name
        for name, flagged in (
            ("sat", row["rejected_sat"]),
            ("inner", row["rejected_inner_frac"]),
            ("outer", row["rejected_outer_frac"]),
            ("snr", row["rejected_snr"]),
        )
        if flagged
    ]
    rej_str = f"[{'|'.join(flags)}]" if flags else ""
    donut_id = row["donut_id"]
    donut_id_str = f"id={donut_id}" if donut_id != 0 else ""
    return f"{parts['snr']}  {rej_str}\n{parts['if']}  {parts['of']}  {parts['osm']}\n{donut_id_str}"


def _rot90_display(row_idx, col_idx, height: int, width: int, n_quarter: int):
    """Map array indices through to display coords.

    The whole-detector counterpart of `_stamp_transform`, which applies the
    same rotation to *offsets* within a stamp.  The two are deliberately
    separate: that one rotates about an origin and so needs no extent, and it
    carries a centroid rounding residual that has no analogue here.  Changing
    the rotation convention means changing both.

    Works elementwise, so it takes whole arrays of coordinates, and in floats,
    so a sub-pixel source position stays sub-pixel.

    Parameters
    ----------
    row_idx, col_idx : array_like
        Indices into the *un-rotated* array -- i.e. ``(y, x)`` in the detector
        frame, already scaled into the binned image (see `_binned_coord`).
    height, width : int
        The un-rotated array's shape, needed because each quarter turn reflects
        one axis about the array's extent.
    n_quarter : int
        Quarter turns, as `lsst.afw.cameraGeom.Orientation.getNQuarter` reports
        them; taken mod 4.

    Returns
    -------
    tuple
        ``(x_display, y_display)``.  The pair is the rotated ``(row, col)``
        *in that order*, which is what applies the transpose -- returning
        ``(col, row)`` would silently mirror every overlay about the diagonal.
    """
    r = np.asarray(row_idx, dtype=float)
    c = np.asarray(col_idx, dtype=float)
    h, w = height, width
    for _ in range(n_quarter % 4):
        # One clockwise quarter turn: the new row is the old column, and the
        # new column counts back from the old array's height.
        r, c, h, w = c, (h - 1) - r, w, h
    return r, c


def _binned_coord(x_det, y_det, row):
    """Scale detector-frame coordinates into one row's binned image indices.

    Returns ``(row_idx, col_idx)`` -- numpy order, ready for `_rot90_display`.

    The ``(b - 1) / 2`` term is not cosmetic.  `lsst.afw.math.binImage`
    averages the block ``[b*j, b*j + b - 1]``, whose center sits at
    ``b*j + (b - 1)/2``, so binned pixel ``j`` is centered on that un-binned
    coordinate rather than on ``b*j``.  Dividing by ``b`` alone therefore
    offsets every marker by ``(b - 1) / (2b)`` binned pixels -- at ``b=4`` that
    is 0.375, which put a marker in the *wrong binned pixel* for half of a
    64-row test sweep.

    Binning and origin are read off the row rather than assumed, so a plot
    regenerated from a run with different settings still lands correctly.
    """
    b = int(row["binning"])
    offset = (b - 1) / 2
    x0 = float(row["bbox_min_x"])
    y0 = float(row["bbox_min_y"])
    return (
        (np.asarray(y_det, dtype=float) - y0 - offset) / b,
        (np.asarray(x_det, dtype=float) - x0 - offset) / b,
    )


def _focal_plane_axes_rects(
    n_quarter_by_det: dict[str, int], layout: _FpLayout
) -> dict[str, tuple[float, float, float, float]]:
    """Explicit ``add_axes`` rectangles for the eight corner panels.

    Each corner owns one quadrant holding its two detectors.  *How* they split
    that quadrant depends on how they display: a detector whose ``n_quarter``
    is even comes out tall (2000x4072 displayed for a corner sensor), so the
    pair sits side by side; an odd one comes out wide and the pair stacks.
    Reading that off the data rather than hardcoding it per corner means the
    layout cannot disagree with the images it is laying out.

    Every panel is ``short`` x ``long`` with ``long = 2 * short``, so a pair
    plus its seam spans ``2 * short + gap_intra`` across and ``long`` along.
    Because ``long`` and ``2 * short`` differ only by ``gap_intra``, each row
    and each column of the 2x2 sums to the same total -- the blocks tile the
    content box with nothing left over, which is exactly why both inter-corner
    gaps come out equal and the intra-corner seam comes out at its requested
    width.  Forcing 2:1 (see `_FP_PANEL_ASPECT`) is what buys that.

    Detectors absent from ``n_quarter_by_det`` still get their rectangles --
    the grid is always the full focal plane, and the caller turns the empty
    axes off, so a missing corner reads as missing rather than silently
    reflowing its neighbors.

    Parameters
    ----------
    n_quarter_by_det : dict [`str`, `int`]
        Detector name to quarter turns, for the detectors that have an image.
    layout : `_FpLayout`
        Figure geometry.

    Returns
    -------
    dict [`str`, `tuple`]
        Detector name to ``(left, bottom, width, height)`` in figure fractions,
        as `matplotlib.figure.Figure.add_axes` takes.
    """
    side = layout.fig_side
    # The content box, in inches, once the borders and the title/legend bands
    # are taken out. Made square by giving the surplus on the wider axis back
    # as extra centred border: a non-square content box would otherwise dump
    # its excess into the seams, which is the larger half of the bug this
    # replaces.
    avail_w = side - 2 * layout.margin
    avail_h = side - 2 * layout.margin - layout.suptitle_h - layout.legend_h
    content = min(avail_w, avail_h)
    x0 = (side - content) / 2
    # Not centred vertically in the figure but within what the bands leave, so
    # the plot does not drift up into the suptitle.
    y0 = layout.margin + layout.legend_h + (avail_h - content) / 2

    # content = (2 * short + gap_intra) + gap_inter + long, with
    # long = 2 * short.
    short = (content - layout.gap_intra - layout.gap_inter) / 4
    long = _FP_PANEL_ASPECT * short
    # The across-extent of a pair box. Differs from `long` by gap_intra, which
    # is why the quadrant origins below cannot use a single stride.
    pair = 2 * short + layout.gap_intra

    rects: dict[str, tuple[float, float, float, float]] = {}
    for _, quad_row, quad_col, first, second in _FP_MOSAIC:
        # Either detector's orientation answers for the pair: the two sensors
        # of a corner are mounted together and so display the same way round.
        # Default 0 (tall, side by side) when neither has an image, which only
        # affects how a pair of blank panels is carved up.
        n_quarter = n_quarter_by_det.get(first, n_quarter_by_det.get(second, 0))
        tall = n_quarter % 2 == 0
        block_w, block_h = (pair, long) if tall else (long, pair)

        # Left edge of this quadrant: the left column starts at x0, the right
        # column starts past the left block's own width plus the inter gap. The
        # left block's width depends on how *it* falls, so measure it rather
        # than assuming a uniform stride.
        if quad_col == 0:
            bx = x0
        else:
            left_first, left_second = next(
                (f, s) for _, qr, qc, f, s in _FP_MOSAIC if qr == quad_row and qc == 0
            )
            left_nq = n_quarter_by_det.get(left_first, n_quarter_by_det.get(left_second, 0))
            bx = x0 + (pair if left_nq % 2 == 0 else long) + layout.gap_inter
        # Grid row 0 is the *top*, so its bottom edge sits a block-height below
        # the content top; row 1 sits at the content bottom.
        if quad_row == 0:
            by = y0 + content - block_h
        else:
            by = y0

        if tall:
            # Side by side, `first` on the left.
            rects[first] = (bx, by, short, long)
            rects[second] = (bx + short + layout.gap_intra, by, short, long)
        else:
            # Stacked, `first` on top.
            rects[first] = (bx, by + short + layout.gap_intra, long, short)
            rects[second] = (bx, by, long, short)

    # Into figure fractions, which is what `add_axes` wants.
    return {
        name: (rx / side, ry / side, rw / side, rh / side) for name, (rx, ry, rw, rh) in rects.items()
    }


def _pair_links(catalog: QTable) -> list[tuple[tuple[str, float, float], tuple[str, float, float]]]:
    """Intra/extra donut pairs to join.

    Only meaningful in ``paired`` mode, where `_build_wf_groups` makes one
    group per intra/extra pair and so ``group_id`` names exactly two donuts on
    the two detectors of one corner.  Every other mode either groups a single
    donut (``unpaired``) or a whole detector or corner at once
    (``full_detector``, ``full_corner``), where one ``group_id`` covers many
    donuts and "the pair" is not a thing the id identifies -- so those return
    nothing rather than a combinatorial tangle.

    Not filtered on fit success: pairing happens before fitting, so a pair
    whose fit failed was still a pair, and seeing it is the point of the plot.

    Returns
    -------
    list of tuple
        ``((det_name, x_det, y_det), (det_name, x_det, y_det))`` per pair.
    """
    if str(catalog.meta.get("wf_mode", "")) != "paired":
        return []
    by_group: dict[str, list] = {}
    for row in catalog:
        group_id = str(row["group_id"])
        if not group_id:
            continue
        by_group.setdefault(group_id, []).append(row)
    links = []
    for _, rows in sorted(by_group.items()):
        # A paired group is exactly two donuts on two distinct detectors.
        # Anything else is not a pair, whatever mode claims.
        if len(rows) != 2:
            continue
        first, second = rows
        if str(first["det_name"]) == str(second["det_name"]):
            continue
        links.append(
            (
                (str(first["det_name"]), first["x_det"].to_value(u.pix), first["y_det"].to_value(u.pix)),
                (str(second["det_name"]), second["x_det"].to_value(u.pix), second["y_det"].to_value(u.pix)),
            )
        )
    return links


def _overlay_points(overlays: QTable | None, det_name: str) -> dict[str, tuple]:
    """One detector's overlay sources, grouped by ``kind``.

    A ``kind`` with no rows is *absent* from the result rather than present and
    empty, so the caller draws nothing for it without a length check.  Absence
    here does not distinguish a stage that never ran from one that ran and
    found nothing -- read ``has_refcat`` on the image table for that.

    Returns
    -------
    dict [`str`, tuple]
        ``kind`` to ``(x_det, y_det, mag)`` arrays, brightest first so a caller
        capping the count keeps the brightest.
    """
    if overlays is None or len(overlays) == 0:
        return {}
    det_col = np.asarray(overlays["det_name"], dtype=str)
    kind_col = np.asarray(overlays["kind"], dtype=str)
    out = {}
    for kind in np.unique(kind_col):
        mask = (det_col == det_name) & (kind_col == kind)
        if not mask.any():
            continue
        rows = overlays[mask]
        mag = rows["mag"].to_value(u.mag)
        # NaN magnitudes (blitz detections have no photometry) sort last, so a
        # capped draw keeps real measurements over unknowns.
        order = np.argsort(np.where(np.isnan(mag), np.inf, mag))
        out[str(kind)] = (
            rows["x_det"].to_value(u.pix)[order],
            rows["y_det"].to_value(u.pix)[order],
            mag[order],
        )
    return out


def _detector_image_rows(detector_images: QTable | None) -> dict[str, Any]:
    """Index the per-detector image table by detector name.

    The inversion of `_build_detector_image_table`, in the spirit of
    `_wf_groups_from_catalog`: the table is the transport, this is the shape
    the drawing code wants.
    """
    if detector_images is None or len(detector_images) == 0:
        return {}
    return {str(row["det_name"]): row for row in detector_images}


def _stamp_transform(row, style: _StampStyle, det_meta: dict):
    """Build the detector-frame to stamp-display-coordinate mapping for a row.

    Returns a function of ``(dx, dy)`` in unbinned detector pixels, giving
    ``(x, y)`` in the drawn stamp's coordinates.

    It must mirror the stamp transform in `_cut_and_evaluate_stamps`,
    ``np.rot90(stamp, k=-n_quarter).T`` -- **including the transpose**.  The
    loop applies the rot90 in (row, col) space; returning ``(r, c)`` rather
    than ``(c, r)`` is what applies the ``.T``.  Under ``origin="lower"`` the
    displayed x axis is the column index and y the row index, so the returned
    pair is ``(x, y)`` after transposition.

    `_rot90_display` applies the same rotation to whole-detector coordinates
    for the focal-plane plot.  Kept separate because these are *offsets* about
    a centroid, so no array extent enters, and because of the rounding residual
    below, which has no counterpart there -- but the rotation convention is
    shared, so a change to one belongs in both.

    The stamp was cut on integer bounds around the *rounded* centroid, so
    display coordinate (0, 0) is that rounded position while the ``nearby_*``
    offsets are measured from ``x_det``/``y_det``.  The rounding residual
    converts between the two, and is applied before the rotation because it is
    a correction in the detector frame.  Sub-pixel, but it is the difference
    between a marker on the source and one up to half a pixel off it.
    """
    # Orientation is per detector, and keyed by the row's own visit rather
    # than the table's: full-array mode has one entry per detector per side of
    # focus.
    n_quarter = det_meta.get(f"{row['det_name']}_{row['visit_id']}", {}).get("n_quarter", 0) % 4
    x_det = row["x_det"].to_value(u.pix)
    y_det = row["y_det"].to_value(u.pix)
    res_x = x_det - round(x_det)
    res_y = y_det - round(y_det)

    def to_display(dx, dy):
        r, c = dy + res_y, dx + res_x
        for _ in range(n_quarter):
            r, c = c, -r
        # Offsets are in unbinned pixels; scale to the drawn image.
        return r * style.px_scale, c * style.px_scale

    return to_display


def _wf_groups_from_catalog(catalog: QTable) -> tuple[list[_WfGroup], list[_WfGroup]]:
    """Invert `_buildCatalog`, recovering the WF plot's per-fit records.

    The catalog is one flat row per donut; the WF plot draws one row per fit,
    so this regroups on ``group_id`` and re-nests.  Pure function of the
    catalog: it touches no figure state, which is what makes it testable
    without rendering anything.

    Parameters
    ----------
    catalog : QTable
        Per-donut table from ``_buildCatalog``.

    Returns
    -------
    plottable : list [`_WfGroup`]
        Fits that produced a model, ordered by ``group_id``.
    unfitted : list [`_WfGroup`]
        Candidate donuts no fit consumed, each alone in its own group, in
        catalog order.  Kept separate because the layout places them below
        the fitted rows rather than interleaved.
    """
    # Group rows a fit claimed; an empty `group_id` means none did.
    groups: dict[str, list] = {}
    for row in catalog:
        group_id = str(row["group_id"])
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(row)

    plottable = []
    for _, rows in sorted(groups.items()):
        first = rows[0]
        # A group whose every model is NaN was never really fit, so it has
        # nothing to draw in the model or residual panels.
        if all(np.all(np.isnan(np.array(r["model_img"]))) for r in rows):
            continue
        # group_* columns are replicated across the group, so the first row
        # carries the whole fit's values.
        fit_info = _WfFitInfo(
            elapsed=first["group_fit_elapsed"].to_value(u.s),
            nfev=first["group_fit_nfev"],
            fwhm=first["group_fwhm"].to_value(u.arcsec),
        )
        donuts = []
        for r in rows:
            model_arr = np.array(r["model_img"])
            donuts.append(
                _WfDonut(
                    donut_id=r["donut_id"],
                    det_name=r["det_name"],
                    defocal=CORNER_DEFOCAL_BY_DET_NAME.get(str(r["det_name"]), ""),
                    img=np.array(r["wf_img"]),
                    model_img=model_arr if not np.all(np.isnan(model_arr)) else None,
                    blend_frac=r["blend_frac"],
                )
            )
        plottable.append(
            _WfGroup(
                det_names=list(dict.fromkeys(r["det_name"] for r in rows)),
                success=bool(first["group_fit_success"]),
                fit_info=fit_info,
                donuts=donuts,
                zk_dev=np.asarray(first["zk_deviation_ccs"].to_value(u.micron), dtype=float),
            )
        )

    # Candidate donuts that no fit consumed (paired-mode surplus: no partner
    # on the other detector, so ``group_id`` is empty). They have no model or
    # Zernikes, but their binned stamp is still worth seeing, so carry them as
    # data-only single-donut records.
    unfitted = []
    for row in catalog:
        if str(row["group_id"]) or not row["candidate"]:
            continue
        img = np.array(row["wf_img"])
        if np.all(np.isnan(img)):
            continue
        unfitted.append(
            _WfGroup(
                det_names=[row["det_name"]],
                success=False,
                fit_info=_WfFitInfo.not_fitted(),
                donuts=[
                    _WfDonut(
                        donut_id=row["donut_id"],
                        det_name=row["det_name"],
                        defocal=CORNER_DEFOCAL_BY_DET_NAME.get(str(row["det_name"]), ""),
                        img=img,
                        model_img=None,
                        blend_frac=row["blend_frac"],
                    )
                ],
                zk_dev=_NO_ZK,
            )
        )

    return plottable, unfitted


def _corner_of(group: _WfGroup) -> str:
    """Which corner raft a group belongs to, by its first recognized detector.

    Falls back to the first corner rather than raising: the 2x2 grid has to be
    drawn regardless, and a group whose detectors are all unrecognized is a
    catalog problem this plot should not die on.
    """
    for name in group.det_names:
        if str(name) in CORNER_BY_DET_NAME:
            return CORNER_BY_DET_NAME[str(name)]
    # `CORNER_PAIRS` is keyed by corner name, so take its first *key*.
    return next(iter(CORNER_PAIRS))


def _pair_up(groups: list[_WfGroup]) -> list[tuple]:
    """Lay single-donut groups out as ``(intra, extra)`` plot rows.

    Only for modes whose groups carry no intra/extra pairing of their own: the
    pairing here is cosmetic, so rows are matched by position and the shorter
    side padded with ``None`` to keep every donut visible.
    """
    intras = [g for g in groups if g.donuts[0].defocal == "intra"]
    extras = [g for g in groups if g.donuts[0].defocal == "extra"]
    return [
        (intras[i] if i < len(intras) else None, extras[i] if i < len(extras) else None)
        for i in range(max(len(intras), len(extras)))
    ]


def _wf_row_pairs(
    plottable: list[_WfGroup],
    unfitted: list[_WfGroup],
    wf_mode: str,
    max_donuts: int,
) -> dict[str, list[tuple]]:
    """Assign every group to a corner and a plot row, padded to a fixed height.

    Pure layout, and the one part of the WF plot where a real bug could hide,
    which is why it is a function rather than three blocks inside the drawing
    loop.

    Every corner comes back with the *same* number of rows: ``max_donuts``, or
    more if some corner exceeded it.  Padding to a config-derived height rather
    than to the tallest corner is what makes two plots of one exposure
    blinkable across fitting modes -- figure dimensions and axes positions then
    depend only on config, not on how many donuts a mode happened to fit.  The
    ``or more`` half matters too: a corner with more fits than ``maxDonuts``
    grows the layout instead of losing rows.

    Parameters
    ----------
    plottable, unfitted : list of `_WfGroup`
        Fitted groups, and surplus donuts no fit claimed.
    wf_mode : str
        ``"paired"`` groups already hold both sides of focus, so each supplies
        a whole row; any other mode's groups are exploded to one donut each and
        paired for layout only.
    max_donuts : int
        ``meta["max_donuts"]``, the per-corner row count to pad to.

    Returns
    -------
    dict [str, list of tuple]
        Corner name to a list of ``(intra, extra)`` halves, either of which may
        be ``None`` for a blank.  One key per corner in `CORNER_PAIRS`, always.
    """
    by_corner: dict[str, list[_WfGroup]] = {c: [] for c in CORNER_PAIRS}
    for group in plottable:
        by_corner[_corner_of(group)].append(group)

    unfitted_by_corner: dict[str, list[_WfGroup]] = {c: [] for c in CORNER_PAIRS}
    for group in unfitted:
        unfitted_by_corner[_corner_of(group)].append(group)

    row_pairs: dict[str, list[tuple]] = {}
    for corner, groups in by_corner.items():
        if wf_mode == "paired":
            # The group *is* an intra/extra pair, so it supplies both halves of
            # the row; the donut of each defocal type is picked out when drawn.
            fit_rows = [(g, g) for g in groups]
        else:
            # These groups don't pair donuts, so flatten to one record per
            # donut and pair for layout only. Exploding is a no-op for
            # "unpaired" (one donut per group already).
            fit_rows = _pair_up([s for g in groups for s in g.exploded()])
        # Surplus donuts have no partner by construction, so they lay out
        # positionally below the fitted rows, one side of each row blank.
        row_pairs[corner] = fit_rows + _pair_up(unfitted_by_corner[corner])

    max_rows = max([max_donuts, 1] + [len(v) for v in row_pairs.values()])
    return {c: pairs + [(None, None)] * (max_rows - len(pairs)) for c, pairs in row_pairs.items()}


class DonutBlitzPlotConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    """Pipeline connections for DonutBlitzPlotTask."""

    cornerResults = connectionTypes.Input(
        doc=(
            "Per-donut catalog from DonutBlitzCornerTask containing all data "
            "needed to regenerate diagnostic plots."
        ),
        name="donutBlitzCornerResults",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        deferLoad=True,
    )
    detectorImages = connectionTypes.Input(
        doc=(
            "Per-detector binned post-ISR images for the focal-plane selection "
            "plot. Optional: written only when DonutBlitzCornerTask ran with "
            "doSelectionOutput, and that plot is simply skipped without it."
        ),
        name="donutBlitzCornerDetectorImages",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        deferLoad=True,
        minimum=0,
    )
    selectionOverlays = connectionTypes.Input(
        doc=(
            "Long-form per-source selection overlays matching detectorImages. "
            "Optional on the same terms; without it the focal-plane plot draws "
            "images and donuts but no selection-funnel markers."
        ),
        name="donutBlitzCornerSelectionOverlays",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        deferLoad=True,
        minimum=0,
    )


class DonutBlitzPlotConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DonutBlitzPlotConnections,  # type: ignore
):
    """Configuration for DonutBlitzPlotTask."""

    colorLog: pexConfig.Field[bool] = pexConfig.Field[bool](
        doc=(
            "Colorize select log messages with ANSI escape codes. If None "
            "(the default), color is enabled only when stdout is an "
            "interactive terminal."
        ),
        default=None,
        optional=True,
    )


class DonutBlitzPlotTask(pipeBase.PipelineTask):
    """PipelineTask regenerating diagnostic plots from a blitz catalog.

    Reads ``donutBlitzCornerResults``.  Can run standalone (reading from the
    butler) or be called as a subtask of ``DonutBlitzCornerTask`` when
    ``savePlots=True``.

    Runs written before that dataset type was renamed hold their catalog as
    ``donutBlitzResults``; point this task at one with
    ``-c donutBlitzPlot:connections.cornerResults=donutBlitzResults``.
    """

    ConfigClass = DonutBlitzPlotConfig
    _DefaultName = "donutBlitzPlot"
    config: DonutBlitzPlotConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._colorLogEnabled = _resolve_color_log_enabled(self.config.colorLog)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        # Meta carries the visit-level scalars every plot labels itself with,
        # so it must survive the read.
        load = dict(parameters={"strip_astropy_meta_yaml": False})
        catalog = inputs["cornerResults"].get(**load)

        # `minimum=0` means these may not have been produced at all, in which
        # case they are absent from `inputs` rather than None -- so ask the
        # dict, not the attribute.
        def _optional(name):
            handle = inputs.get(name)
            return None if handle is None else handle.get(**load)

        self.run(
            catalog,
            detector_images=_optional("detectorImages"),
            selection_overlays=_optional("selectionOverlays"),
        )

    def run(
        self,
        catalog: Table,
        detector_images: Table | None = None,
        selection_overlays: Table | None = None,
    ) -> None:
        """Generate the diagnostic plots for one blitz visit.

        The two optional tables are what the focal-plane selection plot needs.
        Both paths into this method supply them the same way: as a subtask,
        `DonutBlitzCornerTask` passes the in-memory tables it just built,
        regardless of whether it also persisted them; standalone, `runQuantum`
        passes whatever the butler had.  Absent either way, that one plot is
        skipped and the other two are unaffected.

        Parameters
        ----------
        catalog : QTable
            Per-donut table as produced by
            ``DonutBlitzCornerTask._buildCatalog``.  Visit-level and
            per-detector metadata are in ``catalog.meta``.
        detector_images : QTable, optional
            Per-detector binned images, from ``_build_detector_image_table``.
        selection_overlays : QTable, optional
            Long-form selection sources, from ``_build_overlay_table``.
        """
        catalog = QTable(catalog)
        self._saveDonutDiagnosticPlot(catalog)
        self._saveWfDiagnosticPlot(catalog)
        self._saveFocalPlanePlot(
            catalog,
            None if detector_images is None else QTable(detector_images),
            None if selection_overlays is None else QTable(selection_overlays),
        )

    def _saveDonutDiagnosticPlot(self, catalog: QTable) -> None:
        """Save a single diagnostic PNG with one section per detector.

        Layout per detector:
          - Left column: stats text (timing, WCS scatter, donut count, errors)
          - Remaining columns: donut stamps (up to maxDonuts), each annotated
            with flux and field angle

        Parameters
        ----------
        catalog : QTable
            Per-donut table from ``_buildCatalog``.  Per-detector metadata is
            in ``catalog.meta["det_meta"]``, keyed by
            ``f"{det_name}_{visit_id}"``; visit-level scalars are in
            ``catalog.meta``.
        """
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec

        if len(catalog) == 0:
            return

        meta = catalog.meta
        run_elapsed = _meta_value(meta, "run_elapsed", u.s)
        refcat_elapsed = _meta_value(meta, "refcat_elapsed", u.s)
        butler_elapsed = _meta_value(meta, "butler_elapsed", u.s)
        photo_filter_label = meta["photo_filter_name"]
        astrom_filter_label = meta["astrom_filter_name"]
        visit_id = meta["ref_visit_id"]
        det_meta = meta["det_meta"]

        det_id_of = _det_id_by_name(catalog)
        dets_with_data = _donut_rows_by_detector(catalog)

        n_dets = len(dets_with_data)
        if n_dets == 0:
            return

        layout = _DONUT_LAYOUT
        fig_w = (
            layout.stats_col_w
            + (layout.stamps_per_row + layout.rejected_per_row) * layout.stamp_col_w
            + layout.spacer_w
        )
        fig_h = n_dets * layout.row_h + layout.legend_h + layout.suptitle_h

        fig = Figure(figsize=(fig_w, fig_h), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.0, wspace=0.0)
        butler_str = f"  butler={butler_elapsed:.1f}s" if butler_elapsed > 0 else ""
        fig.suptitle(
            f"DonutBlitz diagnostics  visit={visit_id}"
            f"  refcat={refcat_elapsed:.1f}s{butler_str}  run={run_elapsed:.1f}s",
            fontsize=9,
        )

        w_stats = layout.stats_col_w / layout.stamp_col_w
        w_spacer = layout.spacer_w / layout.stamp_col_w
        gs = GridSpec(
            n_dets + 1,
            layout.n_cols,
            figure=fig,
            height_ratios=[layout.row_h] * n_dets + [layout.legend_h],
            width_ratios=(
                [w_stats] + [1] * layout.stamps_per_row + [w_spacer] + [1] * layout.rejected_per_row
            ),
        )
        COL_ACCEPTED_START = 1
        COL_SPACER = 1 + layout.stamps_per_row
        COL_REJECTED_START = COL_SPACER + 1

        style = _StampStyle.from_catalog(catalog)

        for row_idx, (det_name, acc_rows, rej_rows) in enumerate(dets_with_data):
            ax_stats = fig.add_subplot(gs[row_idx, 0])
            ax_stats.axis("off")
            ax_stats.text(
                0.05,
                0.95,
                "\n".join(
                    _detector_stats_lines(
                        det_name=det_name,
                        det_id=det_id_of[det_name],
                        n_donuts=len(acc_rows),
                        # Keyed by detector *and* visit; corner mode has just
                        # this one visit. A miss degrades to the not-reached
                        # defaults rather than raising.
                        det_stats=det_meta.get(f"{det_name}_{visit_id}", {}),
                    )
                ),
                transform=ax_stats.transAxes,
                fontsize=6,
                va="top",
                family="monospace",
            )

            for col_idx in range(layout.stamps_per_row):
                ax = fig.add_subplot(gs[row_idx, COL_ACCEPTED_START + col_idx])
                ax.axis("off")
                if col_idx >= len(acc_rows):
                    continue
                self._drawDonutStamp(ax, acc_rows[col_idx], style, det_meta)

            ax_sp = fig.add_subplot(gs[row_idx, COL_SPACER])
            ax_sp.axis("off")

            for col_idx in range(layout.rejected_per_row):
                ax = fig.add_subplot(gs[row_idx, COL_REJECTED_START + col_idx])
                ax.axis("off")
                if col_idx >= len(rej_rows):
                    continue
                self._drawDonutStamp(ax, rej_rows[col_idx], style, det_meta, rejected=True)

        ax_legend = fig.add_subplot(gs[n_dets, :])
        ax_legend.axis("off")
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor=_COLOR_PHOTO_REFCAT,
                markersize=6,
                label=f"photo refcat ({photo_filter_label})",
            ),
            Line2D(
                [0],
                [0],
                marker="+",
                color=_COLOR_ASTROM_REFCAT,
                markersize=6,
                linestyle="none",
                label=f"astrom refcat ({astrom_filter_label})",
            ),
        ]
        ax_legend.legend(
            handles=legend_handles,
            loc="center",
            ncol=2,
            fontsize=7,
            frameon=False,
            handletextpad=0.5,
            columnspacing=2.0,
        )

        fname = f"donut_diag_{visit_id}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        self.log.info("Saved diagnostic plot: %s", fname)

    def _drawDonutStamp(self, ax, row, style: _StampStyle, det_meta: dict, rejected=False) -> None:
        """One catalog row's cutout, with aperture and refcat overlays.

        Reads the stamp out of ``row`` (whichever of the two stamp columns
        `_StampStyle.from_catalog` resolved to) and annotates it in unbinned
        pixel units scaled by ``style.px_scale``.  Distinct from the wavefront
        plot's `_drawWfImage`, which draws an already-extracted array.

        Was a closure over eight values before it became a method; those are
        now `_StampStyle`, on the same reasoning that made
        `cutDonutStamps._ExposureContext` a record.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw into, already turned off by the caller.
        row : astropy.table.Row
            One catalog row.
        style : `_StampStyle`
            Visit-level drawing parameters.
        det_meta : dict
            ``catalog.meta["det_meta"]``, read for this detector's
            ``n_quarter`` orientation -- which is per detector rather than per
            row, so it cannot come off ``row``.
        rejected : bool, optional
            Draw the rejection cross and colour the caption.
        """
        import matplotlib.patches as mpatches

        stamp = np.array(row[style.stamp_col])
        vmin, vmax = np.nanpercentile(stamp, [1, 99])
        edge = stamp.shape[0] // 2 + 0.5
        ax.imshow(
            stamp,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="gray",
            aspect="equal",
            extent=[-edge, edge, -edge, edge],
        )

        # The radius is per donut, so it rides the row; everything else shaping
        # these circles is visit-level and lives on `style`.
        donut_radius = row["donut_radius"].to_value(u.pix) * style.px_scale
        for radius, color, linestyle in style.circle_radii(donut_radius):
            ax.add_patch(
                mpatches.Circle(
                    (0, 0),
                    radius,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.0,
                    linestyle=linestyle,
                    alpha=0.45,
                    zorder=4,
                )
            )

        if rejected:
            ax.plot([-edge, edge], [-edge, edge], color=_COLOR_REJECTED, lw=1.5, zorder=5)
            ax.plot([-edge, edge], [edge, -edge], color=_COLOR_REJECTED, lw=1.5, zorder=5)

        self._drawRefcatOverlays(ax, row, style, det_meta)

        # Pin the view to the stamp's own edges. Two reasons, and both bite:
        # a stamp then fills its axes exactly and consumes the same figure area
        # whatever its pixel count; and ax.plot of the refcat overlays triggers
        # autoscale where add_patch alone does not, which would otherwise pull
        # in the annulus circles and shrink the stamp only on rows that happen
        # to carry overlays. A config-derived view (donut_radius *
        # bkgAnnulusOuterFrac, say) would instead couple the drawn size to
        # stampSize: at stampSize=215 the image overflows its axes by ~15%.
        ax.set_xlim(-edge, edge)
        ax.set_ylim(-edge, edge)

        # Bottom-anchored just above the axes, so the block grows upward and
        # never overlaps the stamp -- clearance is independent of stamp size.
        # (Top-anchoring inside the axes hung the text down over the image; at
        # stampSize 167 it overlapped by ~1pt, and worse for larger stamps.)
        ax.annotate(
            _donut_annotation(row, rejected),
            xy=(0.05, 1.00),
            xycoords="axes fraction",
            xytext=(0, 1.0),
            textcoords="offset points",
            fontsize=_STAMP_TEXT_FONTSIZE,
            va="bottom",
            ha="left",
            color=_COLOR_REJECTED if rejected else "black",
            bbox=dict(boxstyle="square,pad=0", fc="none", ec="none"),
            zorder=6,
            annotation_clip=False,
        )

    def _drawRefcatOverlays(self, ax, row, style: _StampStyle, det_meta: dict) -> None:
        """Mark the nearby photometric and astrometric refcat sources.

        Both overlays are drawn in the stamp's rotated display frame, which is
        what `_stamp_transform` builds; the two differ only in marker, colour
        and label offset.
        """
        to_display = _stamp_transform(row, style, det_meta)
        # The label y offsets differ in sign so a photometric and an
        # astrometric label on one source do not land on top of each other.
        overlays = (
            ("photo", _COLOR_PHOTO_REFCAT, dict(marker="o", mfc="none"), 3, 3),
            ("astrom", _COLOR_ASTROM_REFCAT, dict(marker="+"), 3, -5),
        )
        for kind, color, marker_kw, label_dx, label_dy in overlays:
            count = min(row[f"n_nearby_{kind}"], _MAX_NEARBY)
            dxs = row[f"nearby_{kind}_dx_det"][:count].to_value(u.pix)
            dys = row[f"nearby_{kind}_dy_det"][:count].to_value(u.pix)
            mags = row[f"nearby_{kind}_mag"][:count].to_value(u.mag)
            for dx, dy, mag in zip(dxs, dys, mags):
                tx, ty = to_display(dx, dy)
                ax.plot(tx, ty, ms=6, mec=color, mew=0.8, zorder=3, **marker_kw)
                if np.isfinite(mag):
                    ax.text(
                        tx + label_dx * style.px_scale,
                        ty + label_dy * style.px_scale,
                        f"{mag:.2f}",
                        color=color,
                        fontsize=_STAMP_TEXT_FONTSIZE,
                        zorder=4,
                    )

    def _drawWfImage(self, ax, img, cmap, vmin, vmax, label="") -> None:
        """One binned image or fitted model under a caller-chosen colormap.

        Takes the array and its scaling from the caller, since a row draws
        image, model and residual with shared limits.  Distinct from
        `_drawDonutStamp`, which reads a catalog row.
        """
        ax.imshow(
            img,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if label:
            ax.set_title(label, fontsize=5, pad=1)

    def _drawZkBar(self, ax, zk_dev, inset_label="") -> None:
        """Vertical bar chart of Zernikes in µm, ±1 µm, no tick labels.

        ``zk_dev`` is Noll-indexed (element j is Noll j) and in µm; it may be
        shorter than `_ZK_BAR_MAX` (or empty) since it stops at the highest
        fitted Noll index.

        Always spans `_ZK_BAR_MIN`..`_ZK_BAR_MAX` regardless of the configured
        ``nollIndices`` so plots stay comparable across configs; indices that
        were not fitted plot as zero rather than dropping out.
        """
        bar_noll = list(range(_ZK_BAR_MIN, _ZK_BAR_MAX + 1))
        values = [zk_dev[j] if j < len(zk_dev) and np.isfinite(zk_dev[j]) else 0.0 for j in bar_noll]
        ax.bar(bar_noll, values, color="k", width=0.8)
        ax.axhline(0, color="k", linewidth=0.4)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlim(_ZK_BAR_MIN - 0.5, _ZK_BAR_MAX + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for indices, color, alpha, spanned in _ZK_FAMILY_BANDS:
            for j in indices:
                ax.axvspan(j - 0.5, j + spanned - 0.5, color=color, alpha=alpha, ec="none")
        if inset_label:
            ax.text(
                0.03,
                0.97,
                inset_label,
                transform=ax.transAxes,
                fontsize=4,
                va="top",
                ha="left",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.6),
            )

    def _drawWfRowHalf(self, fig, inner, row_idx: int, col_start: int, half: _RowHalf) -> None:
        """Draw one side of focus on one row: image, model, residual, bar.

        Four grid columns starting at ``col_start``.  A blank half still claims
        all four axes and turns them off, because the corner grid is padded to
        a fixed height and the axes positions have to stay put -- see
        `_wf_row_pairs`.

        The image and model share one colour scale so they are visually
        comparable; the residual gets its own, and a symmetric colormap, since
        its scale is unrelated and usually much smaller.
        """
        if half.img is None:
            for offset in range(4):
                ax = fig.add_subplot(inner[row_idx, col_start + offset])
                ax.axis("off")
                if offset == 0 and half.det_hdr:
                    ax.set_title(half.det_hdr, fontsize=5, pad=1)
            return

        has_model = half.model is not None
        # `or 1.0` catches an all-zero or all-NaN percentile, which would
        # otherwise make vmin == vmax and render a uniform panel.
        vmax = np.nanpercentile(np.abs(half.img), 99) or 1.0
        resid = (half.img - half.model) if has_model else None
        vmax_resid = (np.nanpercentile(np.abs(resid), 99) or 1.0) if has_model else 1.0

        # The annotation each panel carries, if its value is finite: the
        # donut's id on the image, the fitted blur on the model, the blend
        # fraction on the residual.
        panels = (
            (half.img, _CMAP_DONUT, vmax, f"id={half.donut_id}" if half.donut_id is not None else ""),
            (
                half.model,
                _CMAP_DONUT,
                vmax,
                f"blur={half.fwhm:.2f}arcsec" if np.isfinite(half.fwhm) else "",
            ),
            (
                resid,
                _CMAP_DONUT_SYM,
                vmax_resid,
                f"blend={half.blend_frac:.3f}" if np.isfinite(half.blend_frac) else "",
            ),
        )
        for offset, (img, cmap, limit, annotation) in enumerate(panels):
            ax = fig.add_subplot(inner[row_idx, col_start + offset])
            # Only the leftmost panel of a row carries the column header.
            label = half.det_hdr if offset == 0 else ""
            if img is None:
                ax.axis("off")
                if label:
                    ax.set_title(label, fontsize=5, pad=1)
                continue
            self._drawWfImage(ax, img, cmap, -limit, limit, label=label)
            if annotation:
                ax.text(
                    0.02,
                    0.98,
                    annotation,
                    transform=ax.transAxes,
                    fontsize=4,
                    color="k",
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.6),
                )

        ax_bar = fig.add_subplot(inner[row_idx, col_start + 3])
        if has_model:
            self._drawZkBar(ax_bar, half.zk_dev, inset_label=half.bar_label)
        else:
            ax_bar.axis("off")

    def _saveWfDiagnosticPlot(self, catalog: QTable) -> None:
        """Save a WF diagnostic PNG modeled on the AOS donut-fits layout.

        Layout: 2×2 grid of corners (R00, R04, R40, R44).
        Within each corner: one row per fit result.
        Each row: intra data|model|resid|zk_bar  extra data|model|resid|zk_bar.
        Zernike bars are vertical, ±1 µm, no tick labels.

        Parameters
        ----------
        catalog : QTable
            Per-donut table from ``_buildCatalog``.
        """
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        if len(catalog) == 0:
            return

        meta = catalog.meta
        visit_id = meta["ref_visit_id"]
        refcat_elapsed = _meta_value(meta, "refcat_elapsed", u.s)
        butler_elapsed = _meta_value(meta, "butler_elapsed", u.s)
        butler_times = {key: value.to_value(u.s) for key, value in meta["butler_times"].items()}
        cutout_elapsed = _meta_value(meta, "cutout_elapsed", u.s)
        danish_elapsed = _meta_value(meta, "danish_elapsed", u.s)
        wf_mode = meta["wf_mode"]

        plottable, unfitted = _wf_groups_from_catalog(catalog)

        if not plottable and not unfitted:
            self.log.info("No WF results with model images; skipping WF diagnostic plot.")
            return

        corners = list(CORNER_PAIRS)
        det_id_of = _det_id_by_name(catalog)

        def _det_label(name):
            """Detector header, with the id only when the detector has rows.

            The 2x2 corner grid is always drawn in full, but a detector that
            was not processed (or contributed no donuts) is absent from the
            catalog and so has no id to report.
            """
            det_id = det_id_of.get(name)
            return name if det_id is None else f"{name} ({det_id})"

        row_pairs = _wf_row_pairs(plottable, unfitted, wf_mode, meta["max_donuts"])
        max_rows = len(next(iter(row_pairs.values())))

        layout = _WF_LAYOUT

        corner_w = 10 * layout.cell
        fig_w = 2 * corner_w + 0.3
        fig_h = 2 * max_rows * layout.row_h + 0.4

        fig = Figure(figsize=(fig_w, fig_h))
        outer = GridSpec(
            2,
            2,
            figure=fig,
            hspace=layout.hpad,
            wspace=0.06,
            left=0.01,
            right=0.99,
            top=0.94,
            bottom=0.01,
        )
        corner_pos = {"R00": (0, 0), "R40": (0, 1), "R04": (1, 0), "R44": (1, 1)}

        for corner in corners:
            pairs = row_pairs[corner]
            grow, gcol = corner_pos[corner]
            inner = GridSpecFromSubplotSpec(
                max_rows,
                8,
                subplot_spec=outer[grow, gcol],
                hspace=0.0,
                wspace=0.0,
                width_ratios=[1, 1, 1, 2, 1, 1, 1, 2],
            )
            # Only the first row of a corner carries the detector headers.
            headers = {
                "intra": f"intra {_det_label(f'{corner}_SW1')}",
                "extra": f"extra {_det_label(f'{corner}_SW0')}",
            }

            for row_idx, (group_intra, group_extra) in enumerate(pairs):
                for col_start, defocal, group in ((0, "intra", group_intra), (4, "extra", group_extra)):
                    half = _RowHalf.from_group(
                        group,
                        defocal,
                        det_hdr=headers[defocal] if row_idx == 0 else "",
                    )
                    self._drawWfRowHalf(fig, inner, row_idx, col_start, half)

        proc_total = refcat_elapsed + cutout_elapsed + danish_elapsed
        bt = butler_times or {}
        butler_line = (
            "  ".join(f"{k}={v:.1f}s" for k, v in bt.items() if v > 0.0) or f"total={butler_elapsed:.1f}s"
        )
        fig.suptitle(
            f"WF fits  visit={visit_id}  mode={wf_mode}\n"
            f"butler.get:  {butler_line}\n"
            f"refcat={refcat_elapsed:.1f}s  cutout={cutout_elapsed:.1f}s  "
            f"danish={danish_elapsed:.1f}s  proc_total={proc_total:.1f}s",
            fontsize=7,
        )
        fname = f"wf_diag_{visit_id}.png"
        # No bbox_inches="tight" here: it crops to drawn content, so the output
        # size would still shift with the number of populated rows even though
        # the layout is padded to max_rows. A fixed canvas keeps every plot for
        # a given config pixel-comparable.
        fig.savefig(fname, dpi=300)
        self.log.info("Saved WF diagnostic plot: %s", fname)

    def _saveFocalPlanePlot(
        self,
        catalog: QTable,
        detector_images: QTable | None,
        overlays: QTable | None,
    ) -> None:
        """Save the focal-plane selection plot per detector.

        One panel per corner sensor, each rotated by its own ``n_quarter`` so
        the eight panels together read as the focal plane, arranged as a 2x2 of
        corners.  Over each image go the three stages of the selection funnel
        and the donuts that survived, plus the ``maxFieldDist`` boundary; in
        ``paired`` mode, lines join each intra/extra pair across its corner.

        Skipped with one log line when the image table is absent -- the case
        where `DonutBlitzCornerTask` ran without ``doSelectionOutput`` and this
        task was pointed at the catalog alone.

        Parameters
        ----------
        catalog : QTable
            Per-donut table, read for the accepted/rejected donuts and pairing.
        detector_images : QTable or None
            Per-detector binned images.  None or empty skips the plot.
        overlays : QTable or None
            Long-form selection sources.  None draws images and donuts only.
        """
        from matplotlib.figure import Figure

        image_rows = _detector_image_rows(detector_images)
        if not image_rows:
            self.log.info("No detector images supplied; skipping focal-plane selection plot.")
            return

        # This plot's own meta, not the catalog's: the catalog is empty (and so
        # meta-less) exactly when every donut was rejected, which is a case
        # worth plotting.
        meta = detector_images.meta if detector_images is not None else {}
        visit_id = meta.get("ref_visit_id") or catalog.meta.get("ref_visit_id", 0)

        n_quarter_by_det = {name: int(row["n_quarter"]) for name, row in image_rows.items()}

        layout = _FP_LAYOUT
        fig = Figure(figsize=(layout.fig_side, layout.fig_side))
        axs = {
            name: fig.add_axes(rect)
            for name, rect in _focal_plane_axes_rects(n_quarter_by_det, layout).items()
        }

        det_name_col = np.asarray(catalog["det_name"], dtype=str) if len(catalog) else np.empty(0, dtype=str)
        for det_name, ax in axs.items():
            row = image_rows.get(det_name)
            if row is None:
                # A detector with no image still holds its cells, so the focal
                # plane keeps its shape and the gap is visible as a gap.
                ax.axis("off")
                ax.set_title(det_name, fontsize=6, color="0.5")
                continue
            det_rows = catalog[det_name_col == det_name] if len(catalog) else catalog
            self._drawFocalPlanePanel(ax, row, det_rows, _overlay_points(overlays, det_name), layout)

        fig.suptitle(
            f"Donut selection  visit={visit_id}  "
            f"binning={int(next(iter(image_rows.values()))['binning'])}",
            fontsize=11,
        )
        # The axes are already at their final rectangles from `add_axes`, so
        # there is nothing to adjust before drawing the links -- which matters,
        # because a `ConnectionPatch` resolves its endpoints against the two
        # axes' transforms and moving an axes afterwards would leave the line
        # where the panel was.
        self._drawPairLinks(fig, axs, image_rows, _pair_links(catalog))
        self._drawFocalPlaneLegend(fig, overlays)

        fname = f"selection_diag_{visit_id}.png"
        fig.savefig(fname, dpi=200)
        self.log.info("Saved focal-plane selection plot: %s", fname)

    def _drawFocalPlanePanel(self, ax, row, det_rows: QTable, points: dict, layout: _FpLayout) -> None:
        """One detector: its rotated image, the selection overlays, the donuts.

        Everything is drawn in the *displayed* frame, which is the binned image
        rotated by ``n_quarter``, so every overlay goes through
        `_binned_coord` then `_rot90_display`.
        """
        image = np.asarray(row["image"], dtype=float)
        n_quarter = int(row["n_quarter"])
        height, width = image.shape
        display = np.rot90(image, -n_quarter).T

        # nanpercentile because post-ISR saturation leaves NaNs. The fallback
        # covers a uniform or all-NaN panel, where vmin == vmax renders flat.
        finite = display[np.isfinite(display)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [1, 99])
        else:
            vmin, vmax = 0.0, 1.0
        if not vmax > vmin:
            vmin, vmax = float(vmin), float(vmin) + 1.0
        # origin="upper", and it is load-bearing rather than a default left in
        # place. The display transform is `rot90(arr, -n_quarter).T`, and that
        # transpose is a *reflection*, not a rotation -- it inverts handedness.
        # Drawing the result with y increasing upward leaves the inversion in,
        # so every panel comes out mirrored: the most-vignetted corner points
        # toward the focal-plane center instead of away from it, on all eight
        # sensors. Row-downward display supplies the compensating flip.
        # Verified against the camera: under "upper" the maximum-field cell of
        # all 8 corner sensors lands in the panel corner farthest from the
        # figure center, and under "lower" none of them do. The stamp plots
        # above are unaffected -- they draw an already-CCS stamp about its own
        # center, with no absolute detector frame to be handed-ness relative
        # to. aspect="auto", so the image fills the rectangle
        # `_focal_plane_axes_rects` assigned it. "equal" would re-derive the
        # box from the data's 2.036:1 and shrink the axes inside its rectangle,
        # putting the leftover back into the seams -- which is the bug being
        # fixed. The 1.8% stretch that buys exact tiling is deliberate; see
        # `_FP_PANEL_ASPECT`.
        ax.imshow(display, origin="upper", cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")

        def to_display(x_det, y_det):
            """Detector-frame coordinates to panel's display coordinates."""
            r, c = _binned_coord(x_det, y_det, row)
            return _rot90_display(r, c, height, width, n_quarter)

        # The selection funnel, widest stage first so the narrow ones draw on
        # top. The refcat is by far the most numerous, so it stays the
        # smallest and faintest: it is context, and it should not bury the
        # stages that matter.
        for kind, color, marker, size, alpha, lw in _FP_FUNNEL_STYLES:
            entry = points.get(kind)
            if entry is None:
                continue
            xs, ys, _ = entry
            # Brightest-first from `_overlay_points`, so this keeps the bright
            # end.
            xs, ys = xs[: layout.max_sources], ys[: layout.max_sources]
            tx, ty = to_display(xs, ys)
            ax.scatter(
                tx,
                ty,
                s=size,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=lw,
                alpha=alpha,
            )

        # The donuts that survived, and those a quality cut rejected, from the
        # catalog rather than the overlay table -- it is the one that carries
        # their metrics.
        if len(det_rows):
            candidate = np.asarray(det_rows["candidate"], dtype=bool)
            accepted_marker, accepted_size, accepted_lw = _FP_ACCEPTED_MARKER
            # The accepted marker is hollow so the donut inside it stays
            # visible; the rejected one is an "x", which matplotlib treats as
            # an unfilled marker and colours from `color` rather than
            # `edgecolors` (passing the latter earns a warning and the wrong
            # color). So the two need different keyword sets, not a shared one.
            # The rejected marker keeps the old size: an "x" is already
            # unmistakable against the funnel's rings, so it does not need the
            # accepted marker's extra room.
            for mask, color, marker, size, lw, kwargs in (
                (
                    candidate,
                    _COLOR_ACCEPTED,
                    accepted_marker,
                    accepted_size,
                    accepted_lw,
                    dict(facecolors="none", edgecolors=_COLOR_ACCEPTED),
                ),
                (~candidate, _COLOR_REJECTED, "x", 110.0, 1.2, dict(color=_COLOR_REJECTED)),
            ):
                if not mask.any():
                    continue
                sub = det_rows[mask]
                tx, ty = to_display(sub["x_det"].to_value(u.pix), sub["y_det"].to_value(u.pix))
                ax.scatter(tx, ty, s=size, linewidths=lw, marker=marker, alpha=0.9, **kwargs)
                for x, y, snr in zip(tx, ty, np.asarray(sub["snr"], dtype=float)):
                    if not np.isfinite(snr):
                        continue
                    ax.annotate(
                        f"{snr:.0f}",
                        (x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        color=color,
                        fontsize=4.5,
                        annotation_clip=True,
                    )

        ax.set_xticks([])
        ax.set_yticks([])
        # Before the field-limit patch: `ax.plot`/`add_patch` interact badly
        # with autoscale (see `_drawDonutStamp`), and this curve runs far
        # outside the detector, so an autoscaled panel would shrink the image
        # to nothing.
        ax.set_xlim(-0.5, display.shape[1] - 0.5)
        # Descending, matching origin="upper": row 0 at the top. Setting this
        # ascending would undo the flip the `imshow` above depends on, which is
        # how the mirrored version of this plot came about.
        ax.set_ylim(display.shape[0] - 0.5, -0.5)
        excluded_frac = self._drawFieldLimit(ax, row, n_quarter)

        det_name = str(row["det_name"])
        defocal = CORNER_DEFOCAL_BY_DET_NAME.get(det_name, "")
        det_id = int(row["det_id"])
        # det_id is -1 for a detector that produced no donuts, since the table
        # reads it off one of them; omit it rather than print the placeholder.
        label = f"{det_name} ({det_id}) {defocal}" if det_id >= 0 else f"{det_name} {defocal}"
        # How much of this sensor the vignetting cut removes. Worth stating as
        # a number: it is ~53% on an intra-focal sensor against ~5% on an
        # extra-focal one, which is a big asymmetry to leave to the eye.
        if np.isfinite(excluded_frac) and excluded_frac > 0:
            label += f"  [{100 * excluded_frac:.0f}% vignetted]"
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            color="w",
            fontsize=6,
            va="top",
            ha="left",
        )
        if not bool(row["has_refcat"]):
            # Says *why* there is no refcat overlay, which its mere absence
            # cannot: this detector fell back to blitz detection.
            ax.text(
                0.02,
                0.02,
                f"no refcat ({row['selection_source']})",
                transform=ax.transAxes,
                color=_COLOR_REJECTED,
                fontsize=5,
                va="bottom",
                ha="left",
            )

    def _drawFieldLimit(self, ax, row, n_quarter: int) -> float:
        """Shade and outline the region the vignetting cut excludes.

        ``field_dist`` is a grid on the same pixel grid as the image, so it
        takes the same rotation and needs no coordinate transform of its own.
        Shading it rather than outlining a boundary curve is what makes the
        cut legible: the boundary is an arc of ~31000 px radius across a 4072
        px sensor, so as a line it is an unremarkable near-straight chord,
        while as a shaded region it shows at a glance that over half of each
        intra-focal sensor is unusable.

        Returns
        -------
        float
            Fraction of the detector excluded, for the panel's label.  NaN when
            the grid is unavailable.
        """
        field_dist = np.asarray(row["field_dist"].to_value(u.deg), dtype=float)
        limit = float(row["max_field_dist"].to_value(u.deg))
        if not np.isfinite(field_dist).any():
            return float("nan")

        excluded = np.rot90(field_dist, -n_quarter).T > limit
        if not excluded.any():
            return 0.0

        # Hatched overlay rather than a flat tint: the underlying pixels stay
        # readable, which matters because donuts do land in this region and get
        # cut, and seeing them is the point.
        ax.contourf(
            excluded.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["////"],
            extend="neither",
        )
        ax.contour(
            excluded.astype(float),
            levels=[0.5],
            colors=[_COLOR_FIELD_LIMIT],
            linewidths=1.2,
            linestyles="--",
        )
        return float(excluded.mean())

    def _drawPairLinks(self, fig, axs: dict, image_rows: dict, links: list) -> None:
        """Join each intra/extra donut pair with a line across the two panels.

        Empty outside ``paired`` mode -- see `_pair_links`.  A link whose
        either end is on a detector with no panel is skipped rather than
        half-drawn.
        """
        from matplotlib.patches import ConnectionPatch

        def _xy(det_name, x_det, y_det):
            """One donut's position in its own panel's display coordinates."""
            row = image_rows[det_name]
            image = np.asarray(row["image"])
            r, c = _binned_coord(x_det, y_det, row)
            return _rot90_display(r, c, image.shape[0], image.shape[1], int(row["n_quarter"]))

        for (det_a, x_a, y_a), (det_b, x_b, y_b) in links:
            if det_a not in axs or det_b not in axs:
                continue
            if det_a not in image_rows or det_b not in image_rows:
                continue
            fig.add_artist(
                ConnectionPatch(
                    xyA=_xy(det_a, x_a, y_a),
                    xyB=_xy(det_b, x_b, y_b),
                    coordsA=axs[det_a].transData,
                    coordsB=axs[det_b].transData,
                    arrowstyle="-",
                    color=_COLOR_PAIR_LINK,
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.9,
                    # Without this the line is clipped at each axes' edge,
                    # which is exactly the span it exists to cross.
                    clip_on=False,
                )
            )

    def _drawFocalPlaneLegend(self, fig, overlays: QTable | None) -> None:
        """One shared legend for the selection funnel and the donut markers."""
        from matplotlib.lines import Line2D

        def marker(color, label, marker="o"):
            return Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="none",
                markeredgecolor=color,
                markersize=6,
                label=label,
            )

        handles = []
        if overlays is not None and len(overlays):
            # Straight off the style table the panels draw from, so the legend
            # cannot disagree with the markers it is labelling.
            handles += [
                marker(color, label, marker=mk)
                for (_, color, mk, _, _, _), label in zip(
                    _FP_FUNNEL_STYLES, ("refcat", "blitz detection", "selected")
                )
            ]
        handles += [
            marker(_COLOR_ACCEPTED, "donut (candidate)", marker=_FP_ACCEPTED_MARKER[0]),
            marker(_COLOR_REJECTED, "donut (rejected)", marker="x"),
            Line2D([0], [0], color=_COLOR_FIELD_LIMIT, linestyle="--", label="maxFieldDist"),
            Line2D([0], [0], color=_COLOR_PAIR_LINK, linestyle=":", label="intra/extra pair"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            fontsize=7,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.4,
        )
