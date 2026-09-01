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

"""Diagnostic plots regenerated from the ``donutBlitzResults`` catalog."""

__all__ = [
    "DonutBlitzPlotTaskConnections",
    "DonutBlitzPlotTaskConfig",
    "DonutBlitzPlotTask",
]

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
    CORNER_BY_DET_NAME,
    CORNER_PAIRS,
    _MAX_NEARBY,
    _ZK_JMAX,
    _resolveColorLogEnabled,
)

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


def _detIdByName(catalog: QTable) -> dict[str, int]:
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


class DonutBlitzPlotTaskConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit"),  # type: ignore
):
    """Pipeline connections for DonutBlitzPlotTask."""

    blitzResults = connectionTypes.Input(
        doc=(
            "Per-donut catalog from DonutBlitzMonolithTask containing all data "
            "needed to regenerate diagnostic plots."
        ),
        name="donutBlitzResults",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        deferLoad=True,
    )


class DonutBlitzPlotTaskConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=DonutBlitzPlotTaskConnections,  # type: ignore
):
    """Configuration for DonutBlitzPlotTask."""

    colorLog: pexConfig.Field = pexConfig.Field(
        doc=(
            "Colorize select log messages with ANSI escape codes. If None "
            "(the default), color is enabled only when stdout is an "
            "interactive terminal."
        ),
        dtype=bool,
        default=None,
        optional=True,
    )


class DonutBlitzPlotTask(pipeBase.PipelineTask):
    """PipelineTask that regenerates diagnostic plots from ``donutBlitzResults``.

    Can run standalone (reading from the butler) or be called as a subtask of
    ``DonutBlitzMonolithTask`` when ``savePlots=True``.
    """

    ConfigClass = DonutBlitzPlotTaskConfig
    _DefaultName = "donutBlitzPlotTask"
    config: DonutBlitzPlotTaskConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._colorLogEnabled = _resolveColorLogEnabled(self.config.colorLog)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        catalog = inputs["blitzResults"].get(parameters={"strip_astropy_meta_yaml": False})
        self.run(catalog)

    def run(self, catalog: Table) -> None:
        """Generate donut and WF diagnostic plots from the blitzResults catalog.

        Parameters
        ----------
        catalog : QTable
            Per-donut table as produced by
            ``DonutBlitzMonolithTask._buildCatalog``.  Visit-level and
            per-detector metadata are in ``catalog.meta``.
        """
        catalog = QTable(catalog)
        self._saveDonutDiagnosticPlot(catalog)
        self._saveWfDiagnosticPlot(catalog)

    def _saveDonutDiagnosticPlot(self, catalog: QTable) -> None:
        """Save a single diagnostic PNG with one section per detector.

        Layout per detector:
          - Left column: stats text (timing, WCS scatter, donut count, errors)
          - Remaining columns: donut stamps (up to maxDonuts), each annotated
            with flux and field angle

        Parameters
        ----------
        catalog : QTable
            Per-donut table from ``_buildCatalog``.  Per-detector metadata is in
            ``catalog.meta["det_meta"]``; visit-level scalars are in
            ``catalog.meta``.
        """
        import matplotlib.patches as mpatches
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec

        if len(catalog) == 0:
            return

        meta = catalog.meta
        run_elapsed = meta["run_elapsed"]
        refcat_elapsed = meta["refcat_elapsed"]
        butler_elapsed = meta["butler_elapsed"]
        photo_filter_label = meta["photo_filter_name"]
        astrom_filter_label = meta["astrom_filter_name"]
        visit_id = meta["visit_id"]
        det_meta = meta["det_meta"]

        # Group rows by detector; split on selection outcome. This plot is about
        # donut *selection*, so it splits on "candidate" -- a candidate that no
        # fit consumed still shows in the accepted panel, since it passed every
        # cut this plot reports on.
        det_name_col = np.asarray(catalog["det_name"], dtype=str)
        det_id_by_name = _detIdByName(catalog)
        dets_with_data = []
        for det_name in sorted(set(det_name_col.tolist())):
            det_rows = catalog[det_name_col == det_name]
            acc = det_rows[det_rows["candidate"]]
            rej = det_rows[~det_rows["candidate"]]
            if len(acc) > 0 or len(rej) > 0:
                dets_with_data.append((det_name, acc, rej))

        n_dets = len(dets_with_data)
        if n_dets == 0:
            return

        STAMPS_PER_ROW = 8
        REJECTED_PER_ROW = 2
        STAMP_COL_W = 1.8 * 1.05
        STATS_COL_W = 2.8
        ROW_H = 1.7 * 1.05
        LEGEND_H = 0.35
        SPACER_W = 0.15
        SUPTITLE_H = 0.55

        N_COLS = 1 + STAMPS_PER_ROW + 1 + REJECTED_PER_ROW
        fig_w = STATS_COL_W + (STAMPS_PER_ROW + REJECTED_PER_ROW) * STAMP_COL_W + SPACER_W
        fig_h = n_dets * ROW_H + LEGEND_H + SUPTITLE_H

        fig = Figure(figsize=(fig_w, fig_h), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.0, wspace=0.0)
        butler_str = f"  butler={butler_elapsed:.1f}s" if butler_elapsed > 0 else ""
        fig.suptitle(
            f"DonutBlitz diagnostics  visit={visit_id}"
            f"  refcat={refcat_elapsed:.1f}s{butler_str}  run={run_elapsed:.1f}s",
            fontsize=9,
        )

        w_stats = STATS_COL_W / STAMP_COL_W
        w_spacer = SPACER_W / STAMP_COL_W
        gs = GridSpec(
            n_dets + 1,
            N_COLS,
            figure=fig,
            height_ratios=[ROW_H] * n_dets + [LEGEND_H],
            width_ratios=[w_stats] + [1] * STAMPS_PER_ROW + [w_spacer] + [1] * REJECTED_PER_ROW,
        )
        COL_ACCEPTED_START = 1
        COL_SPACER = 1 + STAMPS_PER_ROW
        COL_REJECTED_START = COL_SPACER + 1

        # Per-detector radius/obscuration ride each donut row (from
        # Donut.donut_radius), so a detector's aperture/annulus circles match its
        # own detected donut size. The visit-level meta scalars below are the
        # fallback for rows that lack the columns -- e.g. an older blitzResults
        # catalog written before these columns existed, read back by a
        # standalone DonutBlitzPlotTask.
        _stamp_outer_margin_frac = catalog.meta["aperture_outer_margin_frac"]
        _stamp_inner_buffer_frac = catalog.meta["aperture_inner_buffer_frac"]
        _stamp_bkg_inner_frac = catalog.meta["bkg_annulus_inner_frac"]
        _stamp_bkg_outer_frac = catalog.meta["bkg_annulus_outer_frac"]

        # Every stamp's view is pinned to its own pixel extent, so a stamp fills
        # its axes exactly and consumes the same figure area no matter how many
        # pixels it contains. Setting the limits explicitly (rather than leaving
        # them to autoscale) is also required because ax.plot of the refcat
        # overlays triggers autoscale where add_patch alone does not, which would
        # otherwise pull in the annulus circles and shrink the stamp only on rows
        # that happen to have overlays.
        #
        # A config-derived view (e.g. donutRadius * bkgAnnulusOuterFrac) would
        # instead couple the drawn size to stampSize: at stampSize=215 the image
        # overflows its axes by ~15%.
        _STAMP_TEXT_FONTSIZE = 3.5

        def _draw_stamp(ax, row, rejected=False):
            stamp = np.array(row["stamp"])
            h_px = stamp.shape[0] // 2
            vmin, vmax = np.nanpercentile(stamp, [1, 99])
            _edge = h_px + 0.5
            ax.imshow(
                stamp,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap="gray",
                aspect="equal",
                extent=[-_edge, _edge, -_edge, _edge],
            )

            # Per-detector radius/obscuration off the row, falling back to the
            # visit-level meta scalar if a row lacks a finite value (stale
            # catalog). row is an astropy Row, so membership is via colnames.
            dr = row["donut_radius"]
            ob = row["obscuration"]
            if dr is not None and ob is not None:
                _circ_specs = [
                    (dr * ob * _stamp_inner_buffer_frac, _COLOR_BKG_ANNULUS, "--"),
                    (dr * ob, _COLOR_APERTURE, "-"),
                    (dr * _stamp_outer_margin_frac, _COLOR_APERTURE, "-"),
                    (dr * _stamp_bkg_inner_frac, _COLOR_BKG_ANNULUS, "--"),
                    (dr * _stamp_bkg_outer_frac, _COLOR_BKG_ANNULUS, "--"),
                ]
                for _rad, _col, _ls in _circ_specs:
                    ax.add_patch(
                        mpatches.Circle(
                            (0, 0),
                            _rad,
                            fill=False,
                            edgecolor=_col,
                            linewidth=1.0,
                            linestyle=_ls,
                            alpha=0.45,
                            zorder=4,
                        )
                    )

            if rejected:
                ax.plot([-_edge, _edge], [-_edge, _edge], color=_COLOR_REJECTED, lw=1.5, zorder=5)
                ax.plot([-_edge, _edge], [_edge, -_edge], color=_COLOR_REJECTED, lw=1.5, zorder=5)

            nq = row["n_quarter"] % 4

            def _xform(dx, dy):
                """Map a raw-pixel offset to stamp display coords.

                Must mirror the stamp transform in `_cut_and_evaluate_stamps`,
                ``np.rot90(stamp, k=-n_quarter).T`` -- including the transpose.
                The loop applies the rot90 in (row, col) space; returning
                ``(r, c)`` rather than ``(c, r)`` is what applies the ``.T``.
                Under ``origin="lower"`` the displayed x axis is the column
                index and y is the row index, so the returned pair is
                ``(x, y)`` after transposition.
                """
                r, c = dy, dx
                for _ in range(nq):
                    r, c = c, -r
                return r, c

            n_photo = min(row["n_nearby_photo"], _MAX_NEARBY)
            px = row["nearby_photo_x"][:n_photo].to_value(u.pix)
            py = row["nearby_photo_y"][:n_photo].to_value(u.pix)
            pm = row["nearby_photo_mag"][:n_photo].to_value(u.mag)
            for dx, dy, mag in zip(px, py, pm):
                tx, ty = _xform(dx, dy)
                ax.plot(tx, ty, "o", ms=6, mfc="none", mec=_COLOR_PHOTO_REFCAT, mew=0.8, zorder=3)
                if np.isfinite(mag):
                    ax.text(tx + 3, ty + 3, f"{mag:.2f}", color=_COLOR_PHOTO_REFCAT, fontsize=3.5, zorder=4)

            n_astrom = min(row["n_nearby_astrom"], _MAX_NEARBY)
            ax_ = row["nearby_astrom_x"][:n_astrom].to_value(u.pix)
            ay_ = row["nearby_astrom_y"][:n_astrom].to_value(u.pix)
            am_ = row["nearby_astrom_mag"][:n_astrom].to_value(u.mag)
            for dx, dy, mag in zip(ax_, ay_, am_):
                tx, ty = _xform(dx, dy)
                ax.plot(tx, ty, "+", ms=6, mec=_COLOR_ASTROM_REFCAT, mew=0.8, zorder=3)
                if np.isfinite(mag):
                    ax.text(tx + 3, ty - 5, f"{mag:.2f}", color=_COLOR_ASTROM_REFCAT, fontsize=3.5, zorder=4)
            # Pin the view to the stamp's own edges: constant figure footprint
            # regardless of pixel count, and no autoscale from the overlays.
            ax.set_xlim(-_edge, _edge)
            ax.set_ylim(-_edge, _edge)

            inner_frac = row["inner_frac"]
            outer_frac = row["outer_frac"]
            outer_sector_minmax = row["outer_sector_minmax_frac"]
            snr = row["snr"]
            if_str = f"if={inner_frac:.3f}" if np.isfinite(inner_frac) else "if=?"
            of_str = f"of={outer_frac:.3f}" if np.isfinite(outer_frac) else "of=?"
            osm_str = f"osm={outer_sector_minmax:.3f}" if np.isfinite(outer_sector_minmax) else "osm=?"
            snr_str = f"snr={snr:.0f}" if np.isfinite(snr) else "snr=?"
            sid = row["id"]
            sid_str = f"id={sid}" if sid != 0 else ""
            _text_color = _COLOR_REJECTED if rejected else "black"

            _flags = [
                name
                for name, val in (
                    ("sat", row["rejected_sat"]),
                    ("inner", row["rejected_inner_frac"]),
                    ("outer", row["rejected_outer_frac"]),
                    ("snr", row["rejected_snr"]),
                )
                if val
            ]
            rej_str = f"[{'|'.join(_flags)}]" if _flags else ""
            # Bottom-anchored just above the axes, so the block grows upward and
            # never overlaps the stamp -- clearance is independent of stamp size.
            # (Top-anchoring inside the axes hung the text down over the image;
            # at stampSize 167 it overlapped by ~1pt, and worse for larger stamps.)
            ax.annotate(
                f"{snr_str}  {rej_str}\n{if_str}  {of_str}  {osm_str}\n{sid_str}",
                xy=(0.05, 1.00),
                xycoords="axes fraction",
                xytext=(0, 1.0),
                textcoords="offset points",
                fontsize=_STAMP_TEXT_FONTSIZE,
                va="bottom",
                ha="left",
                color=_text_color,
                bbox=dict(boxstyle="square,pad=0", fc="none", ec="none"),
                zorder=6,
                annotation_clip=False,
            )

        for row_idx, (det_name, acc_rows, rej_rows) in enumerate(dets_with_data):
            sm = det_meta.get(det_name, {})
            scatter_val = sm.get("scatter_arcsec", float("nan"))
            scatter_str = f'{scatter_val:.3f}"' if np.isfinite(scatter_val) else "N/A"

            ax_stats = fig.add_subplot(gs[row_idx, 0])
            ax_stats.axis("off")
            lines = [
                f"{det_name} ({det_id_by_name[det_name]})",
                f"donuts: {len(acc_rows)}",
                f"isr:    {sm.get('isr_run', float('nan')):.3f}s",
                f"bkg:    {sm.get('bkg_run', float('nan')):.3f}s",
                f"diam:   {sm.get('diam_run', float('nan')):.3f}s",
                f"detect: {sm.get('blind_detect_run', float('nan')):.3f}s",
                f"wcs:    {sm.get('wcs_refit_run', float('nan')):.3f}s  ({scatter_str})",
                f"select: {sm.get('catalog_select_run', float('nan')):.3f}s",
            ]
            if sm.get("wcs_refit_error"):
                lines.append(f"WCS ERR: {sm['wcs_refit_error'][:40]}")
            if sm.get("cat_select_error"):
                lines.append(f"CAT ERR: {sm['cat_select_error'][:40]}")
            ax_stats.text(
                0.05, 0.95, "\n".join(lines), transform=ax_stats.transAxes,
                fontsize=6, va="top", family="monospace",
            )

            for col_idx in range(STAMPS_PER_ROW):
                ax = fig.add_subplot(gs[row_idx, COL_ACCEPTED_START + col_idx])
                ax.axis("off")
                if col_idx >= len(acc_rows):
                    continue
                _draw_stamp(ax, acc_rows[col_idx])

            ax_sp = fig.add_subplot(gs[row_idx, COL_SPACER])
            ax_sp.axis("off")

            for col_idx in range(REJECTED_PER_ROW):
                ax = fig.add_subplot(gs[row_idx, COL_REJECTED_START + col_idx])
                ax.axis("off")
                if col_idx >= len(rej_rows):
                    continue
                _draw_stamp(ax, rej_rows[col_idx], rejected=True)

        ax_legend = fig.add_subplot(gs[n_dets, :])
        ax_legend.axis("off")
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="none",
                markeredgecolor=_COLOR_PHOTO_REFCAT, markersize=6,
                label=f"photo refcat ({photo_filter_label})",
            ),
            Line2D(
                [0], [0], marker="+", color=_COLOR_ASTROM_REFCAT, markersize=6, linestyle="none",
                label=f"astrom refcat ({astrom_filter_label})",
            ),
        ]
        ax_legend.legend(
            handles=legend_handles, loc="center", ncol=2, fontsize=7,
            frameon=False, handletextpad=0.5, columnspacing=2.0,
        )

        fname = f"donut_diag_{visit_id}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        self.log.info("Saved diagnostic plot: %s", fname)

    def _saveWfDiagnosticPlot(self, catalog: QTable) -> None:
        """Save a WF diagnostic PNG modelled on the AOS donut-fits layout.

        Layout: 2×2 grid of corners (R00, R04, R40, R44).
        Within each corner: one row per fit result.
        Each row: intra data|model|resid|zk_bar  extra data|model|resid|zk_bar.
        Zernike bars are vertical, ±1 µm, no tick labels.

        Parameters
        ----------
        catalog : QTable
            Per-donut table from ``_buildCatalog``.
        """
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.figure import Figure
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        if len(catalog) == 0:
            return

        meta = catalog.meta
        visit_id = meta["visit_id"]
        refcat_elapsed = meta["refcat_elapsed"]
        butler_elapsed = meta["butler_elapsed"]
        butler_times = meta["butler_times"]
        cutout_elapsed = meta["cutout_elapsed"]
        danish_elapsed = meta["danish_elapsed"]
        wf_mode = meta["wf_mode"]
        ZK_MIN, ZK_MAX = 4, 28

        # Reconstruct wf_results-like list from QTable by grouping on "group" column.
        # Include only rows with a valid fit (fit_mode != "" and group >= 0).
        groups: dict[int, list] = {}
        for row in catalog:
            fm = row["fit_mode"]
            grp = row["group"]
            if grp < 0 or fm == "":
                continue
            if grp not in groups:
                groups[grp] = []
            groups[grp].append(row)

        plottable = []
        for grp_idx, rows in sorted(groups.items()):
            first = rows[0]
            # Determine if this group has a real model (any non-NaN model_img).
            has_model = any(
                not np.all(np.isnan(np.array(r["model_img"]))) for r in rows
            )
            if not has_model:
                continue
            det_names = list(dict.fromkeys(r["det_name"] for r in rows))
            mode = first["fit_mode"]
            success = bool(first["fit_success"])
            elapsed = first["fit_elapsed"]
            nfev = first["fit_nfev"]
            fwhm = first["fit_fwhm"]
            zk_by_noll = {j: first[f"Z{j}_dev"] for j in range(4, _ZK_JMAX + 1)}

            donuts_out = []
            for r in rows:
                model_arr = np.array(r["model_img"])
                donuts_out.append({
                    "donut_id": r["id"],
                    "det_name": r["det_name"],
                    "defocal": r["defocal"],
                    "img": np.array(r["wf_img"]),
                    "model_img": model_arr if not np.all(np.isnan(model_arr)) else None,
                    "blend_frac": r["blend_frac"],
                    "elapsed": elapsed,
                    "nfev": nfev,
                    "fwhm": fwhm,
                    "success": success,
                })
            plottable.append({
                "mode": mode,
                "det_names": det_names,
                "success": success,
                "fit_info": {"elapsed": elapsed, "nfev": nfev, "fwhm": fwhm},
                "donuts": donuts_out,
                "zk_by_noll": zk_by_noll,
            })

        # Candidate donuts that no fit consumed (paired-mode surplus: no partner
        # on the other detector, so ``group`` is -1 and ``fit_mode`` empty). They
        # have no model or Zernikes, but their binned stamp is still worth
        # seeing, so carry them as data-only single-donut records.
        unfitted = []
        for row in catalog:
            if row["group"] >= 0 or not row["candidate"]:
                continue
            img = np.array(row["wf_img"])
            if np.all(np.isnan(img)):
                continue
            unfitted.append({
                "mode": wf_mode,
                "det_names": [row["det_name"]],
                "success": False,
                "fit_info": {
                    "elapsed": float("nan"), "nfev": 0, "fwhm": float("nan"),
                },
                "donuts": [{
                    "donut_id": row["id"],
                    "det_name": row["det_name"],
                    "defocal": row["defocal"],
                    "img": img,
                    "model_img": None,
                    "blend_frac": row["blend_frac"],
                }],
                "zk_by_noll": {},
            })

        if not plottable and not unfitted:
            self.log.info("No WF results with model images; skipping WF diagnostic plot.")
            return

        _CORNERS = list(CORNER_PAIRS)
        det_id_by_name = _detIdByName(catalog)

        def _corner_of(r):
            for s in r["det_names"]:
                if str(s) in CORNER_BY_DET_NAME:
                    return CORNER_BY_DET_NAME[str(s)]
            return _CORNERS[0]

        # 4-stop diverging colormap: blue → white (zero) → vermillion.
        # Anchors: -vmax=blue, -vmax/10=sky blue, 0=white, +vmax=vermillion.
        # Normalised positions over [-vmax, vmax]: 0.0, 0.45, 0.5, 1.0.
        def _hex_to_rgb(h):
            return tuple(int(h[i : i + 2], 16) / 255 for i in (1, 3, 5))

        _cmap_bwr = LinearSegmentedColormap.from_list(
            "bwr_donut",
            list(zip(
                [0.0, 0.45, 0.5, 1.0],
                [_hex_to_rgb(h) for h in (
                    _COLOR_CMAP_NEG, _COLOR_CMAP_MID, "#FFFFFF", _COLOR_CMAP_POS,
                )],
            )),
        )
        _cmap_bwr_sym = LinearSegmentedColormap.from_list(
            "bwr_donut_sym",
            list(zip(
                [0.0, 0.5, 1.0],
                [_hex_to_rgb(h) for h in (_COLOR_CMAP_NEG, "#FFFFFF", _COLOR_CMAP_POS)],
            )),
        )

        def _draw_stamp(ax, img, cmap, vmin, vmax, label=""):
            ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if label:
                ax.set_title(label, fontsize=5, pad=1)

        def _draw_bar(ax, zk_by_noll, inset_label=""):
            """Vertical bar chart of Zernikes in µm, ±1 µm, no tick labels.

            Always spans ZK_MIN..ZK_MAX regardless of the configured
            ``nollIndices`` so plots stay comparable across configs; indices
            that were not fitted plot as zero rather than dropping out.
            """
            bar_noll = [j for j in range(ZK_MIN, ZK_MAX + 1)]
            vals = [
                v if np.isfinite(v := zk_by_noll.get(j, 0.0)) else 0.0
                for j in bar_noll
            ]
            ax.bar(bar_noll, vals, color="k", width=0.8)
            ax.axhline(0, color="k", linewidth=0.4)
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlim(ZK_MIN - 0.5, ZK_MAX + 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for j in [4, 11, 22]:   # spherical (m=0)
                ax.axvspan(j - 0.5, j + 0.5, color="#000000", alpha=0.15, ec="none")
            for j in [7, 16]:       # coma (m=1)
                ax.axvspan(j - 0.5, j + 1.5, color=_COLOR_COMA, alpha=0.35, ec="none")
            for j in [5, 12, 23]:   # astigmatism (m=2)
                ax.axvspan(j - 0.5, j + 1.5, color=_COLOR_ASTIGMATISM, alpha=0.25, ec="none")
            for j in [9, 18]:       # trefoil (m=3)
                ax.axvspan(j - 0.5, j + 1.5, color=_COLOR_TREFOIL, alpha=0.25, ec="none")
            for j in [14, 25]:      # quadrafoil (m=4)
                ax.axvspan(j - 0.5, j + 1.5, color=_COLOR_QUADRAFOIL, alpha=0.25, ec="none")
            ax.axvspan(19.5, 21.5, color=_COLOR_PENTAFOIL, alpha=0.25, ec="none")   # pentafoil (m=5)
            ax.axvspan(26.5, 28.5, color=_COLOR_HEXAFOIL, alpha=0.25, ec="none")   # hexafoil (m=6)
            if inset_label:
                ax.text(0.03, 0.97, inset_label, transform=ax.transAxes, fontsize=4,
                        va="top", ha="left",
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.6))

        by_corner: dict[str, list] = {c: [] for c in _CORNERS}
        for r in plottable:
            by_corner[_corner_of(r)].append(r)

        unfitted_by_corner: dict[str, list] = {c: [] for c in _CORNERS}
        for r in unfitted:
            unfitted_by_corner[_corner_of(r)].append(r)


        def _explode(r):
            """Split a group record into one record per donut, sharing group fields."""
            return [{**r, "donuts": [d]} for d in r.get("donuts", [])]

        def _pair_up(records):
            """Lay single-donut records out as (intra, extra) plot rows.

            Only for modes whose groups carry no intra/extra pairing of their
            own: the pairing here is cosmetic, so rows are matched by position
            and the shorter side padded with None to keep every donut visible.
            """
            intras = [r for r in records if r["donuts"][0].get("defocal") == "intra"]
            extras = [r for r in records if r["donuts"][0].get("defocal") == "extra"]
            return [
                (intras[i] if i < len(intras) else None,
                 extras[i] if i < len(extras) else None)
                for i in range(max(len(intras), len(extras)))
            ]

        row_pairs: dict[str, list[tuple]] = {}
        for corner, corner_results in by_corner.items():
            if wf_mode == "paired":
                # The group *is* an intra/extra pair, so it supplies both halves
                # of the row; the donut of each defocal type is picked out below.
                fit_rows = [(r, r) for r in corner_results]
            else:
                # These groups don't pair donuts, so flatten to one record per
                # donut and pair for layout only. Exploding is a no-op for
                # "unpaired" (one donut per group already).
                fit_rows = _pair_up(
                    [s for r in corner_results for s in _explode(r)]
                )
            # Surplus donuts have no partner by construction, so they lay out
            # positionally below the fitted rows, one side of each row blank.
            row_pairs[corner] = fit_rows + _pair_up(unfitted_by_corner[corner])

        CELL = 1.0
        ROW_H = 1.0
        HPAD = 0.08
        # Always lay out maxDonuts rows per corner, padding short corners with
        # blank rows, so figure dimensions and axes positions depend only on
        # config -- not on how many donuts a given mode happened to fit. This
        # makes plots for the same exposure blinkable across fitting modes.
        # Never fewer rows than any corner actually has, so a corner with more
        # fits than maxDonuts grows the layout rather than losing rows.
        max_rows = max(
            [meta["max_donuts"], 1]
            + [len(v) for v in row_pairs.values()]
        )
        for corner, pairs in row_pairs.items():
            row_pairs[corner] = pairs + [(None, None)] * (max_rows - len(pairs))

        corner_w = 10 * CELL
        fig_w = 2 * corner_w + 0.3
        fig_h = 2 * max_rows * ROW_H + 0.4

        fig = Figure(figsize=(fig_w, fig_h))
        outer = GridSpec(2, 2, figure=fig, hspace=HPAD, wspace=0.06,
                         left=0.01, right=0.99, top=0.94, bottom=0.01)
        corner_pos = {"R00": (0, 0), "R40": (0, 1), "R04": (1, 0), "R44": (1, 1)}

        for corner in _CORNERS:
            pairs = row_pairs[corner]
            grow, gcol = corner_pos[corner]
            inner = GridSpecFromSubplotSpec(
                max_rows, 8, subplot_spec=outer[grow, gcol],
                hspace=0.0, wspace=0.0,
                width_ratios=[1, 1, 1, 2, 1, 1, 1, 2],
            )
            sw1 = f"{corner}_SW1 ({det_id_by_name[f'{corner}_SW1']})"
            sw0 = f"{corner}_SW0 ({det_id_by_name[f'{corner}_SW0']})"

            def _rec_info(r):
                if r is None:
                    return float("nan"), 0, False, float("nan")
                fi = r.get("fit_info", {})
                return (fi.get("elapsed", float("nan")), fi.get("nfev", 0),
                        r.get("success", False), fi.get("fwhm", float("nan")))

            for row_idx, (r_intra, r_extra) in enumerate(pairs):
                if r_intra is not None:
                    intra_rec = next(
                        (d for d in r_intra.get("donuts", []) if d["defocal"] == "intra"), None
                    )
                    elapsed_i, nfev_i, success_i, fwhm_i = _rec_info(r_intra)
                    zk_by_noll_i = r_intra.get(
                        "zk_by_noll", {j: float("nan") for j in range(4, _ZK_JMAX + 1)}
                    )
                else:
                    intra_rec = None
                    elapsed_i, nfev_i, success_i, fwhm_i = _rec_info(None)
                    zk_by_noll_i = {j: float("nan") for j in range(4, _ZK_JMAX + 1)}

                if r_extra is not None:
                    extra_rec = next(
                        (d for d in r_extra.get("donuts", []) if d["defocal"] == "extra"), None
                    )
                    elapsed_e, nfev_e, success_e, fwhm_e = _rec_info(r_extra)
                    zk_by_noll_e = r_extra.get(
                        "zk_by_noll", {j: float("nan") for j in range(4, _ZK_JMAX + 1)}
                    )
                else:
                    extra_rec = None
                    elapsed_e, nfev_e, success_e, fwhm_e = _rec_info(None)
                    zk_by_noll_e = {j: float("nan") for j in range(4, _ZK_JMAX + 1)}

                intra_img = intra_rec["img"] if intra_rec else None
                intra_mod = intra_rec["model_img"] if intra_rec else None
                intra_sid = intra_rec["donut_id"] if intra_rec else None
                intra_blend = intra_rec.get("blend_frac", float("nan")) if intra_rec else float("nan")

                extra_img = extra_rec["img"] if extra_rec else None
                extra_mod = extra_rec["model_img"] if extra_rec else None
                extra_sid = extra_rec["donut_id"] if extra_rec else None
                extra_blend = extra_rec.get("blend_frac", float("nan")) if extra_rec else float("nan")

                def _bar_label(elapsed, nfev, success):
                    status = "x0" if nfev == 0 else ("ok" if success else "fail")
                    return f"t={elapsed:.1f}s {status} nfev={nfev}"

                intra_label = _bar_label(elapsed_i, nfev_i, success_i)
                extra_label = _bar_label(elapsed_e, nfev_e, success_e)
                intra_hdr = f"intra {sw1}" if row_idx == 0 else ""
                extra_hdr = f"extra {sw0}" if row_idx == 0 else ""

                def _triplet_and_bar(col_start, data, model, det_hdr, label,
                                     sid, fwhm, zk_by_noll, blend_frac_val=float("nan")):
                    if data is not None:
                        vmax = np.nanpercentile(np.abs(data), 99) or 1.0
                        has_model = model is not None
                        resid = (data - model) if has_model else None
                        vmax_r = (np.nanpercentile(np.abs(resid), 99) or 1.0) if has_model else 1.0
                        for ci, (img, cmap, vmin, vmx) in enumerate([
                            (data, _cmap_bwr, -vmax, vmax),
                            (model if has_model else None, _cmap_bwr, -vmax, vmax),
                            (resid, _cmap_bwr_sym, -vmax_r, vmax_r),
                        ]):
                            ax = fig.add_subplot(inner[row_idx, col_start + ci])
                            lbl = det_hdr if ci == 0 else ""
                            if img is None:
                                ax.axis("off")
                                if lbl:
                                    ax.set_title(lbl, fontsize=5, pad=1)
                                continue
                            _draw_stamp(ax, img, cmap, vmin, vmx, label=lbl)
                            ann_kw = dict(transform=ax.transAxes, fontsize=4, color="k",
                                         va="top", ha="left",
                                         bbox=dict(boxstyle="square,pad=0.1",
                                                   fc="white", ec="none", alpha=0.6))
                            if ci == 0 and sid is not None:
                                ax.text(0.02, 0.98, f"id={sid}", **ann_kw)
                            if ci == 1 and np.isfinite(fwhm):
                                ax.text(0.02, 0.98, f"blur={fwhm:.2f}arcsec", **ann_kw)
                            if ci == 2 and np.isfinite(blend_frac_val):
                                ax.text(0.02, 0.98, f"blend={blend_frac_val:.3f}", **ann_kw)
                        ax_bar = fig.add_subplot(inner[row_idx, col_start + 3])
                        if has_model:
                            _draw_bar(ax_bar, zk_by_noll, inset_label=label)
                        else:
                            ax_bar.axis("off")
                    else:
                        for ci in range(4):
                            ax = fig.add_subplot(inner[row_idx, col_start + ci])
                            ax.axis("off")
                            if ci == 0 and det_hdr:
                                ax.set_title(det_hdr, fontsize=5, pad=1)

                _triplet_and_bar(0, intra_img, intra_mod, intra_hdr, intra_label,
                                 intra_sid, fwhm_i, zk_by_noll_i, intra_blend)
                _triplet_and_bar(4, extra_img, extra_mod, extra_hdr, extra_label,
                                 extra_sid, fwhm_e, zk_by_noll_e, extra_blend)

        proc_total = refcat_elapsed + cutout_elapsed + danish_elapsed
        bt = butler_times or {}
        butler_line = (
            "  ".join(f"{k}={v:.1f}s" for k, v in bt.items() if v > 0.0)
            or f"total={butler_elapsed:.1f}s"
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
