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

"""Deprecated alias for the `lsst.ts.wep.blitz` subpackage.

Everything this module used to define now lives in one module per task under
``lsst.ts.wep.blitz``; import from there instead.  The names are re-exported
here so existing imports -- and stored ``pexConfig`` targets recorded against
this module path -- keep resolving.
"""

__all__ = [
    "DonutBlitzMonolithTaskConnections",
    "DonutBlitzMonolithTaskConfig",
    "DonutBlitzMonolithTask",
    "DonutBlitzPlotTaskConnections",
    "DonutBlitzPlotTaskConfig",
    "DonutBlitzPlotTask",
]

from lsst.ts.wep.blitz.blindDetectTask import (  # noqa: F401
    BlindDetect,
    BlindDetectConfig,
    _buildAnnularTemplate,
)
from lsst.ts.wep.blitz.cutDonutStampsTask import (  # noqa: F401
    CutDonutStampsConfig,
    CutDonutStampsTask,
)
from lsst.ts.wep.blitz.cutoutPipeline import (  # noqa: F401
    _buildAfwSourceCat,
    _cutoutPipeline,
    _run_cutout_worker,
)
from lsst.ts.wep.blitz.dataStructures import (  # noqa: F401
    Donut,
    WfResult,
    _NULL_WF,
    _WfGroup,
)
from lsst.ts.wep.blitz.donutBlitzMonolithTask import (  # noqa: F401
    DonutBlitzMonolithTask,
    DonutBlitzMonolithTaskConfig,
    DonutBlitzMonolithTaskConnections,
)
from lsst.ts.wep.blitz.donutBlitzPlotTask import (  # noqa: F401
    DonutBlitzPlotTask,
    DonutBlitzPlotTaskConfig,
    DonutBlitzPlotTaskConnections,
    _COLOR_APERTURE,
    _COLOR_ASTIGMATISM,
    _COLOR_ASTROM_REFCAT,
    _COLOR_BKG_ANNULUS,
    _COLOR_CMAP_MID,
    _COLOR_CMAP_NEG,
    _COLOR_CMAP_POS,
    _COLOR_COMA,
    _COLOR_HEXAFOIL,
    _COLOR_PENTAFOIL,
    _COLOR_PHOTO_REFCAT,
    _COLOR_QUADRAFOIL,
    _COLOR_REJECTED,
    _COLOR_TREFOIL,
)
from lsst.ts.wep.blitz.measureDonutCandidatesTask import (  # noqa: F401
    MeasureDonutCandidatesConfig,
    MeasureDonutCandidatesTask,
)
from lsst.ts.wep.blitz.utils import (  # noqa: F401
    CORNER_BY_DET_NAME,
    CORNER_DET_NAMES,
    CORNER_PAIRS,
    _ANSI_BLUE,
    _ANSI_BOLD,
    _ANSI_CYAN,
    _ANSI_GREEN,
    _ANSI_MAGENTA,
    _ANSI_RED,
    _ANSI_RESET,
    _ANSI_YELLOW,
    _CALIB_STORE,
    _EXTRA_FOCAL_DET_IDS,
    _INSTRUMENT,
    _INTRA_FOCAL_DET_IDS,
    _MAX_NEARBY,
    _ZK_JMAX,
    _bin_stamp_odd,
    _colorize,
    _resolveColorLogEnabled,
    _resolveDonutRadius,
)
from lsst.ts.wep.blitz.wavefrontFittingTask import (  # noqa: F401
    WavefrontFittingTask,
    WavefrontFittingTaskConfig,
    _DANISH_FIELD_RADIUS_RAD,
    _DZ_MODEL_KEYS,
    _LstsqFitResult,
    _WfFitTimeoutError,
    _bkg_free_model,
    _blend_frac,
    _build_wf_groups,
    _dense_dev,
    _dense_intrinsic,
    _fit_timeout,
    _wf_fitting_worker,
)
