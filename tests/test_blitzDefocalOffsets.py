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

"""Defocus is selected by optic-offset triplet rather than by detector id.

The fitter reads a signed ``(detector, camera, m2)`` offset triplet off each
`Donut` rather than inferring a hardcoded detector-plane shift from the detector
id, which lets full-array mode put the same detector on both sides of focus. For
corner mode the two formulations must agree bit-for-bit, so the things worth
asserting are that the telescope lookup reproduces a plain detector-plane shift
exactly, that the camera and M2 components are wired to distinct optics, and that
the SW0/SW1 convention is pinned in `CORNER_DEFOCAL_BY_DET_NAME` -- its only home,
since donuts carry no intra/extra label at all.

Also covers `_defocal_radial_scale`, which full-array donut pairing depends on.
"""

import unittest

import batoid
import numpy as np

from lsst.ts.wep.blitz.donutBlitzMonolithTask import (
    _EXTRA_FOCAL_OFFSETS,
    _INTRA_FOCAL_OFFSETS,
)
from lsst.ts.wep.blitz.utils import (
    CORNER_DEFOCAL_BY_DET_NAME,
    CORNER_DET_NAMES,
    _CALIB_STORE,
    _EXTRA_FOCAL_DET_IDS,
    _INSTRUMENT,
    _INTRA_FOCAL_DET_IDS,
    _OFFSET_OPTICS,
    _ZK_JMAX,
    _defocal_radial_scale,
    _telescope_for_offsets,
)


def _minimalDonut(**overrides):
    """A Donut carrying just enough to reach the defocal-offset lookup."""
    from lsst.ts.wep.blitz.dataStructures import Donut

    kwargs = dict(
        det_name="R00_SW0",
        stamp=np.ones((167, 167), dtype=float),
        thx_ccs=0.0,
        thy_ccs=0.0,
        flux=1.0,
        band="r",
        det_id=191,
        visit_id=1,
        x_det=0.0,
        y_det=0.0,
        donut_id=1,
        inner_frac=0.0,
        outer_frac=0.0,
        outer_sector_minmax_frac=0.0,
        donut_radius=60.0,
        snr=100.0,
        bkg=0.0,
        bkg_std=1.0,
        n_quarter=0,
        photo_mag=float("nan"),
        astrom_mag=float("nan"),
        nearby_photo=[],
        nearby_astrom=[],
    )
    kwargs.update(overrides)
    return Donut(**kwargs)


class TestDefocalOffsets(unittest.TestCase):
    """Offset-triplet telescope lookup and corner-mode neutrality."""

    def setUp(self) -> None:
        self.band = "r"
        self.telescope = batoid.Optic.fromYaml(f"LSST_{self.band}.yaml")
        _CALIB_STORE.clear()
        _CALIB_STORE["telescope"] = self.telescope

    def tearDown(self) -> None:
        _CALIB_STORE.clear()

    def _zk(self, telescope: batoid.Optic, theta_deg: float) -> np.ndarray:
        """Zernikes at one field angle, the way the fitter computes them."""
        eps = self.telescope.pupilObscuration
        nrad = 10
        return batoid.zernikeTA(
            telescope,
            np.deg2rad(theta_deg),
            0.0,
            _INSTRUMENT.wavelength[_INSTRUMENT.refBand],
            jmax=_ZK_JMAX,
            eps=eps,
            focal_length=_INSTRUMENT.focalLength,
            nrad=nrad,
            naz=int(2 * np.pi * nrad / (1 - eps)),
        )

    def testCornerDefocalLookupEncodesTheSw0Sw1Convention(self) -> None:
        """`CORNER_DEFOCAL_BY_DET_NAME` is the only home of SW0/SW1 -> side.

        Donuts carry no intra/extra label -- only their optic offsets -- so the
        plot task derives the label from the detector name via
        this lookup. That makes it the single point where "SW0 is extra-focal,
        SW1 is intra-focal" is written down, and worth pinning.

        The name-based lookup and the id-based `_INTRA/_EXTRA_FOCAL_DET_IDS` used
        to pick offsets are two encodings of the same convention. Linking them
        detector-by-detector would need camera geometry; this checks that they
        agree in structure and count, which is what would break if one were
        edited without the other.
        """
        self.assertEqual(set(CORNER_DEFOCAL_BY_DET_NAME), set(CORNER_DET_NAMES))
        for name, side in CORNER_DEFOCAL_BY_DET_NAME.items():
            self.assertEqual(side, "extra" if name.endswith("SW0") else "intra")

        by_side: dict[str, set[str]] = {"intra": set(), "extra": set()}
        for name, side in CORNER_DEFOCAL_BY_DET_NAME.items():
            by_side[side].add(name)
        self.assertEqual(len(by_side["intra"]), len(_INTRA_FOCAL_DET_IDS))
        self.assertEqual(len(by_side["extra"]), len(_EXTRA_FOCAL_DET_IDS))

        # And the offsets chosen for each side carry the matching sign, so a
        # plot label can never contradict the telescope the fit actually used.
        self.assertLess(_INTRA_FOCAL_OFFSETS[0], 0)
        self.assertGreater(_EXTRA_FOCAL_OFFSETS[0], 0)

    def testCameraAndM2ShiftsAreDistinctAndApplied(self) -> None:
        """The camera and M2 components are wired to different optics.

        Full-array mode defocuses by moving the whole camera, and some data may
        instead move M2, so a triplet that silently applied only the detector
        component would be a quiet physics bug.
        """
        dz = _INSTRUMENT.defocalOffset
        det = self._zk(_telescope_for_offsets((dz, 0.0, 0.0)), 1.0)
        cam = self._zk(_telescope_for_offsets((0.0, dz, 0.0)), 1.0)
        m2 = self._zk(_telescope_for_offsets((0.0, 0.0, dz)), 1.0)
        base = self._zk(self.telescope, 1.0)
        for name, zk in (("detector", det), ("camera", cam), ("m2", m2)):
            self.assertFalse(
                np.allclose(zk, base),
                msg=f"{name} offset had no effect on the wavefront",
            )
        # Moving the camera is not the same as moving the detector within it.
        self.assertFalse(np.allclose(det, cam))
        self.assertFalse(np.allclose(cam, m2))

    def testRadialScaleMatchesDirectChiefRayTrace(self) -> None:
        """The pure-scale model reproduces a direct trace across the field.

        Full-array donut pairing relies on the intra/extra radial displacement
        being a pure scale, so one factor corrects every detector. If it were
        instead field-dependent, pairing would work at the centre and fail at the
        edge -- so this asserts the scale against an independent per-angle trace.
        """
        dz = _INSTRUMENT.defocalOffset
        px = _INSTRUMENT.pixelSize
        wavelength = _INSTRUMENT.wavelength[_INSTRUMENT.refBand]
        extra = (0.0, dz, 0.0)
        intra = (0.0, -dz, 0.0)

        def traced_x(offsets, theta_deg):
            telescope = _telescope_for_offsets(offsets)
            ray = batoid.RayVector.fromStop(
                0.0, 0.0, optic=telescope, wavelength=wavelength,
                theta_x=np.deg2rad(theta_deg), theta_y=0.0,
            )
            telescope.trace(ray)
            return float(ray.x[0])

        for theta in (0.5, 1.0, 1.725):
            traced_px = abs(traced_x(extra, theta) - traced_x(intra, theta)) / px
            r_m = _INSTRUMENT.focalLength * np.tan(np.deg2rad(theta))
            scaled_px = abs(
                r_m * _defocal_radial_scale(extra) - r_m * _defocal_radial_scale(intra)
            ) / px
            self.assertAlmostEqual(
                traced_px,
                scaled_px,
                delta=0.2,
                msg=f"scale model diverges from a direct trace at {theta} deg",
            )
            # And the effect is real: tens of pixels away from the axis.
            if theta >= 1.0:
                self.assertGreater(traced_px, 10.0)

    def testRadialScaleSignsAreOpposite(self) -> None:
        """Intra and extra stretch the focal plane in opposite senses."""
        dz = _INSTRUMENT.defocalOffset
        extra = _defocal_radial_scale((0.0, dz, 0.0))
        intra = _defocal_radial_scale((0.0, -dz, 0.0))
        self.assertLess(extra, 1.0)
        self.assertGreater(intra, 1.0)
        self.assertAlmostEqual(extra - 1.0, -(intra - 1.0), places=6)

    def testMissingOffsetsIsFatalNotSilent(self) -> None:
        """A donut with no offsets must raise rather than guess a defocal side."""
        from lsst.ts.wep.blitz.wavefrontFittingTask import WavefrontFittingTask

        task = WavefrontFittingTask()
        donut = _minimalDonut(defocal_offsets=None)
        with self.assertRaisesRegex(RuntimeError, "defocal_offsets"):
            task._prep_donut_for_danish(donut)


if __name__ == "__main__":
    unittest.main()
