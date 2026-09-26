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

"""`_LSSTCAM` agrees with `Instrument`, and `_rescale_zk_domain` does its job.

Blitz keeps its own LSSTCam constants so that the geometry it assumes is
reviewable in one place, but those constants describe the same telescope
`lsst.ts.wep.instrument.Instrument` does.  Nothing enforces that on its own, so
the comparisons here are **bitwise** (``==``, not `assertAlmostEqual`): a value
that merely rounds to the same number would mean blitz and the rest of ts_wep
had quietly diverged, and the whole point of the holder is that the numbers are
identical and the machinery is not.

`_rescale_zk_domain` is tested for the property that matters -- that it is an
exact no-op when the domains already agree, which is the case for the default
optics model, and that it removes the artifact when they do not.
"""

import copy
import unittest

import batoid
import danish
import numpy as np

from lsst.ts.wep.blitz.dataStructures import _WfGroup
from lsst.ts.wep.blitz.donutBlitzCorner import DonutBlitzCornerConfig
from lsst.ts.wep.blitz.donutBlitzFam import DonutBlitzFamConfig
from lsst.ts.wep.blitz.famPipeline import _RAD_PER_PIXEL
from lsst.ts.wep.blitz.lsstCam import _LSSTCAM, _rescale_zk_domain
from lsst.ts.wep.blitz.utils import (
    _RADIAL_SCALE_REF_THETA,
    _RADIAL_SCALE_WAVELENGTH,
    _defocused_telescope,
)
from lsst.ts.wep.blitz.wavefrontFitting import WavefrontFittingTask
from lsst.ts.wep.instrument import Instrument
from lsst.ts.wep.utils.ioUtils import readConfigYaml

_POLICY_FILE = "policy:instruments/LsstCam.yaml"

# Every band LSSTCam observes in. Spelled out rather than taken from
# `_LSSTCAM.wavelength` so a band silently going missing is a failure.
_BANDS = ("u", "g", "r", "i", "z", "y")


def _radial_scale_at(telescope, offsets, wavelength: float) -> float:
    """`_defocal_radial_scale`, but at a caller-chosen wavelength.

    Reproduced here rather than adding a parameter to the real function, which
    has no use for one: the whole point is that the choice does not matter, so
    production code pins it and only this test varies it.
    """

    def chief_ray_x(optic):
        ray = batoid.RayVector.fromStop(
            0.0,
            0.0,
            optic=optic,
            wavelength=wavelength,
            theta_x=_RADIAL_SCALE_REF_THETA,
            theta_y=0.0,
        )
        optic.trace(ray)
        return float(ray.x[0])

    return chief_ray_x(_defocused_telescope(telescope, offsets)) / chief_ray_x(telescope)


class TestHolderMatchesInstrument(unittest.TestCase):
    """Each `_LSSTCAM` value is bitwise the `Instrument` value it mirrors."""

    def setUp(self) -> None:
        self.instrument = Instrument(configFile=_POLICY_FILE)

    def testScalarsAreBitwiseEqual(self) -> None:
        """The scalars taken from the instrument YAML, plus focal_ratio.

        `focal_ratio` is recomputed from `_LSSTCAM.zk_r_outer` rather than from
        a diameter, so it is the one most likely to catch a formula transcribed
        wrong.
        """
        inst = self.instrument
        for name, holder, instrument in (
            ("zk_r_outer", _LSSTCAM.zk_r_outer, inst.radius),
            ("zk_r_inner", _LSSTCAM.zk_r_inner, inst.radius * inst.obscuration),
            ("obscuration", _LSSTCAM.obscuration, inst.obscuration),
            ("focal_length", _LSSTCAM.focal_length, inst.focalLength),
            ("pixel_size", _LSSTCAM.pixel_size, inst.pixelSize),
            ("focal_ratio", _LSSTCAM.focal_ratio, inst.focalRatio),
        ):
            with self.subTest(constant=name):
                self.assertEqual(
                    float(holder),
                    float(instrument),
                    msg=f"{name} is not bitwise equal to Instrument's value",
                )

    def testWavelengthsAreBitwiseEqualForEveryBand(self) -> None:
        """All six bands, and no extras."""
        self.assertEqual(sorted(_LSSTCAM.wavelength), sorted(_BANDS))
        for band in _BANDS:
            with self.subTest(band=band):
                self.assertEqual(
                    _LSSTCAM.wavelength[band],
                    self.instrument.wavelength[band],
                    msg=f"wavelength[{band!r}] disagrees with Instrument",
                )

    def testWavelengthLookupIsAPlainMapping(self) -> None:
        """``.get`` and ``in`` work on band strings.

        `Instrument.wavelength` is an `EnumDict` that coerces keys in
        ``__getitem__`` but not ``__contains__``, so ``.get("r")`` returns None
        there. Anything reading `_LSSTCAM.wavelength` may use the plain mapping
        protocol without that surprise, and `doAoiThroughput` depends on it.
        """
        self.assertIn("r", _LSSTCAM.wavelength)
        self.assertEqual(_LSSTCAM.wavelength.get("r"), 6.194e-07)
        self.assertIsNone(_LSSTCAM.wavelength.get("not-a-band"))

    def testHolderIsImmutable(self) -> None:
        """`_LSSTCAM` is shared module state, including across forks.

        Every blitz worker reads the same instance, so a caller that could
        assign to it would be editing the telescope for the whole process.
        """
        with self.assertRaises(Exception):
            _LSSTCAM.zk_r_outer = 1.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            _LSSTCAM.wavelength["r"] = 1.0  # type: ignore[index]


class TestDerivedConstants(unittest.TestCase):
    """Module constants computed from the holder, which a rename can break."""

    def setUp(self) -> None:
        self.instrument = Instrument(configFile=_POLICY_FILE)

    def testRadPerPixelIsBitwiseUnchanged(self) -> None:
        """`famPipeline._RAD_PER_PIXEL` expresses the pairing tolerance.

        It converts a tolerance naturally stated in pixels into the field-angle
        space pairing works in, so a drift here would loosen or tighten donut
        pairing rather than announce itself.
        """
        self.assertEqual(
            _RAD_PER_PIXEL,
            self.instrument.pixelSize / self.instrument.focalLength,
        )

    def testRadialScaleWavelengthIsTheReferenceBandValue(self) -> None:
        """`_defocal_radial_scale` traces at the reference band's wavelength.

        The chief-ray trace needs *a* wavelength and the scale barely depends
        on which, so the constant is arbitrary in principle. Pinning it anyway
        keeps the radial correction reproducible rather than merely close.
        """
        self.assertEqual(
            _RADIAL_SCALE_WAVELENGTH,
            self.instrument.wavelength[self.instrument.refBand],
        )
        self.assertIn(_RADIAL_SCALE_WAVELENGTH, set(_LSSTCAM.wavelength.values()))

    def testRadialScaleIsInsensitiveToTheChosenWavelength(self) -> None:
        """Which band the scale is traced at cannot matter at the pixel level.

        This is what makes pinning one wavelength honest rather than a hidden
        approximation: across u to y the scale moves 6.5e-6 relative, while the
        tolerance it feeds is measured in pixels.
        """
        telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        offsets = (_LSSTCAM.defocal_offset, 0.0, 0.0)
        scales = [
            _radial_scale_at(telescope, offsets, wavelength) for wavelength in _LSSTCAM.wavelength.values()
        ]
        self.assertLess((max(scales) - min(scales)) / min(scales), 1e-5)


class TestConfigurableModels(unittest.TestCase):
    """The mask and optics models are independent, resolvable config fields."""

    def testMaskModelDefaultLoadsAndCoversTheFullField(self) -> None:
        """The configured mask must load, and reach the edge of the field.

        `thetaMax` is where a mask element stops being defined, so a model
        cutting off inside the field would silently stop masking the outermost
        donuts -- which are the aberrated ones the fit most depends on.
        `donutBlitzFam` cuts at 1.725 deg, so that is the bar.
        """
        task = WavefrontFittingTask()
        mask_params = task._mask_params
        self.assertEqual(
            sorted(mask_params),
            [
                "CameraBody",
                "Filter_entrance",
                "Filter_exit",
                "L1_entrance",
                "L1_exit",
                "L2_entrance",
                "L2_exit",
                "L3_entrance",
                "L3_exit",
                "M1",
                "M2",
                "M3",
                "Spider_3D",
            ],
        )
        # Spider_3D describes strut geometry rather than an annulus, so it
        # carries no field-angle validity range to check.
        for name, element in mask_params.items():
            if name == "Spider_3D":
                continue
            for side, params in element.items():
                with self.subTest(element=name, side=side):
                    self.assertGreaterEqual(params["thetaMax"], 1.725)

    def testMaskModelIsResolvedOnceNotPerGroup(self) -> None:
        """Held on the task, so forked workers inherit it rather than re-read.

        Full-array mode builds a factory per group and there are thousands, so
        a per-group file read would be paid thousands of times; the parent
        constructs this subtask before it populates the copy-on-write store.
        """
        task = WavefrontFittingTask()
        self.assertIs(task._mask_params, task._mask_params)

    def testLegacyMaskIsSelectableAndUnchanged(self) -> None:
        """The pre-danish mask must stay reproducible, bit for bit.

        It is the only way to reproduce a fit made before the mask default
        moved, so the file in ``policy/masks`` has to be exactly the block that
        shipped in the instrument YAML rather than a re-derivation of it.
        Reached through the same loader as danish's own files, via a
        ``policy:`` prefix.
        """
        config = WavefrontFittingTask.ConfigClass()
        config.maskModel = "policy:masks/LsstCamLegacy.yaml"
        task = WavefrontFittingTask(config=config)

        expected = readConfigYaml(_POLICY_FILE)["maskParams"]
        self.assertEqual(task._mask_params, expected)
        self.assertEqual(
            sorted(task._mask_params),
            ["Filter_entrance", "L1_entrance", "M1", "M2", "M3", "Spider_3D"],
        )
        self.assertEqual(task.resolvedMaskModel(), "LsstCamLegacy.yaml")

    def testAnUnknownMaskModelFailsInTheParent(self) -> None:
        """A typo must not reach the workers.

        The mask is loaded when the subtask is built, which both parent tasks
        do before forking, so a bad name is one exception rather than one per
        worker -- and in full-array mode there are thousands.
        """
        config = WavefrontFittingTask.ConfigClass()
        config.maskModel = "not-a-mask-file.yaml"
        with self.assertRaises(FileNotFoundError):
            WavefrontFittingTask(config=config)

    def testResolvedMaskModelFollowsSymlinks(self) -> None:
        """A generic mask name must record as the file it pointed at.

        ``RubinObsc.yaml`` is a symlink in danish's data directory, so the
        configured name alone does not say which mask a past run used -- the
        link can be repointed. Recording the target is what lets two catalogs
        be compared later.
        """
        config = WavefrontFittingTask.ConfigClass()
        config.maskModel = "RubinObsc.yaml"
        task = WavefrontFittingTask(config=config)
        resolved = task.resolvedMaskModel()
        self.assertNotEqual(resolved, "RubinObsc.yaml")
        self.assertTrue(resolved.startswith("RubinObsc_v"))
        # And the loaded params are the target's, not something else.
        self.assertEqual(task._mask_params, danish.load_mask_params(resolved))

    def testOpticsModelDefaultResolvesForEveryBand(self) -> None:
        """An unexpanded ``{band}`` reaching batoid is the failure to avoid.

        A model name that no code path resolves is exactly what made
        `Instrument.batoidModelName` misleading, so pin that formatting the
        default yields a loadable file in all six bands.
        """
        for config_class in (DonutBlitzCornerConfig, DonutBlitzFamConfig):
            config = config_class()
            self.assertIn("{band}", config.opticsModel)
            for band in _BANDS:
                with self.subTest(config=config_class.__name__, band=band):
                    name = config.opticsModel.format(band=band)
                    self.assertNotIn("{", name)
                    batoid.Optic.fromYaml(f"{name}.yaml")

    def testMaskAndOpticsAreNotCoupled(self) -> None:
        """Setting one must not constrain the other.

        Deliberate: the Zernike domain is fixed in `lsstCam` regardless of the
        optics model, so the pairing is a physics question for whoever runs it
        rather than something config can get wrong on its behalf.
        """
        config = DonutBlitzCornerConfig()
        config.opticsModel = "Rubin_v1000_{band}"
        config.wavefrontFit.maskModel = "RubinObsc_v3.14_r_rtpp0_azp45_pp0d0.yaml"
        config.validate()


class TestAoiThroughput(unittest.TestCase):
    """`doAoiThroughput` reaches danish, and the factory is not rebuilt."""

    def _group(self, band: str = "r", rtp: float = 12.5, alt: float = 1.1):
        return _WfGroup(donuts=[], group_id="g", band=band, rtp=rtp, alt=alt)

    def _task(self, **overrides):
        config = WavefrontFittingTask.ConfigClass()
        for key, value in overrides.items():
            setattr(config, key, value)
        return WavefrontFittingTask(config=config)

    def testBandNameNotWavelengthReachesDanish(self) -> None:
        """danish keys its throughput table by band name.

        It does ``json.load(f)[bandpass_filter]`` against 'u'..'y', so a
        wavelength raises `KeyError` -- the correction is either off or broken,
        never quietly approximate.
        """
        factory = self._task(doAoiThroughput=True)._build_wf_factory(self._group())
        self.assertEqual(factory.bandpass_filter, "r")
        self.assertIsNotNone(factory.thruput_by_aoi)

    def testOffByDefault(self) -> None:
        """The correction is opt-in, so nothing is loaded unless asked."""
        factory = self._task()._build_wf_factory(self._group())
        self.assertIsNone(factory.bandpass_filter)
        self.assertIsNone(factory.thruput_by_aoi)

    def testAnUnsupportedBandWarnsRatherThanRaising(self) -> None:
        """A band danish has no table for must not abort the fit.

        The wavefront is still fittable without the throughput correction, so
        the cost of continuing is a slightly wrong forward model where the cost
        of raising is the whole visit.
        """
        task = self._task(doAoiThroughput=True)
        with self.assertLogs(task.log.name, level="WARNING") as captured:
            factory = task._build_wf_factory(self._group(band="q"))
        self.assertIsNone(factory.bandpass_filter)
        self.assertIn("not supported", "".join(captured.output))

    def testFactoryIsReusedAcrossGroupsOfOneVisit(self) -> None:
        """Band, rotator angle and altitude are per-visit, not per-group.

        Building a factory with AOI on parses a 1 MB throughput table that
        danish does not cache, so rebuilding it per group would cost thousands
        of parses in full-array mode for thousands of identical objects.
        """
        task = self._task(doAoiThroughput=True)
        factories = [task._build_wf_factory(self._group()) for _ in range(50)]
        self.assertEqual(len(task._factory_cache), 1)
        for factory in factories:
            self.assertIs(factory, factories[0])

    def testCacheKeyCoversEveryGroupInputTheFactoryUses(self) -> None:
        """Ignoring any one of these would silently mis-model a group."""
        task = self._task(doAoiThroughput=True)
        base = task._build_wf_factory(self._group())
        for label, group in (
            ("band", self._group(band="i")),
            ("rtp", self._group(rtp=-30.0)),
            ("alt", self._group(alt=0.8)),
        ):
            with self.subTest(varying=label):
                self.assertIsNot(task._build_wf_factory(group), base)
        # The spider angle is the one that reaches the rendered image directly.
        self.assertEqual(task._build_wf_factory(self._group(rtp=-30.0)).spider_angle, -30.0)


class TestRescaleZkDomain(unittest.TestCase):
    """The shim is an exact no-op by default and fixes a real mismatch."""

    def setUp(self) -> None:
        self.r_outer = _LSSTCAM.zk_r_outer
        self.r_inner = _LSSTCAM.zk_r_inner
        rng = np.random.default_rng(0)
        # Noll-indexed, index 0 meaningless, a defocus-dominated spectrum like
        # a real defocused zk_ref rather than white noise.
        self.coef = np.zeros(67)
        self.coef[4:] = rng.normal(scale=1e-7, size=63)
        self.coef[4] = 2.4e-05

    def _zk_ref(self, telescope: batoid.Optic) -> np.ndarray:
        """A reference Zernike array as `_prep_donut_for_danish` builds it."""
        eps = telescope.pupilObscuration
        nrad = 10
        wavelength = _LSSTCAM.wavelength["r"]
        return (
            batoid.zernikeTA(
                telescope,
                np.deg2rad(1.19),
                0.0,
                wavelength,
                jmax=66,
                eps=eps,
                focal_length=_LSSTCAM.focal_length,
                nrad=nrad,
                naz=int(2 * np.pi * nrad / (1 - eps)),
            )
            * wavelength
        )

    def testIdentityWhenDomainsAgree(self) -> None:
        """Equal source and target domains must not perturb the coefficients.

        This is the proof that applying the shim unconditionally is safe: with
        the default optics model the domains do agree, so every number blitz
        fits is untouched apart from the xy round trip's ~1e-17 m.
        """
        out = _rescale_zk_domain(
            self.coef,
            r_outer_from=self.r_outer,
            r_inner_from=self.r_inner,
            r_outer_to=self.r_outer,
            r_inner_to=self.r_inner,
        )
        self.assertEqual(len(out), len(self.coef))
        np.testing.assert_allclose(out, self.coef, atol=1e-15, rtol=0.0)

    def testDefaultOpticsModelIsTheIdentityCase(self) -> None:
        """`LSST_r.yaml`'s pupil is bitwise the domain blitz normalizes on.

        Without this, `testIdentityWhenDomainsAgree` would prove a property of
        the function that no call site actually exercises.
        """
        telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        r_outer = telescope.pupilSize / 2
        self.assertEqual(r_outer, self.r_outer)
        self.assertEqual(telescope.pupilObscuration * r_outer, self.r_inner)

    def testRemovesTheArtifactOfAMismatchedPupil(self) -> None:
        """A pupil-only change must not move the wavefront blitz reports.

        Forcing `pupilSize` on an otherwise identical telescope changes the
        domain `zernikeTA` fits on and nothing physical, so the honest answer
        is that the coefficients are unchanged. Raw, defocus moves ~183 nm;
        after rescaling, every aberration term agrees to well under a nm.

        Piston is excluded because it carries the domain change legitimately:
        an annular basis on one domain has a non-zero mean over another, and
        `batoid.zernikeTA` zeroes index 0 as meaningless regardless.
        """
        telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        defocused = telescope.withGloballyShiftedOptic("Detector", [0.0, 0.0, _LSSTCAM.defocal_offset])
        expected = self._zk_ref(defocused)

        # A shallow copy is enough and is the point: the surfaces are shared,
        # so only the declared pupil differs and nothing optical does.
        mismatched = copy.copy(defocused)
        mismatched.pupilSize = 8.33
        raw = self._zk_ref(mismatched)

        r_outer_from = mismatched.pupilSize / 2
        fixed = _rescale_zk_domain(
            raw,
            r_outer_from=r_outer_from,
            r_inner_from=mismatched.pupilObscuration * r_outer_from,
            r_outer_to=self.r_outer,
            r_inner_to=self.r_inner,
        )

        # The bug is real and large before the fix...
        self.assertGreater(np.abs(raw[4] - expected[4]), 150e-9)
        # ...and gone after it, across every aberration term.
        np.testing.assert_allclose(fixed[4:], expected[4:], atol=1e-9, rtol=0.0)

    def testInnerArgumentsAreRadiiNotFractions(self) -> None:
        """Passing an obscuration fraction where a radius belongs must matter.

        galsim's ``R_inner`` is a radius while batoid's ``eps`` is a fraction
        of the pupil radius, so the conversion is the caller's. If 0.612 and
        2.55816 gave the same answer, every call site would be one silent
        ~192 nm error away from correct and no test would notice.
        """
        as_radius = _rescale_zk_domain(
            self.coef,
            r_outer_from=self.r_outer,
            r_inner_from=self.r_inner,
            r_outer_to=self.r_outer,
            r_inner_to=self.r_inner,
        )
        as_fraction = _rescale_zk_domain(
            self.coef,
            r_outer_from=self.r_outer,
            r_inner_from=_LSSTCAM.obscuration,
            r_outer_to=self.r_outer,
            r_inner_to=self.r_inner,
        )
        self.assertGreater(np.max(np.abs(as_fraction - as_radius)), 1e-9)

    def testDomainsAreKeywordOnly(self) -> None:
        """Four interchangeable floats; a swap is silent and ~183 nm large."""
        with self.assertRaises(TypeError):
            _rescale_zk_domain(  # type: ignore[misc]
                self.coef, self.r_outer, self.r_inner, self.r_outer, self.r_inner
            )


if __name__ == "__main__":
    unittest.main()
