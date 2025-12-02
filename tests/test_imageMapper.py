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

import itertools
import unittest
from typing import Any

import batoid
import numpy as np
from lsst.ts.wep import Image, ImageMapper, Instrument
from scipy.ndimage import binary_erosion, binary_opening, shift


class TestImageMapper(unittest.TestCase):
    """Test the ImageMapper class."""

    @staticmethod
    def _createImageWithBatoid(
        nPixels: int, fieldAngle: float, defocalType: str, instrument: Instrument
    ) -> np.ndarray:
        """Create a simple image with Batoid.

        Note the resulting image has aliasing artifacts.

        Parameters
        ----------
        nPixels : int
            The number of pixels on a side
        fieldAngle : tuple
            The field angle of the source in degrees
        defocalType : str
            The defocal type, either "intra" or "extra"
        instrument : Instrument
            The Instrument object

        Returns
        -------
        np.ndarray
            The Batoid image array
        """
        # Create a dense pupil grid
        lim = 1.05 * instrument.radius
        grid = np.linspace(-lim, lim, 1000)
        x, y = np.meshgrid(grid, grid)

        # Get the Batoid model from the instrument
        defocalSign = +1 if defocalType == "extra" else -1
        offset = defocalSign * instrument.defocalOffset
        model = instrument.getBatoidModel()
        model = model.withLocallyShiftedOptic("Detector", [0, 0, offset])

        # We need to get the image position of the chief ray
        rays = batoid.RayVector.fromStop(
            x=0,
            y=0,
            optic=model,
            wavelength=instrument.wavelength["ref"],
            theta_x=np.deg2rad(fieldAngle)[0],
            theta_y=np.deg2rad(fieldAngle)[1],
        )
        model.trace(rays)
        x0 = rays.x
        y0 = rays.y

        # Now map the pupil grid onto the image
        rays = batoid.RayVector.fromStop(
            x=x.flatten(),
            y=y.flatten(),
            optic=model,
            wavelength=instrument.wavelength["ref"],
            theta_x=np.deg2rad(fieldAngle)[0],
            theta_y=np.deg2rad(fieldAngle)[1],
        )
        model.trace(rays)

        # Now we need to bin the unvignetted rays in the image grid
        # Get the image grid from the instrument and convert to meters
        u, _ = instrument.createImageGrid(nPixels)
        x = u * instrument.donutRadius * instrument.pixelSize

        # Get the bin edges for these pixels
        dx = np.diff(x[0])[0]
        xEdges = np.append(x[0] - dx / 2, x[0, -1] + dx / 2)

        # Bin the centered Batoid rays
        batoidImage, *_ = np.histogram2d(
            rays.x[~rays.vignetted] - x0,
            rays.y[~rays.vignetted] - y0,
            bins=xEdges,
        )

        return batoidImage

    def testCreateWithDefaults(self) -> None:
        ImageMapper()

    def testBadOpticalModel(self) -> None:
        with self.assertRaises(TypeError):
            ImageMapper(opticalModel=1)
        with self.assertRaises(ValueError):
            ImageMapper(opticalModel="bad")

    def testCreatePupilMaskCenter(self) -> None:
        # Create the mapper and pull out the instrument
        mapper = ImageMapper()
        inst = mapper.instrument

        # Centered pupil mask should just be disk with central obscuration
        u, v = inst.createPupilGrid()
        r = np.sqrt(u**2 + v**2)
        truth = (r >= inst.obscuration) & (r <= 1)

        # Check that the binary mask matches the expected truth
        image = Image(np.zeros_like(r), (0, 0), "intra", "ref", "pupil")
        mapper.createPupilMasks(image, isBinary=True)
        self.assertTrue(np.allclose(image.mask, truth))

    def testCreatePupilMaskOffCenter(self) -> None:
        # Create the mapper and pull out the instrument
        mapper = ImageMapper()
        inst = mapper.instrument

        # Get the pupil grid
        u, v = inst.createPupilGrid()

        # Convert units to meters
        x = inst.radius * u
        y = inst.radius * v

        # Set field angle of the source
        fieldAngle = np.array([1.20, 1.27])  # degrees

        # Get the Batoid model from the instrument
        optic = inst.getBatoidModel()

        # Create a grid of rays that intersect the pupil
        rays = batoid.RayVector.fromStop(
            x=x.flatten(),
            y=y.flatten(),
            optic=optic,
            wavelength=inst.wavelength["ref"],
            theta_x=np.deg2rad(fieldAngle)[0],
            theta_y=np.deg2rad(fieldAngle)[1],
        )

        # Trace the rays through the model
        optic.trace(rays)

        # Ask which rays were vignetted
        maskPupilBatoid = ~rays.vignetted.reshape(u.shape)

        # Now use the ts_wep model to get the mask
        image = Image(
            np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
            fieldAngle,
            "intra",
            "ref",
            "pupil",
        )
        mapper.createPupilMasks(image, isBinary=True)

        # Get the difference in the masks
        diff = maskPupilBatoid.astype(float) - image.mask.astype(float)

        # Apply binary opening once to remove small artifacts at edges of masks
        diff = binary_opening(diff)

        self.assertTrue(np.allclose(diff, 0))

    def testCreateImageMaskOffCenter(self) -> None:
        for defocalType in ["intra", "extra"]:
            # Create the mapper and pull out the instrument
            mapper = ImageMapper()
            inst = mapper.instrument

            # Set field angle of the source
            fieldAngle = np.array([1.20, 1.27])  # degrees

            # First, let's get the model image mask
            image = Image(
                image=np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
                fieldAngle=fieldAngle,
                defocalType=defocalType,
            )
            mapper.createImageMasks(image)

            # Now get the Batoid mask
            batoidImage = self._createImageWithBatoid(
                len(image.image),
                fieldAngle,
                defocalType,
                inst,
            )
            maskImageBatoid = batoidImage > 0

            # Get the difference in the masks
            diff = maskImageBatoid.astype(float) - image.mask.astype(float)

            # Binary opening to remove small artifacts at edges of masks
            diff = binary_opening(diff)

            # Calculate the fractional difference
            fracDiff = np.abs(diff).sum() / image.mask.sum()

            self.assertLess(fracDiff, 0.01)

    def testCenterOnProjection(self) -> None:
        # Forward model an image
        mapper = ImageMapper()
        inst = mapper.instrument
        image = Image(
            np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
            (0, 0),
            "intra",
        )
        zk = np.random.default_rng(0).normal(scale=50e-9, size=19)
        image = mapper.mapPupilToImage(image, zk)

        # Decenter the image
        decentered = image.copy()
        decentered.image = np.roll(image.image, (-10, 12), (0, 1))

        # Recenter using the binary template
        recentered = mapper.centerOnProjection(
            decentered,
            zk,
            isBinary=True,
            rMax=np.inf,
        )
        self.assertTrue(np.allclose(recentered.image, image.image))

        # Recenter using the full template
        recentered = mapper.centerOnProjection(
            decentered,
            zk,
            isBinary=False,
            rMax=np.inf,
        )
        self.assertTrue(np.allclose(recentered.image, image.image))

    def testMapPupilToImage(self) -> None:
        for defocalType in ["intra", "extra"]:
            # Create the mapper and pull out the instrument
            mapper = ImageMapper()
            inst = mapper.instrument

            # Set field angle of the source
            fieldAngle = np.array([1.20, 1.27])  # degrees

            # First, let's map the pupil to the image plane
            image = mapper.mapPupilToImage(
                Image(
                    image=np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
                    fieldAngle=fieldAngle,
                    defocalType=defocalType,
                )
            )

            # Now let's simulate the Batoid image
            batoidImage = self._createImageWithBatoid(
                len(image.image),
                fieldAngle,
                defocalType,
                inst,
            )
            batoidImage *= image.image.sum() / batoidImage.sum()

            # Make the Batoid mask
            batoidImageMask = batoidImage > 0

            # Calculate the difference
            diff = image.image - batoidImage

            # Calculate the absolute mean difference
            absMeanDiff = np.abs(diff[batoidImageMask].mean())

            self.assertLess(absMeanDiff, 0.01)

    def testRoundTrip(self) -> None:
        rng = np.random.default_rng(0)

        for opticalModel, fieldAngle, defocalType, zk in itertools.product(
            ["paraxial", "onAxis", "offAxis"],
            [(0, 0), (1.2, -0.7)],
            ["intra", "extra"],
            [np.zeros(1), rng.normal(scale=50e-9, size=19)],
        ):
            # Create the Image mapper
            mapper = ImageMapper(opticalModel=opticalModel)
            inst = mapper.instrument

            # Forward model an image
            image = Image(
                np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
                fieldAngle,
                defocalType,
            )
            image = mapper.mapPupilToImage(image, zk)

            # Map image back to the pupil
            pupilRecon = mapper.mapImageToPupil(image, zk)

            # Create the pupil mask
            mapper.createPupilMasks(pupilRecon)
            pupil = pupilRecon.mask

            # Calculate the difference between the pupil
            # and the reconstructed pupil
            diff = pupilRecon.image - pupil

            self.assertLess(diff.sum() / pupil.sum(), 0.02)
            self.assertLess(diff.max(), 1)

    def testMaskBlends(self) -> None:
        # Create the image mapper
        mapper = ImageMapper()
        inst = mapper.instrument

        # Create a dummy image
        image = Image(
            np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
            (0, 0),
            "intra",
        )

        # Replace the image with the image model
        image.image = mapper.mapPupilToImage(image).image

        # Test that a blend offset of 0 removes all the flux
        image.blendOffsets = [[0, 0]]
        mapper.createImageMasks(image, doMaskBlends=True)
        self.assertTrue(np.allclose(image.mask, 0))
        mapper.createPupilMasks(image, doMaskBlends=True, ignorePlane=True)
        self.assertTrue(np.allclose(image.mask, 0))
        self.assertTrue(np.allclose(mapper.mapImageToPupil(image, doMaskBlends=True).image, 0))
        self.assertTrue(np.allclose(mapper.mapPupilToImage(image, doMaskBlends=True).image, 0))

        # Non-zero blend offset removes a portion of the flux
        image.blendOffsets = [[50, 50]]
        mapper.createImageMasks(image, doMaskBlends=False)
        mask0 = image.mask
        mapper.createImageMasks(image, doMaskBlends=True)
        mask1 = image.mask
        self.assertTrue(0 < mask1.sum() < mask0.sum())

        mapper.createPupilMasks(image, doMaskBlends=False, ignorePlane=True)
        mask0 = image.mask
        mapper.createPupilMasks(image, doMaskBlends=True, ignorePlane=True)
        mask1 = image.mask
        self.assertTrue(0 < mask1.sum() < mask0.sum())

        self.assertTrue(
            0
            < mapper.mapImageToPupil(image, doMaskBlends=True).image.sum()
            < mapper.mapPupilToImage(image, doMaskBlends=False).image.sum()
        )

    def testDilate(self) -> None:
        # Create the image mapper
        mapper = ImageMapper()
        inst = mapper.instrument

        # Create a dummy image
        image = Image(
            np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
            (0, 0),
            "intra",
        )

        # Test error with negative dilate
        with self.assertRaises(ValueError):
            mapper.createImageMasks(image, dilate=-1)
        with self.assertRaises(ValueError):
            mapper.createPupilMasks(image, dilate=-1, ignorePlane=True)

        # Test that you can only dilate a binary mask
        with self.assertRaises(ValueError):
            mapper.createImageMasks(image, isBinary=False, dilate=1)
        with self.assertRaises(ValueError):
            mapper.createPupilMasks(image, isBinary=False, dilate=1, ignorePlane=True)

        # Test that the dilated mask is bigger
        mapper.createImageMasks(image, isBinary=True)
        mask0 = image.mask
        mapper.createImageMasks(image, isBinary=True, dilate=1)
        mask1 = image.mask
        self.assertGreater(
            mask1.sum(),
            mask0.sum(),
        )

        mapper.createPupilMasks(image, isBinary=True, ignorePlane=True)
        mask0 = image.mask
        mapper.createPupilMasks(image, isBinary=True, dilate=1, ignorePlane=True)
        mask1 = image.mask
        self.assertGreater(
            mask1.sum(),
            mask0.sum(),
        )

    def testDilateBlends(self) -> None:
        # Create the image mapper
        mapper = ImageMapper()
        inst = mapper.instrument

        # Create a dummy image
        image = Image(
            np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
            (0, 0),
            "intra",
            blendOffsets=[[-20, 30]],
        )

        # Test error with negative dilate
        with self.assertRaises(ValueError):
            mapper.createImageMasks(image, dilateBlends=-1)
        with self.assertRaises(ValueError):
            mapper.createPupilMasks(image, dilateBlends=-1, ignorePlane=True)

        # Test that you CAN dilate blends for a fractional mask
        mapper.createImageMasks(image, isBinary=False, dilateBlends=1, doMaskBlends=True)
        mapper.createPupilMasks(
            image,
            isBinary=False,
            dilateBlends=1,
            doMaskBlends=True,
            ignorePlane=True,
        )

        # Test that the mask with dilated blends is smaller
        mapper.createImageMasks(image, doMaskBlends=True)
        mask0 = image.mask
        mapper.createImageMasks(image, doMaskBlends=True, dilateBlends=1)
        mask1 = image.mask
        self.assertLess(
            mask1.sum(),
            mask0.sum(),
        )

        mapper.createPupilMasks(image, doMaskBlends=True, ignorePlane=True)
        mask0 = image.mask
        mapper.createPupilMasks(
            image,
            doMaskBlends=True,
            dilateBlends=1,
            ignorePlane=True,
        )
        mask1 = image.mask
        self.assertLess(
            mask1.sum(),
            mask0.sum(),
        )

    def testCreateBlendMask(self) -> None:
        mapper = ImageMapper()
        inst = mapper.instrument

        def _testCreateBlendMask(isIntra: bool, isPupilMask: bool) -> None:
            try:
                # Create a dummy image
                image = Image(
                    np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
                    (0, 0),
                    "intra" if isIntra else "extra",
                    planeType="pupil" if isPupilMask else "image",
                )

                # Create masks
                if isPupilMask:
                    mapper.createPupilMasks(image)
                else:
                    mapper.createImageMasks(image)
                mask0 = image.mask

                # Test that blend mask is same as source mask before shifting
                image.blendOffsets = [[0, 0]]
                blendMask = mapper._createBlendMask(
                    image,
                    mask0,
                    dilateBlends=0,
                    isPupilMask=isPupilMask,
                )
                np.testing.assert_array_almost_equal(mask0, blendMask)

                # Test that blend mask is shifted by specified amount
                image.blendOffsets = [[2, 0]]
                blendMask = mapper._createBlendMask(
                    image,
                    mask0,
                    dilateBlends=0,
                    isPupilMask=isPupilMask,
                )
                shiftSign = -1 if (isPupilMask and not isIntra) else +1
                mask0shifted = shift(mask0, (0, shiftSign * 2))
                np.testing.assert_array_almost_equal(mask0shifted, blendMask)

                # Test that dilation increases mask size
                image.blendOffsets = [[0, 0]]
                blendMask = mapper._createBlendMask(
                    image,
                    mask0,
                    dilateBlends=1,
                    isPupilMask=isPupilMask,
                )
                self.assertGreater(blendMask.sum(), mask0.sum())
            except AssertionError:
                raise AssertionError(f"Failed on test of isIntra={isIntra}, isPupilMask={isPupilMask}")

        # Test image masks
        _testCreateBlendMask(isIntra=True, isPupilMask=False)
        _testCreateBlendMask(isIntra=False, isPupilMask=False)

        # Test pupil masks
        _testCreateBlendMask(isIntra=True, isPupilMask=True)
        _testCreateBlendMask(isIntra=False, isPupilMask=True)

    def testAutoDilateBlendMask(self) -> None:
        mapper = ImageMapper()
        inst = mapper.instrument

        def _testAutoDilate(isIntra: bool, isPupilMask: bool) -> None:
            # Create a dummy image
            image = Image(
                np.zeros((inst.nPupilPixels, inst.nPupilPixels)),
                (0, 0),
                "intra" if isIntra else "extra",
                blendOffsets=[[110, 0]],
            )

            # Get the source mask in order to simulate blended image
            mapper.createImageMasks(image, doMaskBlends=False)
            mask0 = image.mask
            image.image = np.array(mask0.copy(), dtype=np.float64)
            image.image += shift(image.image, image.blendOffsets[0, ::-1])

            # Add some noise
            rng = np.random.default_rng(123)
            image.image += rng.normal(scale=0.1, size=np.shape(image.image))

            if not isIntra and isPupilMask:
                # For extrafocal, the pupil and image are 180 deg rotated
                image.image = np.rot90(image.image, 2)

            # Erode mask and check auto-dilation fixes this erosion
            erodedMask = np.array(binary_erosion(mask0, iterations=3), dtype=np.int64)
            dilatedMask = mapper._autoDilateBlendMask(
                image=image,
                mask=erodedMask,
                sourceMask=mask0,
                autoDilateMaxIter=8,
                autoDilateFracChange=5e-3,
                isPupilMask=isPupilMask,
            )
            blendMask = mapper._createBlendMask(
                image=image,
                mask=mask0,
                dilateBlends=0,
                isPupilMask=isPupilMask,
            )
            try:
                np.testing.assert_array_almost_equal(blendMask, dilatedMask)
            except AssertionError:
                raise AssertionError(f"Failed on test of isIntra={isIntra}, isPupilMask={isPupilMask}")

        # Test image masks
        _testAutoDilate(isIntra=True, isPupilMask=False)
        _testAutoDilate(isIntra=False, isPupilMask=False)

        # Test pupil masks
        _testAutoDilate(isIntra=True, isPupilMask=True)
        _testAutoDilate(isIntra=False, isPupilMask=True)

    def testGetProjectionSize(self) -> None:
        mapper = ImageMapper()

        # Check against tested values
        self.assertEqual(mapper.getProjectionSize((0, 0), "intra"), 135)
        self.assertEqual(mapper.getProjectionSize((0, 0), "extra"), 136)
        self.assertEqual(mapper.getProjectionSize((1.2, 0.3), "intra"), 143)
        self.assertEqual(mapper.getProjectionSize((1.2, 0.3), "extra"), 146)

    def testBatoidRaytraceResiduals(self) -> None:
        # Define sets of configurations to test
        instruments = [
            "policy:instruments/LsstCam.yaml",
            "policy:instruments/AuxTel.yaml",
        ]
        fieldAngles = [
            (0, 0),
            (-0.005, 0.002),
            (0.015, 0),
            (-0.11, -0.23),
            (0.5, 0.01),
            (0.0, -0.75),
            (0.9, 0.9),
            (-1.1, 1.0),
            (-1.2, -1.2),
        ]
        opticalModels = ["paraxial", "onAxis", "offAxis"]
        defocalTypes = ["intra", "extra"]

        # Function that maps config to required precision (% of pixel size)
        def maxPercent(**kwargs: Any) -> float:
            if "Lsst" in instConfig and model == "onAxis":
                return 25
            else:
                return 10

        # Loop over every combo
        for instConfig, angle, model, dfType in itertools.product(
            instruments, fieldAngles, opticalModels, defocalTypes
        ):
            # Skip combos we don't expect to work
            if "Lsst" in instConfig and model == "paraxial":
                continue
            elif "AuxTel" in instConfig and np.hypot(*angle) > 0.07:
                continue
            elif model != "offAxis" and np.hypot(*angle) > 0.015:
                continue

            # Create the image mapper
            mapper = ImageMapper(instConfig=instConfig, opticalModel=model)
            inst = mapper.instrument

            # Determine the defocal offset
            offset = -inst.defocalOffset if dfType == "intra" else inst.defocalOffset

            # Loop over each band
            for band in inst.wavelength:
                # Get the Batoid model
                optic = inst.getBatoidModel(band)

                # Create the RayVector
                dirCos = batoid.utils.fieldToDirCos(*np.deg2rad(angle))
                rays = batoid.RayVector.asPolar(
                    optic=optic,
                    wavelength=inst.wavelength[band],
                    dirCos=dirCos,
                    nrad=5,
                    naz=30,
                )

                # Get normalized pupil coordinates
                pupilRays = optic.stopSurface.interact(rays.copy())
                uPupil = pupilRays.x / inst.radius
                vPupil = pupilRays.y / inst.radius

                # Map to focal plane using the offAxis model
                uImage, vImage, *_ = mapper._constructForwardMap(
                    uPupil,
                    vPupil,
                    inst.getIntrinsicZernikes(*angle, band),
                    None,
                    Image(np.zeros((1, 1)), angle, dfType, band),
                )

                # Convert normalized image coordinates to meters
                xImage = uImage * inst.donutRadius * inst.pixelSize
                yImage = vImage * inst.donutRadius * inst.pixelSize

                # Trace to the focal plane with Batoid
                optic.withLocallyShiftedOptic("Detector", [0, 0, offset]).trace(rays)

                # Calculate the centered ray coordinates
                chief = batoid.RayVector.fromStop(
                    0,
                    0,
                    optic,
                    wavelength=mapper.instrument.wavelength[band],
                    dirCos=dirCos,
                )
                optic.withLocallyShiftedOptic("Detector", [0, 0, offset]).trace(chief)
                xRay = rays.x - chief.x
                yRay = rays.y - chief.y

                # Calculate the residuals
                dx = xImage - xRay
                dy = yImage - yRay
                dr = np.sqrt(dx**2 + dy**2)

                # Only look at unvignetted rays
                dr = dr[~rays.vignetted]

                # Compare residuals to the pixel size
                mp = maxPercent(
                    instConfig=instConfig,
                    angle=angle,
                    model=model,
                    dfType=dfType,
                    band=band,
                )
                if not np.all(dr <= inst.pixelSize * mp / 100):
                    raise AssertionError(
                        "testBatoidRaytraceResiduals failed on configuration\n"
                        f"    instrument = {instConfig}\n"
                        f"    angle = {angle}\n"
                        f"    model = {model}\n"
                        f"    defocalType = {dfType}\n"
                        f"    band = {band}\n"
                        f"The maximum residual was {dr.max():.3e} m, "
                        f"which is {dr.max() / inst.pixelSize * 100:.2f}% of a "
                        f"pixel, exceeding the maximum value of {mp}%."
                    )

    def testDilatedBlendedMasks(self) -> None:
        """
        Blended masks are created by copying the source mask, dilating it,
        and shifting it to the positions of the blends. If you dilate the
        mask enough it will grow larger than the image size. Previously,
        this was not properly modeled, which resulted in sharp edges on
        masks where the dilated source mask hit the side of the image.

        This bug has been fixed so that blend masks do not have these sharp
        edges. This test makes sure the fix is working by dilating and shifting
        the mask by value larger than the image size. That way the blended mask
        should be all zero, but ONLY if the mask model is keeping track of mask
        values that are originally outside of the bounding box (pre-shift).
        """
        # Create default image mapper
        mapper = ImageMapper()

        # Create a dummy pupil image with a far-away blend
        uPupil, vPupil = mapper.instrument.createPupilGrid()
        pupil = Image(
            uPupil,
            (0, 0),
            "intra",
            planeType="pupil",
            blendOffsets=[[0, 200]],
        )

        # Create a blended mask with dilation = blend shift
        mapper.createPupilMasks(pupil, doMaskBlends=True, dilateBlends=200)

        # Confirm that all mask values are zero
        self.assertTrue(np.allclose(pupil.mask, 0))

        # Now do the same with an image mask
        image = Image(
            uPupil,
            (0, 0),
            "intra",
            planeType="image",
            blendOffsets=[[0, 200]],
        )
        mapper.createImageMasks(image, doMaskBlends=True, dilateBlends=200)
        self.assertTrue(np.allclose(image.mask, 0))
