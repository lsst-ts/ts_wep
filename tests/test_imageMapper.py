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

import numpy as np
from batoid import RayVector
from lsst.ts.wep import Image, ImageMapper
from scipy.ndimage import binary_opening


class TestImageMapper(unittest.TestCase):
    """Test the ImageMapper class."""

    @staticmethod
    def _createImageWithBatoid(nPixels, fieldAngle, defocalType, instrument):
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
        offset = defocalSign * instrument.batoidOffsetValue
        optic = instrument.batoidOffsetOptic
        model = instrument.getBatoidModel()
        model = model.withGloballyShiftedOptic(optic, [0, 0, offset])

        # We need to get the image position of the chief ray
        rays = RayVector.fromStop(
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
        rays = RayVector.fromStop(
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

    def testCreateWithDefaults(self):
        ImageMapper()

    def testBadOpticalModel(self):
        with self.assertRaises(TypeError):
            ImageMapper(opticalModel=1)
        with self.assertRaises(ValueError):
            ImageMapper(opticalModel="bad")

    def testCreatePupilMaskCenter(self):
        # Create the mapper and pull out the instrument
        mapper = ImageMapper()
        inst = mapper.instrument

        # Centered pupil mask should just be disk with central obscuration
        u, v = inst.createPupilGrid()
        r = np.sqrt(u**2 + v**2)
        truth = (r >= inst.obscuration) & (r <= 1)

        # Check that the binary mask matches the expected truth
        image = Image(np.zeros_like(r), (0, 0), "intra")
        mask = mapper.createPupilMask(image, binary=True)
        self.assertTrue(np.allclose(mask, truth))

    def testCreatePupilMaskOffCenter(self):
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
        rays = RayVector.fromStop(
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
            np.zeros((180, 180)),
            fieldAngle,
            "intra",
        )
        maskPupilModel = mapper.createPupilMask(image, binary=True)

        # Get the difference in the masks
        diff = maskPupilBatoid.astype(float) - maskPupilModel.astype(float)

        # Apply binary opening once to remove small artifacts at edges of masks
        diff = binary_opening(diff)

        self.assertTrue(np.allclose(diff, 0))

    def testCreateImageMaskOffCenter(self):
        for defocalType in ["intra", "extra"]:
            # Create the mapper and pull out the instrument
            mapper = ImageMapper()
            inst = mapper.instrument

            # Set field angle of the source
            fieldAngle = np.array([1.20, 1.27])  # degrees

            # First, let's get the model image mask
            maskImageModel = mapper.createImageMask(
                Image(
                    image=np.zeros((160, 160)),
                    fieldAngle=fieldAngle,
                    defocalType=defocalType,
                )
            )

            # Now get the Batoid mask
            batoidImage = self._createImageWithBatoid(
                len(maskImageModel),
                fieldAngle,
                defocalType,
                inst,
            )
            maskImageBatoid = batoidImage > 0

            # Get the difference in the masks
            diff = maskImageBatoid.astype(float) - maskImageModel.astype(float)

            # Binary opening to remove small artifacts at edges of masks
            diff = binary_opening(diff)

            # Calculate the fractional difference
            fracDiff = np.abs(diff).sum() / maskImageModel.sum()

            self.assertLess(fracDiff, 0.01)

    def testCenterOnProjection(self):
        # Forward model an image
        mapper = ImageMapper()
        image = Image(
            np.zeros((180, 180)),
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
            binary=True,
            rMax=np.inf,
        )
        self.assertTrue(np.allclose(recentered.image, image.image))

        # Recenter using the full template
        recentered = mapper.centerOnProjection(
            decentered,
            zk,
            binary=False,
            rMax=np.inf,
        )
        self.assertTrue(np.allclose(recentered.image, image.image))

    def testMapPupilToImage(self):
        for defocalType in ["intra", "extra"]:
            # Create the mapper and pull out the instrument
            mapper = ImageMapper()
            inst = mapper.instrument

            # Set field angle of the source
            fieldAngle = np.array([1.20, 1.27])  # degrees

            # First, let's map the pupil to the image plane
            image = mapper.mapPupilToImage(
                Image(
                    image=np.zeros((160, 160)),
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

    def testRoundTrip(self):
        rng = np.random.default_rng(0)

        for opticalModel, fieldAngle, defocalType, zk in itertools.product(
            ["onAxis", "offAxis"],
            [(0, 0), (1.2, -0.7)],
            ["intra", "extra"],
            [np.zeros(1), rng.normal(scale=50e-9, size=19)],
        ):
            # Create the Image mapper
            mapper = ImageMapper(opticalModel=opticalModel)

            # Forward model an image
            image = Image(
                np.zeros((180, 180)),
                fieldAngle,
                defocalType,
            )
            image = mapper.mapImageToPupil(image, zk)

            # Map image back to the pupil
            pupilRecon = mapper.mapImageToPupil(image, zk)

            # Create the pupil mask
            pupil = mapper.createPupilMask(image)

            # Calculate the difference between the pupil
            # and the reconstructed pupil
            diff = pupilRecon.image - pupil

            self.assertLess(diff.sum() / pupil.sum(), 0.02)
            self.assertLess(diff.max(), 1)

    def testMaskBlends(self):
        # Create a dummy image
        image = Image(
            np.zeros((160, 160)),
            (0, 0),
            "intra",
        )

        # Create the image mapper
        mapper = ImageMapper()

        # Replace the image with the image model
        image.image = mapper.mapPupilToImage(image).image

        # Test that a blend offset of 0 removes all the flux
        image.blendOffsets = [[0, 0]]
        self.assertTrue(np.allclose(mapper.createImageMask(image, maskBlends=True), 0))
        self.assertTrue(np.allclose(mapper.createPupilMask(image, maskBlends=True), 0))
        self.assertTrue(
            np.allclose(mapper.mapImageToPupil(image, maskBlends=True).image, 0)
        )
        self.assertTrue(
            np.allclose(mapper.mapPupilToImage(image, maskBlends=True).image, 0)
        )

        # Non-zero blend offset removes a portion of the flux
        image.blendOffsets = [[50, 50]]
        self.assertTrue(
            0
            < mapper.createImageMask(image, maskBlends=True).sum()
            < mapper.createImageMask(image, maskBlends=False).sum()
        )
        self.assertTrue(
            0
            < mapper.createPupilMask(image, maskBlends=True).sum()
            < mapper.createPupilMask(image, maskBlends=False).sum()
        )
        self.assertTrue(
            0
            < mapper.mapImageToPupil(image, maskBlends=True).image.sum()
            < mapper.mapPupilToImage(image, maskBlends=False).image.sum()
        )

    def testDilate(self):
        # Create a dummy image
        image = Image(
            np.zeros((160, 160)),
            (0, 0),
            "intra",
        )

        # Create the image mapper
        mapper = ImageMapper()

        # Test error with negative dilate
        with self.assertRaises(ValueError):
            mapper.createImageMask(image, dilate=-1)
        with self.assertRaises(ValueError):
            mapper.createPupilMask(image, dilate=-1)

        # Test that you can only dilate a binary mask
        with self.assertRaises(ValueError):
            mapper.createImageMask(image, binary=False, dilate=1)
        with self.assertRaises(ValueError):
            mapper.createPupilMask(image, binary=False, dilate=1)

        # Test that the dilated mask is bigger
        self.assertGreater(
            mapper.createImageMask(image, binary=True, dilate=1).sum(),
            mapper.createImageMask(image, binary=True).sum(),
        )
        self.assertGreater(
            mapper.createPupilMask(image, binary=True, dilate=1).sum(),
            mapper.createPupilMask(image, binary=True).sum(),
        )

    def testDilateBlends(self):
        # Create a dummy image
        image = Image(
            np.zeros((160, 160)),
            (0, 0),
            "intra",
            blendOffsets=[[-20, 30]],
        )

        # Create the image mapper
        mapper = ImageMapper()

        # Test error with negative dilate
        with self.assertRaises(ValueError):
            mapper.createImageMask(image, dilateBlends=-1)
        with self.assertRaises(ValueError):
            mapper.createPupilMask(image, dilateBlends=-1)

        # Test that you CAN dilate blends for a fractional binary mask
        mapper.createImageMask(image, binary=False, dilateBlends=1, maskBlends=True)
        mapper.createPupilMask(image, binary=False, dilateBlends=1, maskBlends=True)

        # Test that the mask with dilated blends is smaller
        self.assertLess(
            mapper.createImageMask(image, maskBlends=True, dilateBlends=1).sum(),
            mapper.createImageMask(image, maskBlends=True).sum(),
        )
        self.assertLess(
            mapper.createPupilMask(image, maskBlends=True, dilateBlends=1).sum(),
            mapper.createPupilMask(image, maskBlends=True).sum(),
        )

    def testGetProjectionSize(self):
        mapper = ImageMapper()

        # Check against tested values
        self.assertEqual(mapper.getProjectionSize((0, 0), "intra"), 135)
        self.assertEqual(mapper.getProjectionSize((0, 0), "extra"), 136)
        self.assertEqual(mapper.getProjectionSize((1.2, 0.3), "intra"), 143)
        self.assertEqual(mapper.getProjectionSize((1.2, 0.3), "extra"), 146)
