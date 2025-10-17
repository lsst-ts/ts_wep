import lsst.afw.image
import numpy as np
from lsst.ts.wep.utils import binArray
from scipy.ndimage import binary_dilation
from scipy.signal import correlate, find_peaks, peak_prominences


class DonutSizeCorrelator:
    """A class estimating donut diameters directly from post_isr_image.

    This class estimates donut diameters by correlating the input image
    with a series of annular templates at different diameters.

    The main algorithm works by computing the maximum correlation between
    each donut-sized template and the image, resulting in a curve of
    maximum correlation as a function of donut diameter.

    It then identifies the most prominent peak in this curve to estimate
    the best-fitting diameter. If no strong peaks are found, the algorithm
    falls back to analyzing the second derivative of the log-log curve
    of diameter vs. maximum correlation to detect subtle inflection points.

    To optimize performance, the algorithm uses a multi-scale strategy
    with dynamically increasing resolution.

    Example usage (assuming exposure is a post_isr_image object):
        correlator = DonutSizeCorrelator()
        image = correlator.prepButlerExposure(exposure)
        diam, diamGrid, corrGrid = correlator.getDonutDiameter(image)
        plt.plot(diamGrid, corrGrid)
        plt.axvline(diam, c="r")
    """

    @staticmethod
    def prepButlerExposure(
        exposure: lsst.afw.image.Exposure,
        badPixelTypes: tuple[str, str, str, str] = ("SAT", "BAD", "NO_DATA", "INTRP"),
        nDilation: int = 100,
        fillVal: float = -10.0,
    ) -> np.ndarray:
        """Prep image from the butler for correlation.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            Exposure from the butler.
        badPixelTypes : tuple[str], optional
            Names of flags that will be masked.
            Default is ("SAT", "BAD", "NO_DATA", "INTRP")
        nDilation : int, optional
            Number of times to dilate bad-pixel mask.
            Default is 100.
        fillVal : float, optional
            Value used to fill in mask. Warning, np.nan will
            cause errors during correlation (FFT is used).
            Default is -10.

        Returns
        -------
        np.ndarray
            The image array, prepped for correlation.
        """
        # Extract image and mask arrays
        image = exposure.image.array

        # Subtract off approximate background from image
        image -= np.median(image)

        # Get bad-pixel mask
        bitMask = exposure.mask.getPlaneBitMask(badPixelTypes)
        badPixels = np.bitwise_and(exposure.mask.array, bitMask) > 0

        # Dilate the bad-pixel mask
        mask = binary_dilation(badPixels, iterations=nDilation)

        # Fill masked values
        image[mask] = fillVal

        # Normalize the image
        image /= image[~mask].max()
        return image

    @staticmethod
    def cropAndBinImage(
        image: np.ndarray,
        length: int | None = None,
        pad: int = 500,
        binning: int | None = None,
    ) -> np.ndarray:
        """Crop and bin the array.

        Parameters
        ----------
        image : np.ndarray
            Image array.
        length : int or None, optional
            Size length for crop. Default is None.
        pad : int, optional
            The image is cropped around the brightest pixel, subject
            to the condition that the pixel is not too close to the
            edge of the image. This pad sets that distance.
            Default is 500 pixels.
        binning : int or None, optional
            Binning factor. Default is None.

        Returns
        -------
        np.ndarray
            Cropped and binned image
        """
        # Crop array
        if length is not None:
            # We will crop around the brightest pixel
            # Cutout pads on side so we don't select for
            # bright donuts falling off the sensor
            padded = image[pad:-pad, pad:-pad]

            # Bin the array so we don't select individual hot pixels
            binningFactor = 8
            binned = binArray(padded, binningFactor)

            # Find location of brightest pixel
            y, x = np.where(binned == np.nanmax(binned))

            # Undo binning and pad
            x = binningFactor * x[0] + pad
            y = binningFactor * y[0] + pad

            # Crop around brightest pixel
            height, width = image.shape
            x0 = np.clip(x - length // 2, 0, width - length)
            y0 = np.clip(y - length // 2, 0, height - length)
            image = image[y0 : y0 + length, x0 : x0 + length]

        if binning is not None:
            image = binArray(image, binning, "median")
        return image

    @staticmethod
    def createDonutTemplate(diameter: float) -> np.ndarray:
        """Create simple annulus template for donuts.

        Parameters
        ----------
        diameter : float
            Diameter of donut.

        Returns
        -------
        np.ndarray
            Fractional donut mask
        """
        # Create grid of pixel centers
        x = np.arange(diameter + 5, dtype=float)
        x -= x.mean()
        x, y = np.meshgrid(x, x)

        # Build sub-pixel offsets
        nOffsets = 10
        offsets = np.arange(nOffsets + 1) / nOffsets - 0.5
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        dy = dy.reshape(-1, 1, 1)  # Reshape for broadcasting
        dx = dx.reshape(-1, 1, 1)

        # Distance to each subpixel
        r = np.sqrt((x + dx) ** 2 + (y + dy) ** 2)

        # Mask pixels by distance, then average number inside mask
        inside = (r >= 0.61 * diameter / 2) & (r <= diameter / 2)
        inside = inside.mean(axis=0).astype(float)

        return inside

    def correlateImage(
        self,
        image: np.ndarray,
        resolution: int = 4,
        dMin: int = 20,
        dMax: int = 500,
        length: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correlate the image with donuts of different sizes.

        Parameters
        ----------
        image : np.ndarray
            Image containing donuts
        resolution : int, optional
            Resolution of donut diameter in pixels. Default is 4.
        dMin : int, optional
            Minimum donut diameter in pixels. Default is 20.
        dMax : int, optional
            Maximum donut diameter in pixels. Default is 500.
        length : int or None, optional
            Size of stamp to cutout from center of image.
            Default is None.

        Returns
        -------
        np.ndarray
            Array of diameters in pixels
        np.ndarray
            Array of correlation values
        """
        # Crop and bin the image
        image = self.cropAndBinImage(
            image=image,
            length=length,
            pad=min(dMax // 2, np.min(image.shape) // 4),
            binning=resolution,
        )

        diameters = []
        correlations = []

        for diameter in np.arange(dMin, dMax + resolution, resolution):
            # Don't go past d_max
            if diameter > dMax:
                break
            # Create new template
            template = self.createDonutTemplate(diameter / resolution)

            # Normalize the template
            template /= template.sum()

            # Calculate max correlation in image
            corrImage = correlate(image, template)
            corr = np.nanmax(corrImage)

            # Save values
            diameters.append(diameter)
            correlations.append(corr)

        return np.array(diameters), np.array(correlations)

    def getDonutDiameter(
        self,
        image: np.ndarray,
        dMin: int | None = 10,
        dMax: int | None = 500,
    ) -> tuple[int | float, np.ndarray, np.ndarray]:
        """Get donut diameter by correlating the image.

        Parameters
        ----------
        image : np.ndarray
            Image containing donuts
        dMin : int or None, optional
            Minimum donut diameter in pixels. Default is 10.
        dMax : int or None, optional
            Maximum donut diameter in pixels. Default is 500.

        Returns
        -------
        int or float
            Estimate of the donut diameter in pixels.
            If the algorithm fails, it returns a NaN.
        np.ndarray
            Array of tested diameters
        np.ndarray
            Array of resulting correlations
        """
        # Instantiate empty arrays
        diameters = np.array([], dtype=int)
        correlations = np.array([])

        # We will progress by powers of 2 in resolution
        # and decrease resolution when resolution = 5% of diameter
        # and max diameter we will test is longest side of image
        maxLength = max(image.shape)
        nIterations = np.ceil(np.log2(0.05 * maxLength)).astype(int)

        for n in range(0, nIterations + 1):
            # Set resolution, diameter range, cutout size
            resolution = 2**n
            minDiameter = 4 if n == 0 else 20 * 2 ** (n - 1)
            maxDiameter = min(20 * 2**n, maxLength)
            cropLength = max(500, 500 * round(3 * maxDiameter / 500))

            # Resolve local and global limits
            if dMin is not None:
                minDiameter = max(minDiameter, dMin)
            if dMax is not None:
                maxDiameter = min(maxDiameter, dMax)

            # Skip iterations if limits are inconsistent
            if minDiameter > maxDiameter:
                continue

            # Calculate new correlations
            diam, corr = self.correlateImage(
                image=image,
                resolution=resolution,
                dMin=minDiameter,
                dMax=maxDiameter,
                length=cropLength,
            )

            # Normalize correlation due to change in length
            if len(correlations) > 0:
                corr *= correlations[-1] / corr[0]

            # Append to our existing arrays
            diameters = np.concatenate((diameters, diam[1:]))
            correlations = np.concatenate((correlations, corr[1:]))

        # Find correlation peaks
        peaks, _ = find_peaks(correlations)
        prominences, *_ = peak_prominences(correlations, peaks)

        # If we have peaks, select greatest prominence
        if len(peaks) > 0 and prominences.max() > 1e-2:
            # Index of greatest prominence
            solution = diameters[peaks[prominences.argmax()]]

        # Otherwise, get the first prominent peak in (negative) 2nd deriv
        # (looking for a sharp turn and decline in correlation)
        else:
            # Take log of both dimensions
            logDiam = np.log(diameters)
            logCorr = np.log(correlations)

            # Calculate (negative) second derivative
            secondDeriv = -np.gradient(np.gradient(logCorr, logDiam), logDiam)

            # We will calculate peaks/prominence with respect to zero
            secondDeriv = np.clip(np.append(secondDeriv, 0), 0, None)

            # Now find peaks and prominences
            peaks, _ = find_peaks(secondDeriv)
            prominences, *_ = peak_prominences(secondDeriv, peaks)
            if len(peaks) > 0:
                # Get first peak that is sufficiently prominent
                sufficientlyProminent = (prominences / prominences.max()) > 0.6
                solution = diameters[peaks[sufficientlyProminent][0]]

            # If still no peaks, we have failed
            # Note it would be very surprising to get here!
            else:
                solution = np.nan

        return solution, diameters, correlations
