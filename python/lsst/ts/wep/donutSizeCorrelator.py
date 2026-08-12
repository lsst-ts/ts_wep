import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter
from scipy.signal import correlate, find_peaks, peak_prominences

import lsst.afw.image
from lsst.ts.wep.utils import binArray


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
        image = exposure.image.array.copy()

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
        if np.any(~mask):
            image /= image[~mask].max()
        else:
            # Entire image is masked — return zeros
            image[:] = 0.0
        return image

    @staticmethod
    def selectCropCenters(
        image: np.ndarray,
        pad: int = 500,
        nCenters: int = 5,
        minSeparation: int = 40,
    ) -> list[tuple[int, int]]:
        """Find the most donut-like locations to center crops on.

        We select around the most donut-like locations rather than the
        brightest pixel. Selecting the brightest pixel is fooled by
        bright crescents (partial or mask-clipped donuts), whose flux is
        concentrated into an arc; those do not correlate well with a
        full annulus, so a donut-likeness map avoids centering the crop
        on them.

        Multiple candidates are returned (ranked by donut-likeness) so
        that ``getDonutDiameter`` can measure each and take the median.
        This is robust against a single spurious top-ranked location --
        e.g. a bright cosmic ray / saturated star that happens to
        correlate with an annulus, or a mask-clipped crescent -- which
        can outrank the real donuts but cannot outvote several of them.

        The candidates are spatially declustered so that no single
        feature contributes more than one of them. A large, bright
        feature (e.g. a satellite trail or saturated spike) produces
        several local maxima of the likeness map along its length; if
        those all became candidates they would give that one feature a
        majority of the votes and the median would follow it. Enforcing
        a minimum separation guarantees each candidate is a distinct
        object.

        Parameters
        ----------
        image : np.ndarray
            Image array.
        pad : int, optional
            Edge margin excluded from selection, so we don't center on
            bright donuts falling off the sensor. Default is 500 pixels.
        nCenters : int, optional
            Maximum number of candidate centers to return. Default is 5.
        minSeparation : int, optional
            Minimum separation (in binned pixels) between candidates, so
            no single feature contributes more than one candidate. In
            full-image pixels this is ``8 * minSeparation``. Default is
            40 (320 full pixels); large enough that an extended feature
            (e.g. a satellite trail) yields at most one candidate,
            while still resolving neighbouring donuts.

        Returns
        -------
        list[tuple[int, int]]
            Candidate (y, x) locations in full-image coordinates, ranked
            by donut-likeness (most likely first).
        """
        # Cutout pads on side so we don't select for
        # bright donuts falling off the sensor
        padded = image[pad:-pad, pad:-pad]

        # Bin the array so we don't select individual hot pixels
        binningFactor = 8
        binned = binArray(padded, binningFactor)

        # Donut-likeness map, then all local maxima ranked best first
        likeness = DonutSizeCorrelator.donutLikenessMap(binned, binningFactor)
        localMax = maximum_filter(likeness, size=3)
        peaks = np.argwhere((likeness == localMax) & np.isfinite(likeness))
        order = np.argsort(likeness[peaks[:, 0], peaks[:, 1]])[::-1]
        peaks = peaks[order]

        # Greedily accept peaks, skipping any too close to one already
        # accepted, so each candidate is a distinct feature
        centers: list[tuple[int, int]] = []
        minSepSq = minSeparation**2
        for y, x in peaks:
            if all((y - py) ** 2 + (x - px) ** 2 >= minSepSq for py, px in centers):
                centers.append((int(y), int(x)))
            if len(centers) >= nCenters:
                break

        # Undo binning and pad
        return [
            (binningFactor * y + pad, binningFactor * x + pad) for y, x in centers
        ]

    @staticmethod
    def selectCropCenter(image: np.ndarray, pad: int = 500) -> tuple[int, int]:
        """Find the single most donut-like location (see selectCropCenters)."""
        return DonutSizeCorrelator.selectCropCenters(image, pad, nCenters=1)[0]

    @staticmethod
    def cropAndBinImage(
        image: np.ndarray,
        length: int | None = None,
        pad: int = 500,
        binning: int | None = None,
        center: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Crop and bin the array.

        Parameters
        ----------
        image : np.ndarray
            Image array.
        length : int or None, optional
            Size length for crop. Default is None.
        pad : int, optional
            Edge margin used when selecting the crop center (only when
            ``center`` is None). Default is 500 pixels.
        binning : int or None, optional
            Binning factor. Default is None.
        center : tuple[int, int] or None, optional
            Precomputed (y, x) crop center in full-image coordinates.
            If None, it is selected via ``selectCropCenter``. Passing a
            fixed center keeps the crop on the same donut across the
            multi-scale iterations of ``getDonutDiameter``. Default is
            None.

        Returns
        -------
        np.ndarray
            Cropped and binned image
        """
        # Crop array
        if length is not None:
            if center is None:
                center = DonutSizeCorrelator.selectCropCenter(image, pad)
            y, x = center

            # Crop around selected location
            height, width = image.shape
            x0 = np.clip(x - length // 2, 0, width - length)
            y0 = np.clip(y - length // 2, 0, height - length)
            image = image[y0 : y0 + length, x0 : x0 + length]

        if binning is not None:
            image = binArray(image, binning, "median")
        return image

    @staticmethod
    def donutLikenessMap(
        binned: np.ndarray,
        binningFactor: int,
        diameters: tuple[int, ...] = (40, 60, 90, 140, 210, 320, 480),
        maskThreshold: float = -1.0,
        maskFracMax: float = 0.2,
    ) -> np.ndarray:
        """Score each location by how well it matches a full donut.

        The binned image is correlated with a set of annular templates
        spanning the plausible donut-size range. Each correlation is
        normalized by the template's own energy so that the score
        reflects match quality (shape) rather than raw brightness.
        Taking the maximum across templates yields a scale-invariant
        donut-likeness map: bright crescents (partial or mask-clipped
        donuts) score poorly because they do not fill a full annulus,
        while intact donuts score highly.

        The template normalization only removes an overall brightness
        scale, so a very bright mask-clipped crescent can still outscore
        a fainter intact donut. To guard against this, any location
        whose donut footprint overlaps the bad-pixel mask by more than
        ``maskFracMax`` is vetoed. Masked pixels are filled with a large
        negative value in ``prepButlerExposure``, so mean-binning drives
        their bins strongly negative; bins below ``maskThreshold`` are
        treated as masked. If every location is vetoed (e.g. a heavily
        masked field), the un-vetoed map is returned as a fallback.

        The templates only *locate* the crop; the donut size itself is
        measured downstream by ``getDonutDiameter``. Selection is robust
        only when at least one template is within ~1.4x of the true
        donut size, so the default set is spaced geometrically by ~1.5x
        to cover the whole range a badly-defocused system may produce
        (this code exists to catch exactly those cases, so the range
        cannot be narrowed to a commanded defocus).

        Parameters
        ----------
        binned : np.ndarray
            Coarsely binned image (see ``cropAndBinImage``).
        binningFactor : int
            Binning factor used to produce ``binned``. Template
            diameters are scaled down by this factor to match.
        diameters : tuple[int], optional
            Full-resolution donut diameters (pixels) to probe.
            Default is (40, 60, 90, 140, 210, 320, 480), spanning the
            plausible donut-size range with ~1.5x geometric spacing.
        maskThreshold : float, optional
            Binned pixels below this value are treated as masked.
            Default is -1.0.
        maskFracMax : float, optional
            Veto a location if more than this fraction of its donut
            footprint overlaps the mask. Default is 0.2.

        Returns
        -------
        np.ndarray
            Map, same shape as ``binned``, of donut-likeness scores.
        """
        # Mask indicator: masked pixels were filled with a large
        # negative value, so mean-binning drives their bins negative.
        maskedBins = (binned < maskThreshold).astype(float)

        likeness = np.full(binned.shape, -np.inf)
        likenessNoVeto = np.full(binned.shape, -np.inf)
        for diameter in diameters:
            # Scale template to the binned resolution
            binnedDiameter = diameter / binningFactor
            if binnedDiameter < 3:
                continue
            annulus = DonutSizeCorrelator.createDonutTemplate(binnedDiameter)
            if annulus.shape[0] > binned.shape[0] or annulus.shape[1] > binned.shape[1]:
                continue

            # Fraction of this donut footprint that overlaps the mask
            footSum = annulus.sum()
            if footSum == 0:
                continue
            footprint = annulus / footSum
            maskFrac = correlate(maskedBins, footprint, mode="same")

            # Normalize so score reflects match quality, not brightness
            template = annulus - annulus.mean()
            norm = np.sqrt((template**2).sum())
            if norm == 0:
                continue
            template /= norm
            corr = correlate(binned, template, mode="same")

            likenessNoVeto = np.maximum(likenessNoVeto, corr)

            # Veto locations whose donut overlaps the mask (e.g. bright
            # crescents from mask-clipped donuts)
            vetoed = corr.copy()
            vetoed[maskFrac > maskFracMax] = -np.inf
            likeness = np.maximum(likeness, vetoed)

        # Fallback if the veto removed every candidate
        if not np.isfinite(likeness).any():
            return likenessNoVeto
        return likeness

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
        center: tuple[int, int] | None = None,
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
        center : tuple[int, int] or None, optional
            Precomputed (y, x) crop center in full-image coordinates,
            passed through to ``cropAndBinImage``. Default is None.

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
            center=center,
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

    def _diameterAtCenter(
        self,
        image: np.ndarray,
        cropCenter: tuple[int, int],
        dMin: int | None = 10,
        dMax: int | None = 500,
    ) -> tuple[int | float, np.ndarray, np.ndarray]:
        """Measure the donut diameter for a single, fixed crop center.

        Runs the multi-scale correlation sweep with the crop locked to
        ``cropCenter`` at every resolution, then estimates the diameter
        from the resulting correlation curve. See ``getDonutDiameter``
        for the return contract.
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

            # Calculate new correlations. The crop center is fixed across
            # resolutions; re-selecting per iteration let the chosen donut
            # jump between scales, splicing correlation curves from
            # different objects into one curve and producing spurious
            # peaks (e.g. a diameter of 18 when the true donut is ~190).
            diam, corr = self.correlateImage(
                image=image,
                resolution=resolution,
                dMin=minDiameter,
                dMax=maxDiameter,
                length=cropLength,
                center=cropCenter,
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

    def getDonutDiameter(
        self,
        image: np.ndarray,
        dMin: int | None = 10,
        dMax: int | None = 500,
        nCenters: int = 5,
    ) -> tuple[int | float, np.ndarray, np.ndarray]:
        """Get donut diameter by correlating the image.

        The donut-likeness map can rank a spurious feature first -- a
        bright cosmic ray or saturated star that happens to correlate
        with an annulus, or a mask-clipped crescent. Rather than trust a
        single crop location, we measure the diameter at each of the top
        ``nCenters`` candidates and return the median, so no single bad
        location determines the result. The returned diameter/correlation
        arrays are those of the candidate whose diameter is nearest the
        median (so a plot of the curve matches the reported value).

        Parameters
        ----------
        image : np.ndarray
            Image containing donuts
        dMin : int or None, optional
            Minimum donut diameter in pixels. Default is 10.
        dMax : int or None, optional
            Maximum donut diameter in pixels. Default is 500.
        nCenters : int, optional
            Number of candidate donut locations to measure and take the
            median over. Default is 5.

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
        # Rank candidate donut locations by donut-likeness
        selectPad = min((dMax or 500) // 2, np.min(image.shape) // 4)
        centers = self.selectCropCenters(image, selectPad, nCenters=nCenters)

        # Measure the diameter at each candidate, recording how prominent
        # its correlation peak is. A real donut yields a prominent peak;
        # crescents, mask-block edges and background texture yield a
        # monotonic curve with no real peak. This is the same "realness"
        # signal that _diameterAtCenter uses to pick its primary branch.
        results = []
        for center in centers:
            solution, diameters, correlations = self._diameterAtCenter(
                image, center, dMin=dMin, dMax=dMax
            )
            if not np.isfinite(solution):
                continue
            peaks, _ = find_peaks(correlations)
            if len(peaks) > 0:
                maxProm = peak_prominences(correlations, peaks)[0].max()
            else:
                maxProm = 0.0
            results.append((solution, diameters, correlations, maxProm))

        # If every candidate failed, return the last attempt (NaN)
        if len(results) == 0:
            return solution, diameters, correlations

        # Keep only candidates with a genuinely prominent peak. Without
        # this, a sparse field (few real donuts, several junk features)
        # lets the junk outvote the donuts in the median, since the real
        # donuts can be a minority of the candidates. Fall back to all
        # candidates if none clear the threshold.
        prominent = [r for r in results if r[3] > 1e-2]
        if len(prominent) > 0:
            results = prominent

        # Take the median diameter, robust against a single bad location
        solutions = np.array([r[0] for r in results])
        medianDiam = np.median(solutions)

        # Return the arrays of the candidate closest to the median so the
        # reported diameter and the plotted curve stay consistent
        best = int(np.argmin(np.abs(solutions - medianDiam)))
        return solutions[best], results[best][1], results[best][2]
