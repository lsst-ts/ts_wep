from __future__ import annotations

import numpy as np
from astropy.table import QTable
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.ndimage import maximum_filter

import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.image import Exposure, ImageF


def _nOffsetsFor(diameter: float) -> int:
    """Number of sub-pixel sampling offsets per axis for anti-aliasing a disk.

    Chooses how finely to super-sample each pixel when rasterizing a disk
    template (see `_createDisk`). Small disks need more offsets because a single
    sample per pixel badly quantizes their edge; large disks need fewer because
    their edge already spans many pixels. The heuristic 60/diameter is clipped to
    [2, 10] to bound both the aliasing error (floor) and the cost (ceiling): the
    inner loops in `_createDisk` scale as (nOffsets + 1)**2.

    Parameters
    ----------
    diameter : float
        Disk diameter in (binned) pixels.

    Returns
    -------
    int
        Number of sub-pixel offsets per axis, in [2, 10].
    """
    return int(np.clip(int(np.ceil(60.0 / diameter)), 2, 10))


def _createDisk(diameter: float, nOffsets: int | None = None) -> np.ndarray:
    """Rasterize an anti-aliased filled-disk template centered on the grid.

    Each pixel value is the fraction of its area covered by a disk of the given
    diameter, estimated by super-sampling: the disk membership test is evaluated
    on an (nOffsets + 1) x (nOffsets + 1) grid of sub-pixel offsets spanning
    [-0.5, 0.5) in each axis and averaged. This yields soft edge pixels in
    (0, 1) rather than a hard binary mask, which keeps the correlation scores
    smooth in diameter and centroid position.

    The output grid is sized `diameter + 5` per axis and centered, so the disk is
    fully contained with a small margin. The template is not normalized; its sum
    (`~pi*(diameter/2)**2`) is used elsewhere (via _buildRegions) to build the
    matched-filter template norms.

    Parameters
    ----------
    diameter : float
        Disk diameter in (binned) pixels.
    nOffsets : int or None
        Number of sub-pixel offsets per axis for anti-aliasing. If None, chosen
        automatically via `_nOffsetsFor(diameter)`.

    Returns
    -------
    np.ndarray
        2D float64 array of per-pixel area-coverage fractions in [0, 1],
        centered on the grid.
    """
    if nOffsets is None:
        nOffsets = _nOffsetsFor(diameter)
    x = np.arange(diameter + 5, dtype=float)
    x -= x.mean()
    xx, yy = np.meshgrid(x, x)
    offsets = np.arange(nOffsets + 1) / nOffsets - 0.5
    R2 = (diameter / 2) ** 2
    acc = np.zeros(xx.shape, dtype=np.float32)
    for dy in offsets:
        y2 = (yy + dy) ** 2
        for dx in offsets:
            acc += ((xx + dx) ** 2 + y2 <= R2)
    acc /= (nOffsets + 1) ** 2
    return acc.astype(np.float64)


def _cropSame(full, imgShape, kerShape):
    """Crop a full (linear) correlation output back to the input image shape.

    FFT-based correlation is computed on a zero-padded 'full' grid of size
    imgShape + kerShape - 1; this extracts the central `imgShape` region that
    corresponds to the kernel centered on each input pixel, i.e. the equivalent
    of scipy's mode='same'. The crop origin (r0, c0) uses the (k - 1) // 2
    convention so the kernel's center pixel aligns with the image pixel.

    Parameters
    ----------
    full : np.ndarray
        Full linear correlation result, shape >= imgShape + kerShape - 1.
    imgShape : tuple[int, int]
        Shape of the original (unpadded) image.
    kerShape : tuple[int, int]
        Shape of the kernel (disk template).

    Returns
    -------
    np.ndarray
        Central region of `full`, of shape `imgShape`.
    """
    r0 = (kerShape[0] - 1) // 2
    c0 = (kerShape[1] - 1) // 2
    return full[r0:r0 + imgShape[0], c0:c0 + imgShape[1]]


def _shiftSubtractNoise(image):
    """Robust per-pixel noise sigma via shift-and-subtract.

    Differencing adjacent ROWS cancels any fixed per-column offset (the residual
    amplifier banding in these images) along with the sky gradient and nearly all
    donut signal, so real sources do not inflate the estimate the way a plain MAD
    of the image would. The IQR of the difference is then converted to a sigma of
    the original image: /1.349 for IQR->sigma, /sqrt(2) because differencing two
    independent pixels doubles the variance.

    Same estimator as used on stamp backgrounds elsewhere in ts_wep (see
    MeasureDonutCandidatesTask._measureFlux in donutBlitzMonolith), applied to the whole
    binned image rather than to a stamp's background annulus.

    Returns
    -------
    float
        Noise sigma, or NaN if it cannot be estimated (degenerate image).
    """
    diff = np.diff(np.asarray(image, dtype=float), axis=0)
    if diff.size == 0 or not np.any(np.isfinite(diff)):
        return np.nan
    q75, q25 = np.nanpercentile(diff, [75, 25])
    sigma = (q75 - q25) / 1.349 / np.sqrt(2)
    return float(sigma) if np.isfinite(sigma) and sigma > 0 else np.nan


def makeDiameterLadder(dMin, dMax, innerFrac=0.61, k=3):
    """kth-root-of-(1/innerFrac) geometric ladder, clipped to [dMin, dMax]."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ratio = (1.0 / innerFrac) ** (1.0 / k)
    n = int(np.ceil(np.log(dMax / dMin) / np.log(ratio)))
    ladder = dMin * ratio ** np.arange(n + 1)
    return ladder[ladder <= dMax + 1e-9]


class DiskCorrelationBank:
    """Precomputed disk templates + FFTs for a fixed binned shape.

    Provides correlations of both the image and the image-squared against each
    disk template. The image-squared correlations supply the local image energy
    needed for the normalized cross-correlation (cosine-similarity) score in
    DonutDetectDiameterTask._annulusTerms.

    Correlation outputs are expressed in TEMPLATE-SUM units (the anti-aliased
    disk template sums), not raw pixel counts.
    """

    def __init__(self, binnedShape, diameters, innerFrac=0.61, backgroundSteps=2):
        self.binnedShape = tuple(int(s) for s in binnedShape)
        self.diameters = np.sort(np.asarray(diameters, dtype=float))
        self.innerFrac = innerFrac
        self.backgroundSteps = int(backgroundSteps)

        disks = [_createDisk(d) for d in self.diameters]
        self.diskShapes = [dk.shape for dk in disks]

        maxH = max(s[0] for s in self.diskShapes)
        maxW = max(s[1] for s in self.diskShapes)
        needed = np.array(self.binnedShape) + np.array([maxH, maxW]) - 1
        self.fshape = [int(next_fast_len(int(n), real=True)) for n in needed]

        self.F_disks = [rfftn(dk, self.fshape, workers=1) for dk in disks]

        nDiam = len(self.diameters)

        # Inner disk index: the disk whose diameter == innerFrac * d_i.
        inner = innerFrac * self.diameters
        self.innerIndex = np.full(nDiam, -1, dtype=int)
        for i, di in enumerate(inner):
            j = int(np.argmin(np.abs(self.diameters - di)))
            if np.isclose(self.diameters[j], di, rtol=1e-2):
                self.innerIndex[i] = j

        # Outer background index: n ladder steps up, clipped to the top rung.
        self.outerIndex = np.full(nDiam, -1, dtype=int)
        for i in range(nDiam):
            oi = min(i + self.backgroundSteps, nDiam - 1)
            self.outerIndex[i] = oi if oi > i else -1

        # Per-diameter matched-filter template norm ||t||, against the SAME
        # both-sided background (inner hole + outer collar) used at apply time.
        # Also cache the background weight A_S/A_B and the total support area
        # A_U = A_S + A_B, both needed by the energy-normalized score.
        self.annulusNorm = np.zeros(nDiam)
        self.bgWeight = np.zeros(nDiam)     # A_S / A_B
        self.supportArea = np.zeros(nDiam)  # A_U = A_S + A_B  (template-sum units)
        for i in range(nDiam):
            ii = self.innerIndex[i]
            if ii < 0:
                continue
            oi = self.outerIndex[i]
            signal, background = self._buildRegions(disks, i, ii, oi)
            A_S = signal.sum()
            A_B = background.sum()
            if A_S <= 0 or A_B <= 0:
                continue
            # Zero-mean matched-filter template: t = signal - (A_S/A_B)*background.
            t = signal - (A_S / A_B) * background
            self.annulusNorm[i] = np.sqrt((t ** 2).sum())
            self.bgWeight[i] = A_S / A_B
            self.supportArea[i] = A_S + A_B

    @staticmethod
    def _buildRegions(disks, i, ii, oi):
        """Return (signalIndicator, backgroundIndicator) on a common centered
        grid, from the anti-aliased disk templates.

        signal     = disk(d_i) - disk(d_ii)                 [the annulus]
        background = disk(d_ii) + (disk(d_oi) - disk(d_i))  [hole + outer collar]
                   = disk(d_oi) - signal
        """
        big = disks[oi] if oi >= 0 else disks[i]
        H, W = big.shape

        def centerPad(a):
            out = np.zeros((H, W), dtype=float)
            h, w = a.shape
            r0 = (H - h) // 2
            c0 = (W - w) // 2
            out[r0:r0 + h, c0:c0 + w] = a
            return out

        dI = centerPad(disks[i])
        dII = centerPad(disks[ii])
        signal = dI - dII
        if oi >= 0:
            dOI = centerPad(disks[oi])
            background = dOI - signal
        else:
            background = dII
        return signal, background

    def correlate(self, image, secondMoment=True):
        """Correlate `image` (and optionally `image**2`) against each disk.

        Parameters
        ----------
        image : np.ndarray
            Binned image, shape must equal self.binnedShape.
        secondMoment : bool
            If True, also return the image-squared correlations (needed for the
            energy normalization). Pass False for the bad-pixel mask, which only
            needs the first moment -- this saves one rfftn + one irfftn/diameter.

        Returns
        -------
        diskCorr : list[np.ndarray]
            <image, disk_i> at each location (template-sum units), one per diam.
        diskCorrSq : list[np.ndarray] or None
            <image**2, disk_i>, or None if secondMoment is False.
        """
        image = np.asarray(image, dtype=float)
        if image.shape != self.binnedShape:
            raise ValueError(
                f"image shape {image.shape} != bank shape {self.binnedShape}"
            )
        F_image = rfftn(image, self.fshape, workers=1)
        F_image2 = rfftn(image * image, self.fshape, workers=1) if secondMoment else None

        corr = []
        corrSq = [] if secondMoment else None
        for F_disk, shape in zip(self.F_disks, self.diskShapes):
            c = irfftn(F_image * F_disk, self.fshape, workers=1)
            corr.append(_cropSame(c, self.binnedShape, shape))
            if secondMoment:
                c2 = irfftn(F_image2 * F_disk, self.fshape, workers=1)
                corrSq.append(_cropSame(c2, self.binnedShape, shape))
        return corr, corrSq


class DonutDetectDiameterConfig(pexConfig.Config):
    edgeMargin = pexConfig.Field(
        doc="Width of detector edge region to exclude, in full-res pixels.",
        dtype=int,
        default=1,
    )
    detectionBinning = pexConfig.Field(
        doc="Integer factor to bin the image before correlation.",
        dtype=int,
        default=8,
    )
    dMinFull = pexConfig.Field(
        doc="Smallest donut diameter to probe, in full-res pixels.",
        dtype=float,
        default=30.0,  # ~1/4 x nominal
    )
    dMaxFull = pexConfig.Field(
        doc="Largest donut diameter to probe, in full-res pixels.",
        dtype=float,
        default=480.0, # ~4 x nominal
    )
    innerFrac = pexConfig.Field(
        doc="Inner/outer diameter ratio (central obscuration) of the donut.",
        dtype=float,
        default=0.61,
    )
    rootOrder = pexConfig.Field(
        doc=("Root order k for the diameter ladder: ratio = (1/innerFrac)**(1/k). "
             "Larger k -> finer diameter sampling at proportionally higher cost."),
        dtype=int,
        default=5,
    )
    badPixelTypes = pexConfig.ListField(
        doc="Mask plane names treated as bad. Matching pixels are zeroed before "
            "binning and correlation (see _prepImage).",
        dtype=str,
        default=["SAT", "BAD", "NO_DATA", "INTRP"],
    )
    backgroundSteps = pexConfig.Field(
        doc=("Number of ladder steps beyond each annulus's outer radius to "
             "include (together with the inner hole) when estimating the local "
             "background for the zero-mean matched filter. Larger values sample "
             "more background (lower noise) but reach further into neighbours "
             "in crowded fields."),
        dtype=int,
        default=2,
    )
    energyFloorFactor = pexConfig.Field(
        doc=("Noise floor added to the local image energy in the normalized "
             "cross-correlation denominator, as a fraction of the median local "
             "energy across the map. Prevents flat/empty regions (tiny energy) "
             "from producing spuriously high cosine scores. 0 disables the "
             "floor (pure cosine similarity)."),
        dtype=float,
        default=0.1,
    )
    nPeaks = pexConfig.Field(
        doc="Maximum number of donut candidates to pool the sizing curve over.",
        dtype=int,
        default=5,
    )
    peakMinSeparationFactor = pexConfig.Field(
        doc="Minimum separation between candidate peaks, in units of the "
            "winning (per-peak) binned donut diameter. Declusters extended "
            "features.",
        dtype=float,
        default=1.0,
    )
    likenessEdgeClip = pexConfig.Field(
        doc="Number of binned pixels to clip from the likeness map edges "
            "(where the correlation is biased by zero-padding).",
        dtype=int,
        default=10,
    )
    subtractColumnMedian = pexConfig.Field(
        doc="Subtract the column median from the binned image before correlation.",
        dtype=bool,
        default=False,
    )
    likenessThreshold = pexConfig.Field(
        doc=("Minimum likeness (cosine shape-agreement score) for a peak to be "
             "kept. Rejects wrong-shape features at any brightness. Set <= 0 to "
             "disable."),
        dtype=float,
        default=0.6,
    )
    snrThreshold = pexConfig.Field(
        doc=("Minimum matched-filter SNR (at the winning diameter) for a peak to "
             "be kept. Rejects noise peaks at any shape. Set <= 0 to disable."),
        dtype=float,
        default=15.0,
    )

    def validate(self):
        super().validate()
        if self.rootOrder < 1:
            raise pexConfig.FieldValidationError(
                self.__class__.rootOrder, self, "rootOrder must be >= 1"
            )
        if self.dMaxFull <= self.dMinFull:
            raise pexConfig.FieldValidationError(
                self.__class__.dMaxFull, self, "dMaxFull must exceed dMinFull"
            )
        if self.detectionBinning < 1:
            raise pexConfig.FieldValidationError(
                self.__class__.detectionBinning, self,
                "detectionBinning must be >= 1",
            )


# ----------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------

class DonutDetectDiameterTask(pipeBase.Task):
    """Detect donuts and estimate their diameters via disk cross-correlation.

    Bins the post-ISR exposure and correlates it against a geometric ladder of
    filled-disk templates (whose differences form annuli). For each diameter, a
    NORMALIZED cross-correlation (cosine-similarity) score is formed:

        rho = <image, t> / (||t|| * ||image_patch||)

    where t is the zero-mean annulus template and ||image_patch|| is the local
    mean-removed image energy under the template's support (obtained from the
    image-squared correlation). This makes the score a pure shape-agreement
    measure in ~[-1, 1], independent of BOTH donut size and brightness -- so a
    small crisp donut is not out-scored by a large or merely-bright feature.

    Bad pixels (config.badPixelTypes) are zeroed before binning (_prepImage).
    Peaks are declustered by a per-peak separation scaled to the winning
    diameter, then quality-cut, in _selectPeaks. Finally the pooled per-detection
    likeness sizing curve yields a single exposure diameter.

    Quality-control cuts and diagnostics
    ------------------------------------
    Up to `config.nPeaks` peaks are returned; a peak must pass BOTH a likeness
    and a matched-filter SNR threshold (config.likenessThreshold,
    config.snrThreshold) to survive. The cuts are applied inside peak selection
    (_selectPeaks), so pruning a failing candidate lets the descent through the
    sorted maxima continue -- the nPeaks slots fill with genuine survivors
    rather than being spent on rejects. Each peak carries two complementary
    scores:

        likeness -- the cosine score above. Measures SHAPE agreement only; being
            scale- and brightness-invariant, it discards the amplitude, which is
            why pure-noise local maxima can still reach ~0.5-0.6 (hence the
            default likeness threshold sits just above that noise floor).
        snr      -- the matched-filter SNR, dot / (||t|| * pixelNoise). Restores
            the amplitude information the cosine normalization throws away.

    They fail differently -- and so are required jointly: the cosine rejects
    wrong-shape features at any brightness, the SNR rejects noise at any shape.
    On a measured failure (2026070900182 det 203, 2 real donuts and 3 noise
    peaks) the two real peaks separated from the three spurious ones by ~1.7x in
    likeness but ~15x in SNR.

    Also reported per peak (diagnostic only, no cut applied) is
    `curve_argmax_edge`, true when the sizing curve's maximum lands on a ladder
    endpoint -- a monotonic curve carrying no interior scale, which is how a
    spurious result ends up pinned to an endpoint rung.
    """

    ConfigClass = DonutDetectDiameterConfig
    _DefaultName = "donutDetectDiameter"
    config: DonutDetectDiameterConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config

        self._dMinBinned = cfg.dMinFull / cfg.detectionBinning
        self._dMaxBinned = cfg.dMaxFull / cfg.detectionBinning
        self._diameters = makeDiameterLadder(
            self._dMinBinned, self._dMaxBinned, cfg.innerFrac, cfg.rootOrder
        )
        self._bank: DiskCorrelationBank | None = None

    # --- helpers --------------------------------------------------------

    def _getBank(self, binnedShape):
        if self._bank is None or self._bank.binnedShape != tuple(binnedShape):
            self._bank = DiskCorrelationBank(
                binnedShape, self._diameters, self.config.innerFrac,
                backgroundSteps=self.config.backgroundSteps,
            )
        return self._bank

    def _prepImage(self, exposure: Exposure):
        """Edge-trim, bad-pixel-fill, and bin.

        Returns
        -------
        binnedImage : np.ndarray
            Binned, bad-pixel-filled image.
        bbox : lsst.geom.Box2I
            The edge-eroded bounding box (for pixel-coordinate mapping).
        """
        c = self.config
        bbox = exposure.getBBox().erodedBy(c.edgeMargin)
        sub = exposure[bbox]

        image = sub.image.array.astype(float).copy()

        bitMask = sub.mask.getPlaneBitMask(list(c.badPixelTypes))
        badPixels = np.bitwise_and(sub.mask.array, bitMask) > 0

        image[badPixels] = 0.0
        if not np.any(~badPixels):
            image[:] = 0.0

        binning = c.detectionBinning
        if binning > 1:
            filledImg = ImageF(image.astype(np.float32))
            binnedImage = afwMath.binImage(filledImg, binning).array.astype(float)
        else:
            binnedImage = image

        if c.subtractColumnMedian:
            binnedImage -= np.nanmedian(binnedImage, axis=0)

        return binnedImage, bbox

    def _annulusTerms(self, diskCorr, diskCorrSq, bank, pixelNoise=None):
        """Per-rung NORMALIZED cross-correlation (cosine-similarity) response,
        and the matched-filter SNR of the same template.

        For each diameter with a valid inner disk, yields the shape-agreement
        score

            rho = dot / (||t|| * eLoc)

        where
            dot  = <image, t> = sumS - bgWeight * sumB   (a SUM, matches ||t||)
            eLoc = sqrt( <image^2, U> - <image, U>^2 / A_U )   local image energy
                   (mean-removed, over the signal+background support U)

        rho lies in ~[-1, 1] and is invariant to both donut SIZE and BRIGHTNESS.
        A small noise floor (energyFloorFactor * median(eLoc)) is added to the
        denominator so flat/empty regions (tiny eLoc) can't yield spuriously
        high scores.

        Alongside it, the matched-filter SNR

            snr = dot / (||t|| * pixelNoise)

        which is exact for white noise of per-pixel sigma `pixelNoise`, since
        Var(dot) = sigma^2 * sum(t^2) = (sigma * ||t||)^2. Unlike rho, this
        KEEPS the amplitude information -- see the class docstring for why both
        are needed.

        Parameters
        ----------
        pixelNoise : float or None
            Robust per-binned-pixel noise sigma. None (or non-finite) yields
            `snr = None`, for callers that only want the shape score.

        Yields
        ------
        i : int
            Index into bank.diameters.
        rho : np.ndarray
            Normalized cross-correlation response for diameter i.
        snr : np.ndarray or None
            Matched-filter SNR for diameter i, or None if pixelNoise is None.
        """
        haveNoise = pixelNoise is not None and np.isfinite(pixelNoise) and pixelNoise > 0

        innerIdx = bank.innerIndex
        outerIdx = bank.outerIndex
        annulusNorm = bank.annulusNorm
        bgWeight = bank.bgWeight
        supportArea = bank.supportArea

        eps = 1e-12
        floorFrac = self.config.energyFloorFactor

        for i in range(len(innerIdx)):
            ii = innerIdx[i]
            if ii < 0 or annulusNorm[i] <= 0:
                continue

            oi = outerIdx[i]

            # Annulus (signal) sum
            sumS = diskCorr[i] - diskCorr[ii]

            # Background sum, and the support set U = signal + background.
            if oi >= 0:
                sumB = diskCorr[oi] - sumS
                sumU = diskCorr[oi]        # <image,   U>
                sumsqU = diskCorrSq[oi]    # <image^2, U>
            else:
                sumB = diskCorr[ii]
                sumU = diskCorr[i]         # signal + inner hole
                sumsqU = diskCorrSq[i]

            # Numerator: proper dot product with the zero-mean template t.
            dot = sumS - bgWeight[i] * sumB

            # Local mean-removed image energy under U:
            #   var = <image^2, U> - <image, U>^2 / A_U   (proportional to
            #   local variance * A_U); eLoc = sqrt(var) = ||image_patch||.
            A_U = supportArea[i]
            var = sumsqU - (sumU * sumU) / A_U
            eLoc = np.sqrt(np.maximum(var, 0.0))

            # Noise floor: fraction of the median local energy for this diameter.
            if floorFrac > 0:
                floor = floorFrac * np.median(eLoc[np.isfinite(eLoc)])
            else:
                floor = 0.0

            denom = annulusNorm[i] * (eLoc + floor) + eps
            rho = dot / denom

            # Matched-filter SNR: same numerator, but normalized by the noise
            # rather than by the local image energy -- keeps the amplitude.
            snr = dot / (annulusNorm[i] * pixelNoise) if haveNoise else None

            yield i, rho, snr

    def _likenessMap(self, diskCorr, diskCorrSq, bank, pixelNoise=None):
        """Scale- and brightness-invariant donut-likeness map (max over diameters
        of the normalized cross-correlation), plus the matched-filter SNR at the
        winning diameter.

        Also records, per pixel, which diameter won (argDiameter), used by
        _selectPeaks for per-peak declustering. The per-diameter rho maps are
        retained (rhoMaps / rhoDiametersBinned) so the sizing step can reuse them
        rather than recomputing the whole _annulusTerms pass per peak.

        Returns
        -------
        lsst.pipe.base.Struct
            likeness : np.ndarray
                max over diameters of rho -- the detection score.
            snrAtWinner : np.ndarray
                Matched-filter SNR at the diameter that won the likeness max,
                i.e. the amplitude that goes with the reported score. All-NaN if
                pixelNoise is unusable.
            argDiameter : np.ndarray
                Winning diameter per pixel, in BINNED px (0 where nothing won).
            rhoMaps : list[np.ndarray]
                Per-diameter rho response maps, one per VALID diameter (those
                with an inner disk), in ladder order. Reused by _sizingCurveAt.
            rhoDiametersBinned : np.ndarray
                Binned diameters aligned with rhoMaps.
        """
        shape = diskCorr[0].shape
        likeness = np.full(shape, -np.inf)
        snrAtWinner = np.full(shape, np.nan)
        argDiameter = np.zeros(shape, dtype=float)

        rhoMaps = []
        rhoDiametersBinned = []
        for i, rho, snr in self._annulusTerms(diskCorr, diskCorrSq, bank, pixelNoise):
            rhoMaps.append(rho)
            rhoDiametersBinned.append(bank.diameters[i])
            better = rho > likeness
            argDiameter = np.where(better, bank.diameters[i], argDiameter)
            likeness = np.where(better, rho, likeness)
            if snr is not None:
                snrAtWinner = np.where(better, snr, snrAtWinner)

        return pipeBase.Struct(
            likeness=likeness,
            snrAtWinner=snrAtWinner,
            argDiameter=argDiameter,
            rhoMaps=rhoMaps,
            rhoDiametersBinned=np.array(rhoDiametersBinned),
        )

    def _selectPeaks(self, likeness, bank, snrAtWinner=None, argDiameter=None,
                     nPeaks=None, minSeparationFactor=None):
        """Candidate (y, x) peaks in the binned likeness map, best-first.

        Local maxima, declustered by a per-peak minimum separation scaled to the
        diameter that won at each peak. Two peaks are considered distinct only if
        their squared separation exceeds the LARGER of the two peaks' squared
        separation radii (max(sepSq, cSepSq)) -- so a large donut cannot sit
        immediately beside a small one (the large radius dominates), while two
        small donuts may legitimately be close.

        Quality cuts (likeness and matched-filter SNR thresholds) are applied
        here, inside the selection loop, rather than afterwards: a candidate
        failing either cut is discarded and the descent through the sorted maxima
        continues, so up to `nPeaks` genuine survivors can still be returned even
        when high-SNR/low-likeness or high-likeness/low-SNR features are pruned.
        A peak must pass BOTH cuts (they fail differently -- see class docstring).
        Either cut is disabled by setting its threshold <= 0; the SNR cut is
        additionally skipped for any peak whose SNR is non-finite (e.g. when the
        pixel-noise estimate failed).

        Parameters
        ----------
        likeness : np.ndarray
            Binned likeness map.
        bank : DiskCorrelationBank
            The correlation bank (for the max diameter fallback).
        snrAtWinner : np.ndarray or None
            Matched-filter SNR at the winning diameter, aligned with `likeness`.
            If None, the SNR cut is skipped entirely.
        argDiameter : np.ndarray or None
            Winning diameter per pixel (binned px), aligned with `likeness`,
            used to scale each peak's declustering radius. If None, the max
            ladder diameter is used for every peak.
        nPeaks : int or None
            Maximum number of peaks to return (defaults to config.nPeaks).
        minSeparationFactor : float or None
            Per-peak separation scale (defaults to
            config.peakMinSeparationFactor).

        Returns
        -------
        list[tuple[int, int]]
            Selected (y, x) peaks in the (uncropped) binned frame, best-first.
        """
        if nPeaks is None:
            nPeaks = self.config.nPeaks
        if minSeparationFactor is None:
            minSeparationFactor = self.config.peakMinSeparationFactor

        likeThresh = self.config.likenessThreshold
        snrThresh = self.config.snrThreshold

        clip = self.config.likenessEdgeClip
        argD = argDiameter
        if clip > 0:
            likeness = likeness[clip:-clip, clip:-clip]
            if argD is not None:
                argD = argD[clip:-clip, clip:-clip]
            if snrAtWinner is not None:
                snrAtWinner = snrAtWinner[clip:-clip, clip:-clip]

        finite = np.isfinite(likeness)
        if not finite.any():
            return []

        localMax = maximum_filter(likeness, size=3)
        peakMask = (likeness == localMax) & finite
        ys, xs = np.where(peakMask)
        if len(ys) == 0:
            return []

        order = np.argsort(likeness[ys, xs])[::-1]
        ys, xs = ys[order], xs[order]

        if argD is not None:
            dpk = argD[ys, xs]
            dpk = np.where(dpk > 0, dpk, bank.diameters.max())
        else:
            dpk = np.full(len(ys), bank.diameters.max())

        chosen: list[tuple[int, int]] = []
        chosenSepSq: list[float] = []
        nRejLike = 0
        nRejSnr = 0
        for k in range(len(ys)):
            y, x = ys[k], xs[k]

            # Quality cuts: prune here so the descent can keep filling to nPeaks.
            if likeThresh > 0 and likeness[y, x] < likeThresh:
                nRejLike += 1
                continue
            if snrThresh > 0 and snrAtWinner is not None:
                snr = snrAtWinner[y, x]
                # Only reject on a usable (finite) SNR; skip the cut otherwise.
                if np.isfinite(snr) and snr < snrThresh:
                    nRejSnr += 1
                    continue

            sepSq = (minSeparationFactor * dpk[k]) ** 2
            ok = True
            for (cy, cx), cSepSq in zip(chosen, chosenSepSq):
                d2 = (y - cy) ** 2 + (x - cx) ** 2
                if d2 < max(sepSq, cSepSq):
                    ok = False
                    break
            if ok:
                chosen.append((int(y), int(x)))
                chosenSepSq.append(sepSq)
            if len(chosen) >= nPeaks:
                break

        if nRejLike or nRejSnr:
            self.log.info(
                "Quality cuts rejected %d candidates (%d likeness < %.2f, "
                "%d SNR < %.1f); %d peaks selected",
                nRejLike + nRejSnr, nRejLike, likeThresh,
                nRejSnr, snrThresh, len(chosen),
            )

        chosen = [(y + clip, x + clip) for y, x in chosen]
        return chosen

    def _sizingCurveAt(self, y, x, rhoMaps, rhoDiametersBinned, window=1):
        """Likeness sizing curve at a pixel: cosine response vs. diameter.

        Reads the per-diameter rho response maps precomputed by _likenessMap and
        samples them at (y, x), so no correlation algebra is repeated here. The
        curve's peak locates the donut size while rejecting wrong-shape features:
        a ring of small donuts whose mean-flux profile would peak at a large
        (wrong) diameter does NOT produce a matching likeness peak there.

        The value reported at each diameter is the local maximum over a
        (2*window + 1) box about (y, x), to tolerate small centroid offsets
        between the peak and the true donut center.

        Parameters
        ----------
        y, x : int
            Peak location in the (uncropped) binned frame.
        rhoMaps : list[np.ndarray]
            Per-diameter rho response maps from _likenessMap, in ladder order.
        rhoDiametersBinned : np.ndarray
            Binned diameters aligned with rhoMaps.
        window : int
            Half-width of the local-max box (0 = the pixel itself).

        Returns
        -------
        ds : np.ndarray
            Diameters (binned px) at which the curve is defined.
        vals : np.ndarray
            The cosine response at each diameter in `ds`.
        """
        def localMax(amap):
            if window <= 0:
                return amap[y, x]
            y0, y1 = max(0, y - window), min(amap.shape[0], y + window + 1)
            x0, x1 = max(0, x - window), min(amap.shape[1], x + window + 1)
            return amap[y0:y1, x0:x1].max()

        vals = np.array([localMax(rho) for rho in rhoMaps])
        return np.asarray(rhoDiametersBinned, dtype=float), vals

    @staticmethod
    def _peakDiameter(diametersBinned, curve):
        """Diameter at the curve peak, parabolic sub-grid refinement in
        log-diameter. NaN if fewer than 3 points; grid value if peak is at an
        edge (unrefinable). The refinement offset is clamped to [-1, 1] grid
        steps so a near-flat noisy triplet can't extrapolate wildly.
        """
        if len(curve) < 3:
            return np.nan
        k = int(np.argmax(curve))
        if k == 0 or k == len(curve) - 1:
            return float(diametersBinned[k])
        lx = np.log(diametersBinned[k - 1:k + 2])
        ly = curve[k - 1:k + 2]
        denom = ly[0] - 2 * ly[1] + ly[2]
        if denom == 0:
            return float(diametersBinned[k])
        delta = np.clip(0.5 * (ly[0] - ly[2]) / denom, -1.0, 1.0)
        logStep = lx[2] - lx[1]
        return float(np.exp(lx[1] + delta * logStep))

    def _exposureDiameter(self, peaks, rhoMaps, rhoDiametersBinned):
        """Single exposure diameter (full-res px) from the pooled sizing curve,
        plus per-detection scatter and the underlying curves for inspection.

        Since the exposure has one true diameter, per-detection curves are noisy
        realizations of one curve; the median across detections at each diameter
        gives a robust combined curve whose single peak is the answer.

        Sizing reuses the per-diameter rho maps from _likenessMap (rhoMaps /
        rhoDiametersBinned) rather than recomputing them per peak.

        Returns
        -------
        lsst.pipe.base.Struct
            diameter : float
                Exposure diameter in full-res px (NaN if unmeasurable).
            scatter : float
                Per-detection fractional scatter (NaN if < 2 detections).
            perDetDiameter : np.ndarray
                Per-peak curve diameter in full-res px, ALIGNED WITH `peaks`
                (NaN for peaks whose curve was unusable).
            argmaxAtEdge : np.ndarray of bool
                Per-peak flag, aligned with `peaks`: the sizing curve's argmax
                sits on a ladder endpoint, i.e. the curve is monotonic over the
                probed range and carries no interior scale. True for pure-noise
                peaks, which pile up at a ladder endpoint. False for peaks with
                no curve.
            curveInfo : dict
                diametersFull / perDetCurves / combined / peaksUsed, where the
                curve lists cover only the USED peaks (see peaksUsed).
        """
        binning = self.config.detectionBinning
        nPk = len(peaks)
        curves = []
        peaksUsed = []
        perDetAligned = np.full(nPk, np.nan)
        edgeAligned = np.zeros(nPk, dtype=bool)
        dGrid = None
        for k, (y, x) in enumerate(peaks):
            dBin, sizing = self._sizingCurveAt(
                y, x, rhoMaps, rhoDiametersBinned, window=1
            )
            if len(sizing) < 3:
                continue
            if dGrid is None:
                dGrid = dBin
            elif len(sizing) != len(dGrid):
                continue
            curves.append(sizing)
            peaksUsed.append((y, x))
            perDetAligned[k] = self._peakDiameter(dBin, sizing) * binning
            kMax = int(np.argmax(sizing))
            edgeAligned[k] = kMax == 0 or kMax == len(sizing) - 1

        curveInfo = {
            "diametersFull": (dGrid * binning) if dGrid is not None else None,
            "perDetCurves": curves,
            "combined": None,
            "peaksUsed": peaksUsed,
        }

        result = pipeBase.Struct(
            diameter=np.nan,
            scatter=np.nan,
            perDetDiameter=perDetAligned,
            argmaxAtEdge=edgeAligned,
            curveInfo=curveInfo,
        )
        if not curves:
            return result

        combined = np.median(np.vstack(curves), axis=0)
        curveInfo["combined"] = combined

        result.diameter = self._peakDiameter(dGrid, combined) * binning

        finite = perDetAligned[np.isfinite(perDetAligned)]
        if len(finite) >= 2:
            result.scatter = np.std(finite) / np.median(finite)
        return result

    def run(self, exposure: Exposure) -> pipeBase.Struct:
        binnedImage, bbox = self._prepImage(exposure)
        bank = self._getBank(binnedImage.shape)
        pixelNoise = _shiftSubtractNoise(binnedImage)

        # Image needs both moments (for energy normalization); mask needs only
        # the first moment (saves one rfftn + one irfftn per diameter).
        diskCorr, diskCorrSq = bank.correlate(binnedImage, secondMoment=True)
        maps = self._likenessMap(diskCorr, diskCorrSq, bank, pixelNoise)
        likeness = maps.likeness

        peaks = self._selectPeaks(
            likeness,
            bank,
            snrAtWinner=maps.snrAtWinner,
            argDiameter=maps.argDiameter
        )
        sizing = self._exposureDiameter(
            peaks, maps.rhoMaps, maps.rhoDiametersBinned
        )

        diameterFull = sizing.diameter
        scatter = sizing.scatter
        perDetFull = sizing.perDetDiameter
        curveInfo = sizing.curveInfo
        if np.isfinite(scatter):
            self.log.info(
                "Exposure diameter %.1f px (full); per-detection scatter "
                "%.1f%% over %d donuts", diameterFull, scatter * 100,
                len(curveInfo["peaksUsed"]),
            )

        # Binned, edge-trimmed (y, x) -> full-exposure pixels.
        # A bin covers [i*b, (i+1)*b); its center is i*b + (b-1)/2, plus origin.
        binning = self.config.detectionBinning
        half = (binning - 1) / 2.0
        minx, miny = bbox.getMinX(), bbox.getMinY()
        if peaks:
            ys = np.array([p[0] for p in peaks], dtype=float)
            xs = np.array([p[1] for p in peaks], dtype=float)
            cx = binning * xs + half + minx
            cy = binning * ys + half + miny
        else:
            cx = np.array([])
            cy = np.array([])

        # Per-peak QC diagnostics. The likeness/SNR cuts were already applied in
        # _selectPeaks; these columns report the surviving peaks' scores (plus
        # the no-cut curve_argmax_edge flag). See class docstring.
        ys = [p[0] for p in peaks]
        xs = [p[1] for p in peaks]
        detections = QTable(
            {
                "id": np.arange(1, len(peaks) + 1, dtype=np.int64),
                "centroid_x": cx,
                "centroid_y": cy,
                "diameter": perDetFull,
                "likeness": np.array([likeness[y, x] for y, x in zip(ys, xs)]),
                "snr": np.array([maps.snrAtWinner[y, x] for y, x in zip(ys, xs)]),
                "peak_diameter": np.array(
                    [maps.argDiameter[y, x] for y, x in zip(ys, xs)]
                ) * binning,
                "curve_argmax_edge": sizing.argmaxAtEdge,
            }
        )

        return pipeBase.Struct(
            detections=detections,
            diameter=diameterFull,
            diameterScatter=scatter,
            sizingCurves=curveInfo,
            binnedImage=binnedImage,
            likeness=likeness,
            snr=maps.snrAtWinner,
            argDiameter=maps.argDiameter,
            pixelNoise=pixelNoise,
            diskCorr=diskCorr,
            diskCorrSq=diskCorrSq,
            diametersBinned=bank.diameters,
            diametersFull=bank.diameters * binning,
            bbox=bbox,
            binning=binning,
        )