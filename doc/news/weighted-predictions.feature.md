Added uncertainty-weighted averaging of Zernike predictions in AiDonutAlgorithm.
When the model returns per-prediction error estimates, predictions with lower
estimated error receive higher weight via softmax weighting with a configurable
temperature parameter.
**Breaking change:** ``AiDonutAlgorithm`` now interprets a 2-tuple model
output as ``(zk, zkScore)`` instead of ``(zk, fwhm)``. Models that
previously returned ``(zk, fwhm)`` must be updated to return either
``zk`` alone or the 3-tuple ``(zk, zkScore, fwhm)``.
