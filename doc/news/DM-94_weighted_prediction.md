Added uncertainty-weighted averaging of Zernike predictions in AiDonut.
When the model returns per-prediction error estimates, predictions with lower
estimated error receive higher weight via softmax weighting with a configurable
temperature parameter.
