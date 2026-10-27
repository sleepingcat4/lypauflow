# LyapuFlow for images and audio

A PyTorch implementation of [LyapuFlow: Controlling Generative Flows with Lyapunov Feedback for Inverse Problems](https://arxiv.org/abs/2610.04326), with a waveform audio extension.

**Status:** independent implementation of the controller, not the authors' repository or a verified reproduction of their benchmark scores. The paper studies PDEs and images; the audio tasks here are new applications. No trained weights or datasets are bundled. The small convolutional priors must be trained on your data. The optional image adapter loads SD3.5-Medium.

## Install

Python 3.10+ and PyTorch 2.2+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

For the optional SD3 image backbone:

```bash
pip install -e '.[sd3]'
```

SD3.5 weights require Hugging Face access under the model's terms. Authenticate through your existing Hugging Face setup. CUDA and enough GPU memory for the transformer, text encoders, VAE, and input gradients are recommended. Full derivatives cost more memory than stop-gradient sampling. CPU operation is supported by the adapter but expensive.

## Run without datasets or pretrained weights

```bash
python -m lyapuflow demo --domain image --task superres --output runs/image-demo
python -m lyapuflow demo --domain audio --task bandwidth --output runs/audio-demo
```

These use an analytic Gaussian flow, with synthetic references used only to create measurements and compute metrics. They exercise the controller and file pipeline; they do not demonstrate perceptual reconstruction with a learned prior. Missing pixels/frequencies cannot be inferred meaningfully from that Gaussian prior.

Each reconstruction writes a reference, guided reconstruction, unguided sample from the same initial noise, raw reconstruction and observation tensors, per-step feedback diagnostics, and `metrics.json`. Image outputs are PNG and audio outputs are PCM WAV. Metrics use the raw floating-point reconstruction before clipping for media export.

## Train an image prior and reconstruct

Put training images in a directory (nested folders work):

```bash
python -m lyapuflow train --domain image --data data/images \
  --size 64 --iterations 10000 --batch-size 8 --device cuda \
  --output runs/image-prior.pt
python -m lyapuflow reconstruct --domain image \
  --checkpoint runs/image-prior.pt --input data/test.png \
  --task superres --factor 4 --device cuda --output runs/image-sr
```

Image tasks: `superres` (antialiased bicubic), `gaussian`, `motion`, `inpaint`, `denoise`.
`--kernel-size`, `--blur-sigma`, and `--angle` configure deblurring. Images are RGB, resized and center-cropped to the training resolution, and normalized to [-1,1]. Train/test images should be disjoint. The compact dilated convolutional model is a baseline, not SD3 or an architecture claimed to reproduce the paper's image results.

## Train an audio prior and reconstruct

Place WAV files in a directory. Training uses random fixed-length crops, mono conversion, and resampling; inference takes the first segment and pads short inputs.

```bash
python -m lyapuflow train --domain audio --data data/wavs \
  --sample-rate 16000 --length 16384 --iterations 10000 \
  --batch-size 8 --device cuda --output runs/audio-prior.pt
python -m lyapuflow reconstruct --domain audio \
  --checkpoint runs/audio-prior.pt --input data/test.wav \
  --task bandwidth --cutoff 3000 --sample-rate 16000 \
  --device cuda --output runs/audio-bandwidth
```

Audio tasks:

- `bandwidth`: windowed-sinc low-pass observation at the original sample rate.
- `superres`: low-pass filtering plus decimation (`--factor 4` by default).
- `inpaint`: a missing central third of the segment.
- `denoise`: identity measurement with additive noise.
- `dereverb`: known causal room impulse response. Pass `--ir room.wav`; otherwise a documented synthetic four-tap room is used.

Audio stays in waveform space; no nondifferentiable vocoder is involved. Training uses normalized PCM amplitudes without per-clip loudness normalization. The sample rate must match the checkpoint. Reported audio metrics are MSE, SI-SDR, and measurement RMSE. They are not claims about perceptual speech quality. Entire long files require a separate overlapping-window pipeline; this CLI intentionally reconstructs one segment.

## Pretrained SD3.5 image reconstruction

```bash
python -m lyapuflow sd3 --input data/cat.png \
  --prompt 'a photo of a cat' --task superres --size 768 \
  --steps 28 --alpha 5 --trust-region 1 --switch-step 28 \
  --guidance 2 --device cuda --output runs/sd3-cat
```

For 28-step Gaussian deblurring use `--task gaussian --switch-step 3`; for motion deblurring use `--task motion --switch-step 7`. The default blur kernel is 61 pixels and default observation noise is 0.03. Table 10 contains dataset-specific choices (including alpha=7 for DIV2K motion at 28 steps).

The adapter converts the diffusion convention to noise-to-data flow time (`t=1-sigma`, `v=-prediction`), uses classifier-free guidance, and differentiates through the VAE decoder. Sampling uses uniform flow time; exact scheduler, Gaussian sigma, and motion PSF details are not fully specified by the paper. Here Gaussian sigma is configurable (default 3), the motion kernel is a straight line, and blur boundaries replicate pixels. These choices can affect scores. No AFHQ/FFHQ/DIV2K benchmark, FID, LPIPS, or SSIM reproduction has been run. The optional adapter requires actual weights and is tested structurally with a small fake pipeline, not a downloaded SD3.5 model.

## Use actual measurements or a custom pretrained flow

The CLI normally simulates measurements from `--input`, then reports reference-based metrics. To supply a real observation, pass `--observation measurement.pt`, a tensor saved with `torch.save`, shaped `[1,C,...]` in the operator's output space. `--input` is still a shape/reference template and metrics are meaningful only if it is a matching clean reference. For masks, the CLI uses its fixed central-hole mask; use the API for arbitrary masks, arbitrary audio lengths, custom encoders/decoders, or reference-free inference:

```python
import torch
from lyapuflow import ControllerConfig, LyapuFlow
from lyapuflow.operators import Measurement, ImageSuperResolution

# pretrained_velocity(z, t): t is [B], increasing from noise (0) to data (1).
# differentiable_decoder(z): decoded images [B,C,H,W], or waveforms [B,C,T].
residual = Measurement(ImageSuperResolution(4), observed_image)
sampler = LyapuFlow(
    pretrained_velocity, residual,
    ControllerConfig(steps=28, alpha=5, trust_region=1, switch_step=28),
    decoder=differentiable_decoder,
)
result = sampler.sample(initial_noise)
reconstruction = result.sample
```

For data-space flow use no decoder. Models and decoders should be in `eval()` mode; freeze their parameter gradients with `requires_grad_(False)` while keeping input autograd enabled. The API does not change the caller's model mode or device. Every function must preserve the batch axis and each sample must be independent of other samples. Do not use batch-coupled normalization or cross-sample residuals. A residual returns `[B,...]`, at least two axes. The loss is half the **sum** of squared residuals per example, not a batch average. The observation must match the model/decoder's data normalization.

## Algorithm and numerical behavior

The implementation maps to the paper as follows:

| Component | Implementation |
|---|---|
| Endpoint prediction and residual energy | `controller.py`, Eqs. 5–6 |
| Full state and explicit time derivatives | Autograd through endpoint, model, decoder; Algorithm 1 |
| Minimum-norm half-space control | Positive-part feedback, Eq. 11 |
| Flow-relative trust region | Per-example L2 norm bound, Eqs. 17–18 |
| Latent guidance and gradient switching | Decoder gradients and `switch_step`, Algorithm 2 |
| Euler integration | Explicit noise-to-data time grid, Eq. 14 |
| Training | Straight-line conditional flow matching; supporting baseline |

`switch_step=0` enables full derivatives at every step. `switch_step=steps` detaches velocity at every step; both its state and time dependence are removed, and feedback reduces to Eq. 22. Intermediate values restore full derivatives later. A custom strictly increasing time grid from 0 to 1 can be supplied to `sample(times=...)`.

Gradients whose squared norm is <= `gradient_epsilon` receive zero control. A stationary nominal flow also has zero allowed control under a finite relative trust region. `trust_region=None` disables clipping in the API. Clipping, degenerate gradients, finite Euler steps, and the stop-gradient approximation can prevent the requested energy decrease; it is not guaranteed for every discrete step. History records energy, nominal energy drift, norms, clipping/degeneracy flags, and remaining continuous-time slack. In stop-gradient mode this slack is for the surrogate derivatives. History uses detached CPU tensors and sampling retains no trajectory graph. Do not wrap sampling in `torch.inference_mode()`; input autograd is essential.
