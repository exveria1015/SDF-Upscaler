# Adaptive Color-SDF RGB

Status: experimental HAT-free RGB backend.

## Goal

Alpha has one scalar inside/outside boundary, while RGB has shared geometry
with vector-valued colours on both sides of each interface. Adaptive Color-SDF
extends the successful SAL/Adaptive-SDF constraints to RGB without treating
the R, G, and B channels as unrelated masks.

The x4 path is:

1. SAL-Classic linear-light premultiplied RGB base;
2. hidden-RGB-invariant visible-colour extension;
3. the existing 11-channel RGB structure field;
4. signed-distance reconstruction from colours sampled on both sides of an
   internal RGB contour;
5. a capped 3x3 linear-light median residual in opaque pixels;
6. iterative premultiplied-RGB back-projection;
7. a local LR no-regression gate and whole-image LR fallback.

Partial-alpha RGB remains exactly on the SAL-Classic path. The selected alpha
plane is never modified by the RGB backend.

## CLI

The deterministic alpha route is the default:

```bash
uv run sdf-upscale input.png \
  --backend adaptive \
  --output output.png
```

Learned-alpha checkpoint orchestration remains in the source Upscaler
repository. The standalone Python API accepts a supplied normalized
`alpha_hr` without depending on Torch.

The first profile exposes structure, median, and back-projection strengths as
CLI ablations. Defaults are deliberately bounded in linear RGB.

## First self-downsample result

The first exact x4 diagnostic used the supplied 1082x1650 RGBA character. One
transparent pixel was cropped from every side to obtain a divisible 1080x1648
GT. A 270x412 LR input was generated with linear-light premultiplied Lanczos,
then reconstructed to the GT dimensions. Both RGB routes used the exact same
M-alpha-1 output.

| Metric | SAL-Classic RGB + M-alpha-1 | Adaptive Color-SDF + M-alpha-1 |
|---|---:|---:|
| Composite PSNR, five backgrounds | 27.38445 dB | **27.73078 dB** |
| Composite SSIM, five backgrounds | 0.941196 | **0.945156** |
| Linear premultiplied RGB MAE | 0.0111713 | **0.0105185** |
| Linear premultiplied RGB PSNR | 28.73944 dB | **29.11824 dB** |
| LR premultiplied reprojection MAE | 0.0022723 | **0.0015974** |
| Visible straight-RGB MAE, 8-bit | 7.36884 | **6.99439** |

The candidate changes 15.61% of RGB pixels with 1.51/255 mean absolute delta.
Its manifest reports no whole-image reprojection fallback.

This is one self-downsample image, not a general quality claim. Promotion
requires a held-out RGBA/vector-art set, topology and thin-line guards, and
neutral results on gradients, text, anime, and photographic textures.

## Web UI x4 screening

An independent first screen used 12 nonblank 768x768 crops from six pages in
`GoofyGoof/Web_UI`. Each 192x192 LR was generated with the same linear-light
Lanczos x4 degradation used by the baseline. Since these screenshots are
opaque, SAL-Classic has no alpha boundary to refine and is exactly its
linear-light Lanczos RGB base. Parameters were not tuned on these crops.

| Metric, mean over 12 crops | SAL / linear Lanczos | Adaptive Color-SDF | Paired delta |
|---|---:|---:|---:|
| RGB PSNR | 25.7915 dB | **26.0549 dB** | **+0.2634 dB** |
| Y PSNR | 27.2231 dB | **27.4510 dB** | **+0.2278 dB** |
| RGB SSIM | 0.887867 | **0.890831** | **+0.002964** |
| Edge Y PSNR | 18.6172 dB | **18.8317 dB** | **+0.2146 dB** |
| Edge-gradient L1, 8-bit | 3.01464 | **2.96625** | **-0.04838** |
| Linear-Lanczos LR round-trip L1, 8-bit | 0.95961 | **0.63454** | **-0.32506** |

Adaptive improves RGB PSNR and edge Y PSNR on all 12 crops. The paired
10,000-resample bootstrap 95% intervals exclude zero for RGB PSNR
(`[+0.2001, +0.3226]` dB), edge Y PSNR (`[+0.1550, +0.2767]` dB),
edge-gradient L1 improvement (`[+0.0223, +0.0743]`), and LR round-trip L1
improvement (`[+0.2148, +0.4559]`). RGB SSIM improves on 10/12 crops.

The output is a bounded anti-ringing/edge reconstruction improvement, not
semantic text recovery. The source screenshots are JPEG pseudo-GT and only six
pages are represented, so the result is encouraging screening evidence rather
than a general UI benchmark. Raw metrics and the visual comparison live under
`outputs/evaluation/webui_goofygoof_x4/`.
