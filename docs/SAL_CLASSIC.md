# SAL-Classic x4

Deterministic, HAT-free RGBA x4 upscaling.

## Pipeline

- Alpha: Lanczos base → Segment-SDF → adaptive Lanczos residual → calibration/support gates → conditional SCS
- RGB: linear-light premultiplied Lanczos → alpha-edge segment-guided normalized Lanczos residual
- Output: straight RGBA PNG

The RGB residual uses only visible LR samples (`alpha > 0`), restricts support to
the same contour and local arc neighborhood, rejects ill-conditioned signed
Lanczos sums, and clamps the candidate to the local linear-RGB range. Fully
opaque interiors remain the deterministic Lanczos baseline.

## CLI

```bash
sdf-upscale input.png \
  --backend color \
  -o output_4x.png
```

Useful bounded ablations are exposed as `--color-strength` and
`--color-edge-band`. The lower-level Python API accepts the complete
`rgb_options` mapping.

## Python

```python
from PIL import Image
import numpy as np
from sdf_upscaler import upscale_color_sdf_x4

rgba = np.asarray(Image.open("input.png").convert("RGBA"), dtype=np.uint8)
output, diagnostics = upscale_color_sdf_x4(rgba)
Image.fromarray(output).save("output_4x.png")
```
