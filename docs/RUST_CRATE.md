# Rust crate

Status: canonical Alpha-SDF, Color-SDF, and Adaptive-SDF x4 routes are
implemented in pure Rust with Python-reference quantized parity.

## Workspace

```text
crates/
  sdf-upscaler/       # dependency-light slice-based library
  sdf-upscale-cli/    # image decoding, PNG writing, and sdf-upscale binary
```

The workspace uses Rust 2024 and declares MSRV 1.94. Publishing remains
disabled until the repository owner selects a license.

The parity routes do not call Python, NumPy, Pillow, SciPy, or scikit-image at
runtime. Private compatibility kernels reproduce the reference stack's
observable resize, filtering, contour, nearest-neighbor, EDT, Canny, and
skeleton behavior.

## Library API

```rust
use sdf_upscaler::{
    AdaptiveOptions, AlphaInput, ColorOptions, PlaneRef, Rgba8Ref,
    upscale_adaptive_sdf_x4, upscale_alpha_sdf_x4, upscale_color_sdf_x4,
};

let alpha = PlaneRef::packed(width, height, &alpha_u8)?;
let alpha_x4 = upscale_alpha_sdf_x4(AlphaInput::U8(alpha))?;

let rgba = Rgba8Ref::packed(width, height, &rgba_u8)?;
let color_x4 = upscale_color_sdf_x4(rgba, &ColorOptions::default())?;
let adaptive_x4 =
    upscale_adaptive_sdf_x4(rgba, None, &AdaptiveOptions::default())?;
```

`PlaneRef` and `Rgba8Ref` validate nonzero dimensions, stride, buffer length,
and overflow. Floating alpha rejects NaN and infinity. Alpha input accepts
`u8` or normalized `f32`; RGBA input is packed straight-sRGB `u8`.

Adaptive-SDF accepts an optional x4 `PlaneRef<f32>`. Finite supplied values are
clipped to `[0, 1]`, used directly by the RGB reconstruction, and preserved
after ties-to-even `u8` quantization.

### Canonical routes

- `upscale_alpha_sdf_x4` implements the full Alpha-SDF contour, support,
  calibration, topology, and SCS route.
- `upscale_color_sdf_x4` implements SAL-Classic RGB over the canonical alpha
  segment graph with a linear-light premultiplied fallback.
- `upscale_adaptive_sdf_x4` adds hidden-RGB-invariant structure extraction,
  edge-local structure fusion, opaque-interior median residuals,
  premultiplied back-projection, local reprojection gating, and a whole-image
  no-regression fallback.

The earlier ROI Alpha implementation is retained as
`upscale_alpha_sdf_x4_experimental`; its diagnostics continue to identify it
as experimental rather than canonical parity.

## CLI

Adaptive-SDF is the default:

```bash
cargo run --release -p sdf-upscale-cli -- \
  input.png --output output.png
```

Select another backend explicitly:

```bash
cargo run --release -p sdf-upscale-cli -- \
  input.png --output color.png --backend color

cargo run --release -p sdf-upscale-cli -- \
  input.png --output alpha.png --backend alpha
```

Directory mode and overwrite protection are available:

```bash
cargo run --release -p sdf-upscale-cli -- \
  --dir assets --output outputs --backend adaptive --overwrite
```

Color and Adaptive write RGBA PNGs; Alpha writes grayscale PNGs. Each output
contains `sdf-upscaler.version` and `sdf-upscaler.pipeline` text chunks and a
JSON sidecar with the selected route's summary diagnostics. Decoder-level EXIF
orientation is not currently applied.

## Frozen parity gates

Raw cross-language fixtures live in `fixtures/parity/alpha`,
`fixtures/parity/color`, and `fixtures/parity/adaptive`. Generator manifests
record shapes, dtypes, reference versions, options, and SHA-256 values.

| Route | Fixed quantized output SHA-256 |
|---|---|
| Alpha-SDF alpha u8 | `777276c9bbee3a9ebd57b6ae9ae80e178280d6dc0fea6ace970347b4f95d4bd3` |
| Color-SDF RGBA u8 | `f00ce3c04297942c59ad8b7986cb781151a8a3a22374ce1d1484e8beee10219c` |
| Adaptive-SDF RGBA u8 | `289a9a0f6444883306fee5d7599ce0af3d34244780515e25409bcac91fb38074` |

Acceptance additionally covers:

- constant, hard-shape, broad-soft, line-art, partial-alpha, holes, open
  contours, odd-size, and SCS Alpha cases;
- flat, chroma-split, partial-alpha, transparent, line-art, odd-size, and
  hidden-RGB Color cases;
- hidden-RGB pairs, opaque, partial-alpha, transparent, line-art, odd-size,
  and supplied-alpha Adaptive cases;
- exact 8-bit Pillow Lanczos behavior and fixed structure edge masks,
  component labels, 11-channel features, and structure-fusion output;
- `lr_error_final <= lr_error_base + 1e-9` and no fallback on the Adaptive
  fixed fixture.

The public quality claim is quantized Python-reference parity. A few internal
Alpha and diagnostic `f32` planes can differ by platform-level ULPs while all
gated quantized outputs remain exact.

## Validation

```bash
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo test --workspace --all-targets --release
cargo clippy --workspace --all-targets -- -D warnings
```
