# SDF-Upscaler

Deterministic x4 upscaling for RGBA assets and structured RGB:

- **Alpha-SDF** reconstructs bounded transparency coverage.
- **Color-SDF** uses the SAL-Classic linear-light premultiplied base and a
  contour-constrained RGB residual.
- **Adaptive-SDF** adds bounded structure and median residuals with LR
  back-projection guards.

The package does not require Torch, HAT checkpoints, or a GPU. Its current
scope is fixed x4 reconstruction with NumPy arrays and PNG-oriented CLI I/O.
It is a research implementation, not a claim of state-of-the-art natural-image
super-resolution.

## Install locally

```bash
uv sync
```

No project license has been selected yet. Do not publish or redistribute this
repository until the owner adds one.

## CLI

```bash
# Adaptive-SDF RGBA x4
uv run sdf-upscale input.png -o output.png

# Color-SDF / SAL-Classic RGBA x4
uv run sdf-upscale input.png -o output.png --backend color

# Alpha-SDF only; writes a grayscale alpha PNG
uv run sdf-upscale alpha.png -o alpha_x4.png --backend alpha

# Recursively process a directory
uv run sdf-upscale --dir assets -o outputs --backend adaptive
```

Existing output files require `--overwrite`.

## Python API

```python
from sdf_upscaler import (
    upscale_adaptive_sdf_x4,
    upscale_alpha_sdf_x4,
    upscale_color_sdf_x4,
)

alpha_hr = upscale_alpha_sdf_x4(alpha_lr)
color_rgba_hr, color_diagnostics = upscale_color_sdf_x4(rgba_lr)
adaptive_rgba_hr, adaptive_diagnostics = upscale_adaptive_sdf_x4(rgba_lr)
```

Contracts:

- RGBA inputs are `uint8`, `HxWx4`, straight color.
- Alpha inputs are `uint8` in `[0, 255]` or floating point in `[0, 1]`.
- Alpha output is `float32` in `[0, 1]`.
- RGBA outputs are `uint8`, `(4H)x(4W)x4`.
- Hidden RGB under fully transparent input is excluded from visible output.
- Supplying `alpha_hr` to Adaptive-SDF preserves that alpha after uint8
  quantization.

The original implementation names remain available for compatibility, such as
`resize_alpha_main_x4`, `upscale_rgba_sal_classic_x4`, and
`upscale_rgba_adaptive_color_sdf_x4`.

## Pure-Rust crate

The repository also contains a pure-Rust Cargo workspace:

```bash
cargo test --workspace

# Adaptive-SDF RGBA x4 (default)
cargo run --release -p sdf-upscale-cli -- \
  input.png -o output.png

# Color-SDF or Alpha-SDF
cargo run --release -p sdf-upscale-cli -- \
  input.png -o color_x4.png --backend color
cargo run --release -p sdf-upscale-cli -- \
  input.png -o alpha_x4.png --backend alpha
```

The dependency-light library package is `sdf-upscaler` and the image-I/O
binary is `sdf-upscale`. It exposes canonical pure-Rust Alpha-, Color-, and
Adaptive-SDF x4 APIs. Their quantized outputs are byte-exact on the pinned
Python fixed fixtures and compact behavioral corpora, including hidden-RGB
invariance and supplied-alpha preservation. The older Alpha ROI prototype
remains available under the explicit `alpha-experimental` backend.

See [the Rust crate status and parity route](docs/RUST_CRATE.md).

## Development

```bash
uv run pytest
uv run ruff check src tests
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The golden-parity test freezes outputs copied from the source Upscaler working
tree. See [PROVENANCE.md](PROVENANCE.md) and the design notes in
[docs/](docs/).
