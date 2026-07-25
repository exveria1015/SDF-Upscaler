use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use image::GenericImageView;
use sdf_upscaler::{
    ALPHA_EXPERIMENTAL_PIPELINE_ID, AlphaDiagnostics, AlphaInput, AlphaOptions, PlaneRef, VERSION,
    quantize_alpha_u8, upscale_alpha_sdf_x4_experimental,
};
use serde::Serialize;
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["png", "webp", "tif", "tiff", "bmp", "jpg", "jpeg"];

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    /// Experimental ROI implementation; not yet Python Alpha-SDF parity.
    AlphaExperimental,
}

#[derive(Debug, Parser)]
#[command(
    name = "sdf-upscale",
    version,
    about = "Pure-Rust experimental Alpha-SDF x4 upscaling"
)]
struct Args {
    /// Single input image.
    input: Option<PathBuf>,

    /// Recursively process supported images under a directory.
    #[arg(long, value_name = "DIR")]
    dir: Option<PathBuf>,

    /// Output PNG or output directory.
    #[arg(short, long, value_name = "PATH")]
    output: PathBuf,

    /// Rust backend. Canonical Color/Adaptive backends are not exposed yet.
    #[arg(long, value_enum, default_value_t = Backend::AlphaExperimental)]
    backend: Backend,

    /// Replace existing output files.
    #[arg(long)]
    overwrite: bool,

    #[arg(long, default_value_t = 2)]
    roi_radius_lr: usize,

    #[arg(long, default_value_t = 4.0)]
    effect_band_hr: f32,

    #[arg(long, default_value_t = 0.20)]
    residual_cap: f32,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    reprojection_guard: bool,
}

#[derive(Debug, Serialize)]
struct CliManifest {
    schema_version: u32,
    kind: &'static str,
    crate_version: &'static str,
    input: String,
    output: String,
    input_alpha_source: &'static str,
    diagnostics: AlphaDiagnostics,
}

fn main() -> Result<()> {
    run(Args::parse())
}

fn run(args: Args) -> Result<()> {
    if (args.input.is_some()) == (args.dir.is_some()) {
        bail!("provide exactly one input image or --dir");
    }
    let options = AlphaOptions {
        roi_radius_lr: args.roi_radius_lr,
        effect_band_hr: args.effect_band_hr,
        residual_cap: args.residual_cap,
        reprojection_guard: args.reprojection_guard,
    };
    match args.backend {
        Backend::AlphaExperimental => {}
    }
    if let Some(input) = args.input {
        process_one(&input, &args.output, options, args.overwrite)?;
        println!("{}", args.output.canonicalize()?.display());
        return Ok(());
    }

    let input_root = args.dir.expect("directory checked above").canonicalize()?;
    let output_root = absolute_output(&args.output)?;
    let inputs = discover(&input_root)?;
    if inputs.is_empty() {
        bail!("no supported images under {}", input_root.display());
    }
    for (index, input) in inputs.iter().enumerate() {
        let relative = input.strip_prefix(&input_root)?;
        let output = output_root.join(relative).with_extension("png");
        process_one(input, &output, options, args.overwrite)?;
        println!(
            "[{}/{}] {} -> {}",
            index + 1,
            inputs.len(),
            input.display(),
            output.display()
        );
    }
    Ok(())
}

fn discover(root: &Path) -> Result<Vec<PathBuf>> {
    let mut result = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .filter(|path| {
            path.extension()
                .and_then(|value| value.to_str())
                .is_some_and(|value| {
                    IMAGE_EXTENSIONS
                        .iter()
                        .any(|extension| value.eq_ignore_ascii_case(extension))
                })
        })
        .collect::<Vec<_>>();
    result.sort();
    Ok(result)
}

fn process_one(input: &Path, output: &Path, options: AlphaOptions, overwrite: bool) -> Result<()> {
    if output.exists() && !overwrite {
        bail!(
            "output exists; pass --overwrite to replace it: {}",
            output.display()
        );
    }
    let decoded = image::open(input)
        .with_context(|| format!("failed to decode input image: {}", input.display()))?;
    let (width_u32, height_u32) = decoded.dimensions();
    let width = width_u32 as usize;
    let height = height_u32 as usize;
    let (alpha, source) = if decoded.color().has_alpha() {
        (
            decoded
                .to_rgba8()
                .pixels()
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            "embedded_alpha",
        )
    } else {
        (decoded.to_luma8().into_raw(), "grayscale_or_luma")
    };
    let plane = PlaneRef::packed(width, height, &alpha)?;
    let result = upscale_alpha_sdf_x4_experimental(AlphaInput::U8(plane), options)?;
    let output_bytes = quantize_alpha_u8(result.alpha.as_slice());
    write_alpha_png_atomic(
        output,
        result.alpha.width(),
        result.alpha.height(),
        &output_bytes,
    )?;
    let manifest = CliManifest {
        schema_version: 1,
        kind: ALPHA_EXPERIMENTAL_PIPELINE_ID,
        crate_version: VERSION,
        input: input.to_string_lossy().into_owned(),
        output: output.to_string_lossy().into_owned(),
        input_alpha_source: source,
        diagnostics: result.diagnostics,
    };
    write_manifest_atomic(&manifest_path(output), &manifest)?;
    Ok(())
}

fn write_alpha_png_atomic(output: &Path, width: usize, height: usize, data: &[u8]) -> Result<()> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = temporary_path(output);
    let file = File::create(&temporary)
        .with_context(|| format!("failed to create {}", temporary.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width as u32, height as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.add_text_chunk("sdf-upscaler.version".to_owned(), VERSION.to_owned())?;
    encoder.add_text_chunk(
        "sdf-upscaler.pipeline".to_owned(),
        ALPHA_EXPERIMENTAL_PIPELINE_ID.to_owned(),
    )?;
    let mut writer = encoder.write_header()?;
    writer.write_image_data(data)?;
    writer.finish()?;
    fs::rename(&temporary, output)
        .with_context(|| format!("failed to publish {}", output.display()))?;
    Ok(())
}

fn write_manifest_atomic(path: &Path, manifest: &CliManifest) -> Result<()> {
    let temporary = temporary_path(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(manifest)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn manifest_path(output: &Path) -> PathBuf {
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("alpha");
    output.with_file_name(format!("{stem}.manifest.json"))
}

fn temporary_path(path: &Path) -> PathBuf {
    path.with_file_name(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("output"),
        std::process::id()
    ))
}

fn absolute_output(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()?.join(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use tempfile::tempdir;

    #[test]
    fn single_file_writes_x4_png_metadata_and_manifest() {
        let directory = tempdir().unwrap();
        let input = directory.path().join("input.png");
        let output = directory.path().join("output.png");
        let image = GrayImage::from_fn(5, 4, |x, y| {
            Luma([if (1..4).contains(&x) && (1..3).contains(&y) {
                255
            } else {
                0
            }])
        });
        image.save(&input).unwrap();
        process_one(&input, &output, AlphaOptions::default(), false).unwrap();

        let decoded = image::open(&output).unwrap();
        assert_eq!(decoded.dimensions(), (20, 16));
        assert!(manifest_path(&output).is_file());
        let error = process_one(&input, &output, AlphaOptions::default(), false).unwrap_err();
        assert!(error.to_string().contains("--overwrite"));
    }
}
