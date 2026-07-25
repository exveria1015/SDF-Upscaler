use std::fmt;

/// Validation and arithmetic failures returned by the Rust kernels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UpscaleError {
    EmptyImage,
    DimensionOverflow,
    InvalidStride { width: usize, stride: usize },
    BufferTooSmall { required: usize, actual: usize },
    InvalidAlpha { index: usize },
    InvalidOptions(&'static str),
}

impl fmt::Display for UpscaleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => write!(formatter, "image dimensions must be non-zero"),
            Self::DimensionOverflow => write!(formatter, "image dimensions overflow"),
            Self::InvalidStride { width, stride } => {
                write!(formatter, "stride {stride} is smaller than width {width}")
            }
            Self::BufferTooSmall { required, actual } => {
                write!(
                    formatter,
                    "buffer has {actual} elements; {required} are required"
                )
            }
            Self::InvalidAlpha { index } => {
                write!(
                    formatter,
                    "alpha at packed index {index} is not finite and in [0, 1]"
                )
            }
            Self::InvalidOptions(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for UpscaleError {}
