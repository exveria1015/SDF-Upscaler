use crate::UpscaleError;

/// Checked borrowed 2D plane with a row stride measured in elements.
#[derive(Clone, Copy, Debug)]
pub struct PlaneRef<'a, T> {
    width: usize,
    height: usize,
    stride: usize,
    data: &'a [T],
}

impl<'a, T> PlaneRef<'a, T> {
    pub fn new(
        width: usize,
        height: usize,
        stride: usize,
        data: &'a [T],
    ) -> Result<Self, UpscaleError> {
        if width == 0 || height == 0 {
            return Err(UpscaleError::EmptyImage);
        }
        if stride < width {
            return Err(UpscaleError::InvalidStride { width, stride });
        }
        let required = height
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(stride))
            .and_then(|prefix| prefix.checked_add(width))
            .ok_or(UpscaleError::DimensionOverflow)?;
        if data.len() < required {
            return Err(UpscaleError::BufferTooSmall {
                required,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            stride,
            data,
        })
    }

    pub fn packed(width: usize, height: usize, data: &'a [T]) -> Result<Self, UpscaleError> {
        Self::new(width, height, width, data)
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn stride(&self) -> usize {
        self.stride
    }

    pub fn row(&self, y: usize) -> &'a [T] {
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }
}

/// Checked owned packed 2D plane.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T> Plane<T> {
    pub fn new(width: usize, height: usize, data: Vec<T>) -> Result<Self, UpscaleError> {
        if width == 0 || height == 0 {
            return Err(UpscaleError::EmptyImage);
        }
        let required = width
            .checked_mul(height)
            .ok_or(UpscaleError::DimensionOverflow)?;
        if data.len() != required {
            return Err(UpscaleError::BufferTooSmall {
                required,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn as_ref(&self) -> PlaneRef<'_, T> {
        PlaneRef {
            width: self.width,
            height: self.height,
            stride: self.width,
            data: &self.data,
        }
    }
}

/// Checked borrowed straight-sRGB RGBA8 image with a byte stride.
#[derive(Clone, Copy, Debug)]
pub struct Rgba8Ref<'a> {
    width: usize,
    height: usize,
    stride_bytes: usize,
    data: &'a [u8],
}

impl<'a> Rgba8Ref<'a> {
    pub fn new(
        width: usize,
        height: usize,
        stride_bytes: usize,
        data: &'a [u8],
    ) -> Result<Self, UpscaleError> {
        if width == 0 || height == 0 {
            return Err(UpscaleError::EmptyImage);
        }
        let row_bytes = width
            .checked_mul(4)
            .ok_or(UpscaleError::DimensionOverflow)?;
        if stride_bytes < row_bytes {
            return Err(UpscaleError::InvalidStride {
                width: row_bytes,
                stride: stride_bytes,
            });
        }
        let required = height
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(stride_bytes))
            .and_then(|prefix| prefix.checked_add(row_bytes))
            .ok_or(UpscaleError::DimensionOverflow)?;
        if data.len() < required {
            return Err(UpscaleError::BufferTooSmall {
                required,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            stride_bytes,
            data,
        })
    }

    pub fn packed(width: usize, height: usize, data: &'a [u8]) -> Result<Self, UpscaleError> {
        let stride_bytes = width
            .checked_mul(4)
            .ok_or(UpscaleError::DimensionOverflow)?;
        Self::new(width, height, stride_bytes, data)
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn stride_bytes(&self) -> usize {
        self.stride_bytes
    }

    pub fn row(&self, y: usize) -> &'a [u8] {
        let row_bytes = self.width * 4;
        let start = y * self.stride_bytes;
        &self.data[start..start + row_bytes]
    }
}

/// Checked owned packed straight-sRGB RGBA8 image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8 {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Rgba8 {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self, UpscaleError> {
        if width == 0 || height == 0 {
            return Err(UpscaleError::EmptyImage);
        }
        let required = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(UpscaleError::DimensionOverflow)?;
        if data.len() != required {
            return Err(UpscaleError::BufferTooSmall {
                required,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }

    pub fn as_ref(&self) -> Rgba8Ref<'_> {
        Rgba8Ref {
            width: self.width,
            height: self.height,
            stride_bytes: self.width * 4,
            data: &self.data,
        }
    }
}

/// Accepted public Alpha-SDF input representations.
#[derive(Clone, Copy, Debug)]
pub enum AlphaInput<'a> {
    U8(PlaneRef<'a, u8>),
    F32(PlaneRef<'a, f32>),
}

impl<'a> From<PlaneRef<'a, u8>> for AlphaInput<'a> {
    fn from(value: PlaneRef<'a, u8>) -> Self {
        Self::U8(value)
    }
}

impl<'a> From<PlaneRef<'a, f32>> for AlphaInput<'a> {
    fn from(value: PlaneRef<'a, f32>) -> Self {
        Self::F32(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_ref_validates_stride_and_length() {
        assert_eq!(
            PlaneRef::new(3, 2, 2, &[0_u8; 6]).unwrap_err(),
            UpscaleError::InvalidStride {
                width: 3,
                stride: 2,
            }
        );
        assert_eq!(
            PlaneRef::new(3, 2, 4, &[0_u8; 6]).unwrap_err(),
            UpscaleError::BufferTooSmall {
                required: 7,
                actual: 6,
            }
        );
    }

    #[test]
    fn rgba_ref_validates_byte_stride_and_length() {
        assert_eq!(
            Rgba8Ref::new(3, 2, 11, &[0_u8; 24]).unwrap_err(),
            UpscaleError::InvalidStride {
                width: 12,
                stride: 11,
            }
        );
        assert_eq!(
            Rgba8Ref::new(3, 2, 16, &[0_u8; 27]).unwrap_err(),
            UpscaleError::BufferTooSmall {
                required: 28,
                actual: 27,
            }
        );
    }
}
