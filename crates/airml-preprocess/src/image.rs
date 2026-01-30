//! Image preprocessing for ML models
//!
//! Provides image loading, resizing, and normalization utilities.

use image::{DynamicImage, GenericImageView, ImageReader};
use ndarray::{Array4, ArrayD};
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during image preprocessing
#[derive(Error, Debug)]
pub enum ImagePreprocessError {
    #[error("Failed to open image: {0}")]
    OpenError(String),

    #[error("Failed to decode image: {0}")]
    DecodeError(String),

    #[error("Failed to resize image: {0}")]
    ResizeError(String),

    #[error("Invalid image dimensions")]
    InvalidDimensions,
}

type Result<T> = std::result::Result<T, ImagePreprocessError>;

/// Resize mode for image preprocessing
#[derive(Debug, Clone, Copy, Default)]
pub enum ResizeMode {
    /// Resize to exact dimensions (may distort aspect ratio)
    #[default]
    Exact,
    /// Resize maintaining aspect ratio, then center crop
    CenterCrop,
    /// Resize maintaining aspect ratio, pad remaining area
    Pad { fill: [u8; 3] },
}

/// Image preprocessor with configurable normalization
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    /// Target width
    pub width: u32,
    /// Target height
    pub height: u32,
    /// Mean values for normalization (per channel)
    pub mean: [f32; 3],
    /// Standard deviation values for normalization (per channel)
    pub std: [f32; 3],
    /// Resize mode
    pub resize_mode: ResizeMode,
    /// Whether to use RGB (true) or BGR (false)
    pub rgb: bool,
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self::imagenet()
    }
}

impl ImagePreprocessor {
    /// Create a preprocessor with ImageNet normalization settings
    /// Input: 224x224, RGB, normalized with ImageNet mean/std
    pub fn imagenet() -> Self {
        Self {
            width: 224,
            height: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            resize_mode: ResizeMode::Exact,
            rgb: true,
        }
    }

    /// Create a preprocessor with CLIP normalization settings
    /// Input: 224x224, RGB, normalized with CLIP mean/std
    pub fn clip() -> Self {
        Self {
            width: 224,
            height: 224,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
            resize_mode: ResizeMode::CenterCrop,
            rgb: true,
        }
    }

    /// Create a preprocessor for YOLO-style models
    /// Input: 640x640, RGB, normalized to [0, 1]
    pub fn yolo(size: u32) -> Self {
        Self {
            width: size,
            height: size,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            resize_mode: ResizeMode::Pad { fill: [114, 114, 114] },
            rgb: true,
        }
    }

    /// Create a custom preprocessor
    pub fn custom(width: u32, height: u32, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            width,
            height,
            mean,
            std,
            resize_mode: ResizeMode::Exact,
            rgb: true,
        }
    }

    /// Set resize mode
    pub fn with_resize_mode(mut self, mode: ResizeMode) -> Self {
        self.resize_mode = mode;
        self
    }

    /// Set color mode (RGB or BGR)
    pub fn with_rgb(mut self, rgb: bool) -> Self {
        self.rgb = rgb;
        self
    }

    /// Load and preprocess an image from a file
    pub fn load_and_process<P: AsRef<Path>>(&self, path: P) -> Result<Array4<f32>> {
        let img = self.load_image(path)?;
        self.process(&img)
    }

    /// Load an image from a file
    pub fn load_image<P: AsRef<Path>>(&self, path: P) -> Result<DynamicImage> {
        ImageReader::open(path.as_ref())
            .map_err(|e| ImagePreprocessError::OpenError(e.to_string()))?
            .decode()
            .map_err(|e| ImagePreprocessError::DecodeError(e.to_string()))
    }

    /// Process a DynamicImage into a tensor
    pub fn process(&self, img: &DynamicImage) -> Result<Array4<f32>> {
        let resized = self.resize(img)?;
        let tensor = self.to_tensor(&resized);
        Ok(tensor)
    }

    /// Process a DynamicImage into a dynamic-dimension tensor
    pub fn process_dynamic(&self, img: &DynamicImage) -> Result<ArrayD<f32>> {
        let tensor = self.process(img)?;
        Ok(tensor.into_dyn())
    }

    fn resize(&self, img: &DynamicImage) -> Result<DynamicImage> {
        match self.resize_mode {
            ResizeMode::Exact => {
                Ok(img.resize_exact(
                    self.width,
                    self.height,
                    image::imageops::FilterType::Triangle,
                ))
            }
            ResizeMode::CenterCrop => {
                let resized = self.resize_preserve_aspect(img);
                Ok(self.center_crop(&resized))
            }
            ResizeMode::Pad { fill } => {
                Ok(self.resize_and_pad(img, fill))
            }
        }
    }

    fn resize_preserve_aspect(&self, img: &DynamicImage) -> DynamicImage {
        let (orig_w, orig_h) = img.dimensions();
        let scale = f32::max(
            self.width as f32 / orig_w as f32,
            self.height as f32 / orig_h as f32,
        );

        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;

        img.resize(new_w, new_h, image::imageops::FilterType::Triangle)
    }

    fn center_crop(&self, img: &DynamicImage) -> DynamicImage {
        let (w, h) = img.dimensions();
        let x = (w.saturating_sub(self.width)) / 2;
        let y = (h.saturating_sub(self.height)) / 2;

        img.crop_imm(x, y, self.width, self.height)
    }

    fn resize_and_pad(&self, img: &DynamicImage, fill: [u8; 3]) -> DynamicImage {
        let (orig_w, orig_h) = img.dimensions();
        let scale = f32::min(
            self.width as f32 / orig_w as f32,
            self.height as f32 / orig_h as f32,
        );

        let new_w = (orig_w as f32 * scale) as u32;
        let new_h = (orig_h as f32 * scale) as u32;

        let resized = img.resize(new_w, new_h, image::imageops::FilterType::Triangle);

        let mut padded = image::RgbImage::from_pixel(
            self.width,
            self.height,
            image::Rgb(fill),
        );

        let x_offset = (self.width - new_w) / 2;
        let y_offset = (self.height - new_h) / 2;

        image::imageops::overlay(&mut padded, &resized.to_rgb8(), x_offset.into(), y_offset.into());

        DynamicImage::ImageRgb8(padded)
    }

    fn to_tensor(&self, img: &DynamicImage) -> Array4<f32> {
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();

        let mut tensor = Array4::<f32>::zeros((1, 3, height as usize, width as usize));

        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_img.get_pixel(x, y);
                let (r, g, b) = if self.rgb {
                    (pixel[0], pixel[1], pixel[2])
                } else {
                    (pixel[2], pixel[1], pixel[0]) // BGR
                };

                // Normalize: (pixel / 255.0 - mean) / std
                tensor[[0, 0, y as usize, x as usize]] =
                    (r as f32 / 255.0 - self.mean[0]) / self.std[0];
                tensor[[0, 1, y as usize, x as usize]] =
                    (g as f32 / 255.0 - self.mean[1]) / self.std[1];
                tensor[[0, 2, y as usize, x as usize]] =
                    (b as f32 / 255.0 - self.mean[2]) / self.std[2];
            }
        }

        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imagenet_preprocessor() {
        let preprocessor = ImagePreprocessor::imagenet();
        assert_eq!(preprocessor.width, 224);
        assert_eq!(preprocessor.height, 224);
    }

    #[test]
    fn test_clip_preprocessor() {
        let preprocessor = ImagePreprocessor::clip();
        assert_eq!(preprocessor.width, 224);
        assert_eq!(preprocessor.height, 224);
    }

    #[test]
    fn test_yolo_preprocessor() {
        let preprocessor = ImagePreprocessor::yolo(640);
        assert_eq!(preprocessor.width, 640);
        assert_eq!(preprocessor.height, 640);
    }
}
