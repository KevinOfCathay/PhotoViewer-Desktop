//! 图片编辑：缩放、翻转、旋转、黑白与亮度对比度。
//!
//! 编辑以「参数 + 原图」表达：参数描述全部变换，预览与写盘共用同一套变换，
//! 因此屏幕上的结果与保存到磁盘的文件一致。

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use image::DynamicImage;
use image::ImageFormat;
use image::RgbaImage;
use image::imageops::{FilterType, flip_horizontal, flip_vertical, resize, rotate180, rotate270, rotate90};


/// 可选的缩放档位（百分比）。
pub const SCALE_STEPS: [u8; 3] = [25, 50, 75];

/// 交互预览的最大边长（像素）。原图更大时先缩小再套用变换，保证实时响应。
pub const PREVIEW_MAX_SIDE: u32 = 2048;

/// 编辑参数：所有变换相对原图定义，按固定顺序应用。
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct EditState {
    /// 缩放档位，None 表示保持原始尺寸。
    pub scale: Option<u8>,
    /// 水平翻转。
    pub flip_h: bool,
    /// 垂直翻转。
    pub flip_v: bool,
    /// 旋转角度（0/90/180/270）。
    pub rotate: u16,
    /// 黑白化。
    pub grayscale: bool,
    /// 亮度（-100~100）。
    pub brightness: f32,
    /// 对比度（-100~100）。
    pub contrast: f32,
}

impl EditState {
    /// 是否存在任何修改。
    pub fn is_modified(&self) -> bool {
        *self != Self::default()
    }

    /// 缩放后的像素尺寸。
    fn scaled_size(&self, width: u32, height: u32) -> (u32, u32) {
        match self.scale {
            Some(pct) => {
                let factor = pct as f32 / 100.0;
                (
                    ((width as f32 * factor).round() as u32).max(1),
                    ((height as f32 * factor).round() as u32).max(1),
                )
            }
            None => (width, height),
        }
    }

    /// 变换后的像素尺寸：缩放与旋转会改变尺寸，翻转与调色不会。
    pub fn output_size(&self, width: u32, height: u32) -> (u32, u32) {
        let (w, h) = self.scaled_size(width, height);
        match self.rotate % 360 {
            90 | 270 => (h, w),
            _ => (w, h),
        }
    }

    /// 对原图套用全部变换，得到新的图片。
    pub fn apply(&self, img: &RgbaImage) -> RgbaImage {
        let mut out = match self.scale {
            Some(_) => {
                let (w, h) = self.scaled_size(img.width(), img.height());
                resize(img, w, h, FilterType::CatmullRom)
            }
            None => img.clone(),
        };
        out = match self.rotate % 360 {
            90 => rotate90(&out),
            180 => rotate180(&out),
            270 => rotate270(&out),
            _ => out,
        };
        if self.flip_h {
            out = flip_horizontal(&out);
        }
        if self.flip_v {
            out = flip_vertical(&out);
        }
        if self.grayscale {
            to_grayscale(&mut out);
        }
        adjust(&mut out, self.brightness, self.contrast);
        out
    }
}

/// 把图片写回原路径，按原文件格式编码。
///
/// 先写同目录的临时文件再替换，编码失败时原图保持完好。
pub fn save_overwrite(path: &Path, img: &RgbaImage) -> Result<(), String> {
    let format = ImageFormat::from_path(path).map_err(|err| format!("无法识别原图格式: {err}"))?;
    let temp = temp_path(path);
    let encoded = {
        perf_scope!("写盘编码");
        DynamicImage::ImageRgba8(img.clone()).save_with_format(&temp, format)
    };
    if let Err(err) = encoded {
        let _ = std::fs::remove_file(&temp);
        return Err(format!("保存失败: {err}"));
    }
    if let Err(err) = std::fs::rename(&temp, path) {
        let _ = std::fs::remove_file(&temp);
        return Err(format!("替换原图失败: {err}"));
    }
    Ok(())
}

/// 临时文件路径：与原图同目录，保证重命名是同盘操作。
fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".edit-tmp");
    path.with_file_name(name)
}

/// 发送给预览渲染线程的任务。
pub enum Job {
    /// 按当前参数渲染预览。
    Render {
        token: u64,
        /// 原图代号，用于识别换图。
        source_id: u64,
        base: Arc<RgbaImage>,
        edit: EditState,
    },
}

/// 预览渲染线程回传的结果。
pub enum Msg {
    /// 预览渲染完成。
    Rendered { token: u64, image: Arc<RgbaImage> },
}

/// 后台预览渲染器：大图渲染耗时可观，放到独立线程避免拖住界面。
pub struct EditorWorker {
    jobs: Sender<Job>,
    /// 后台线程回传的结果。
    pub msgs: Receiver<Msg>,
}

impl EditorWorker {
    /// 启动预览渲染线程。
    pub fn spawn() -> Self {
        let (job_tx, job_rx) = channel::<Job>();
        let (msg_tx, msg_rx) = channel::<Msg>();
        std::thread::spawn(move || {
            // 降采样后的基准图：参数变化时不必重复缩放大图。
            let mut base_key: Option<(u64, usize)> = None;
            let mut base_small: Option<Arc<RgbaImage>> = None;
            while let Ok(job) = job_rx.recv() {
                // 只处理最新参数，丢掉积压的旧任务。
                let mut job = job;
                while let Ok(next) = job_rx.try_recv() {
                    job = next;
                }
                let Job::Render {
                    token,
                    source_id,
                    base,
                    edit,
                } = job;
                let key = (source_id, Arc::as_ptr(&base) as usize);
                if base_key != Some(key) {
                    let small = {
                        perf_scope!("预览降采样");
                        downscale(base.as_ref(), PREVIEW_MAX_SIDE)
                            .unwrap_or_else(|| base.as_ref().clone())
                    };
                    base_small = Some(Arc::new(small));
                    base_key = Some(key);
                }
                let small = base_small.as_ref().expect("基准图已缓存");
                let out = {
                    perf_scope!("编辑预览渲染");
                    edit.apply(small.as_ref())
                };
                let _ = msg_tx.send(Msg::Rendered {
                    token,
                    image: Arc::new(out),
                });
            }
        });
        Self {
            jobs: job_tx,
            msgs: msg_rx,
        }
    }

    /// 提交一次预览渲染。
    pub fn render(&self, token: u64, source_id: u64, base: Arc<RgbaImage>, edit: EditState) {
        let _ = self.jobs.send(Job::Render {
            token,
            source_id,
            base,
            edit,
        });
    }
}

/// 把图片等比缩小到不超过 `cap`，本来就在上限内时返回 None。
fn downscale(img: &RgbaImage, cap: u32) -> Option<RgbaImage> {
    let longest = img.width().max(img.height());
    if longest <= cap {
        return None;
    }
    let factor = cap as f32 / longest as f32;
    let width = ((img.width() as f32 * factor).round() as u32).max(1);
    let height = ((img.height() as f32 * factor).round() as u32).max(1);
    Some(resize(img, width, height, FilterType::CatmullRom))
}

/// 黑白化：按亮度权重合成灰度，保留透明度。
fn to_grayscale(img: &mut RgbaImage) {
    for pixel in img.pixels_mut() {
        let luma = 0.299 * pixel.0[0] as f32 + 0.587 * pixel.0[1] as f32 + 0.114 * pixel.0[2] as f32;
        let value = luma.round().clamp(0.0, 255.0) as u8;
        pixel.0[0] = value;
        pixel.0[1] = value;
        pixel.0[2] = value;
    }
}

/// 亮度与对比度调整：先加亮度，再围绕中灰拉伸对比度。
fn adjust(img: &mut RgbaImage, brightness: f32, contrast: f32) {
    let offset = brightness.clamp(-100.0, 100.0) / 100.0 * 255.0;
    let factor = 1.0 + contrast.clamp(-100.0, 100.0) / 100.0;
    if offset.abs() < f32::EPSILON && (factor - 1.0).abs() < f32::EPSILON {
        return;
    }
    for pixel in img.pixels_mut() {
        for channel in pixel.0.iter_mut().take(3) {
            let value = (*channel as f32 + offset - 128.0) * factor + 128.0;
            *channel = value.round().clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一张 2x1 的红绿测试图。
    fn sample() -> RgbaImage {
        let mut img = RgbaImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        img.put_pixel(1, 0, image::Rgba([0, 255, 0, 255]));
        img
    }

    #[test]
    fn default_state_is_not_modified() {
        assert!(!EditState::default().is_modified());
        let mut edit = EditState::default();
        edit.brightness = 10.0;
        assert!(edit.is_modified());
    }

    #[test]
    fn horizontal_flip_swaps_pixels() {
        let mut edit = EditState::default();
        edit.flip_h = true;
        let out = edit.apply(&sample());
        assert_eq!(out.get_pixel(0, 0).0, [0, 255, 0, 255]);
        assert_eq!(out.get_pixel(1, 0).0, [255, 0, 0, 255]);
    }

    #[test]
    fn vertical_flip_swaps_rows() {
        let mut img = RgbaImage::new(1, 2);
        img.put_pixel(0, 0, image::Rgba([1, 2, 3, 255]));
        img.put_pixel(0, 1, image::Rgba([4, 5, 6, 255]));
        let mut edit = EditState::default();
        edit.flip_v = true;
        let out = edit.apply(&img);
        assert_eq!(out.get_pixel(0, 0).0, [4, 5, 6, 255]);
        assert_eq!(out.get_pixel(0, 1).0, [1, 2, 3, 255]);
    }

    #[test]
    fn rotate_swaps_dimensions_and_moves_pixels() {
        let mut edit = EditState::default();
        edit.rotate = 90;
        let out = edit.apply(&sample());
        assert_eq!((out.width(), out.height()), (1, 2));
        // 顺时针 90 度：原来的一行变成一列，左端在上。
        assert_eq!(out.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(out.get_pixel(0, 1).0, [0, 255, 0, 255]);
    }

    #[test]
    fn output_size_matches_applied_result() {
        let mut edit = EditState::default();
        edit.scale = Some(50);
        assert_eq!(edit.output_size(400, 200), (200, 100));

        edit.rotate = 90;
        assert_eq!(edit.output_size(400, 200), (100, 200));
        let out = edit.apply(&RgbaImage::new(400, 200));
        assert_eq!((out.width(), out.height()), (100, 200));
    }

    #[test]
    fn scale_halves_the_image() {
        let mut edit = EditState::default();
        edit.scale = Some(50);
        let img = RgbaImage::new(8, 4);
        let out = edit.apply(&img);
        assert_eq!((out.width(), out.height()), (4, 2));
    }

    #[test]
    fn grayscale_makes_channels_equal() {
        let mut edit = EditState::default();
        edit.grayscale = true;
        let out = edit.apply(&sample());
        let red = out.get_pixel(0, 0).0;
        assert_eq!(red[0], red[1]);
        assert_eq!(red[1], red[2]);
        assert_eq!(red[3], 255, "透明度应保留");
    }

    #[test]
    fn brightness_raises_and_contrast_pulls_to_middle() {
        let mut edit = EditState::default();
        edit.brightness = 100.0;
        let out = edit.apply(&sample());
        assert_eq!(out.get_pixel(0, 0).0[0], 255);

        let mut edit = EditState::default();
        edit.contrast = 100.0;
        let out = edit.apply(&sample());
        // 对比度翻倍后围绕 128 拉伸：0 → 0，255 → 255，中间的 128 附近放大。
        assert_eq!(out.get_pixel(0, 0).0[0], 255);
        assert_eq!(out.get_pixel(1, 0).0[1], 255);
    }

    /// 保存失败时原图保持完好，临时文件不残留。
    #[test]
    fn failed_save_keeps_original_intact() {
        let dir = std::env::temp_dir().join("photo-viewer-editor-fail-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.unknown");
        std::fs::write(&path, b"not an image").unwrap();

        assert!(save_overwrite(&path, &RgbaImage::new(2, 2)).is_err());
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"not an image",
            "保存失败时原文件不应被改动"
        );
        let leftovers: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains("edit-tmp"))
            .collect();
        assert!(leftovers.is_empty(), "临时文件不应残留: {leftovers:?}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn downscale_keeps_small_images_untouched() {
        let img = RgbaImage::new(10, 10);
        assert!(downscale(&img, PREVIEW_MAX_SIDE).is_none());
    }

    /// 保存后的文件内容与变换结果一致，即所见即所得。
    #[test]
    fn saved_file_matches_transformed_pixels() {
        let dir = std::env::temp_dir().join("photo-viewer-editor-save-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.png");
        let mut img = RgbaImage::new(4, 2);
        for (x, y, color) in [
            (0, 0, [255, 0, 0, 255]),
            (1, 0, [0, 255, 0, 255]),
            (2, 0, [0, 0, 255, 255]),
            (3, 0, [255, 255, 0, 255]),
            (0, 1, [0, 255, 255, 255]),
            (1, 1, [255, 0, 255, 255]),
            (2, 1, [255, 255, 255, 255]),
            (3, 1, [0, 0, 0, 255]),
        ] {
            img.put_pixel(x, y, image::Rgba(color));
        }
        img.save(&path).unwrap();

        let mut edit = EditState::default();
        edit.flip_h = true;
        edit.grayscale = true;
        let expected = edit.apply(&img);
        save_overwrite(&path, &expected).unwrap();

        let saved = image::open(&path).unwrap().to_rgba8();
        assert_eq!(saved.dimensions(), expected.dimensions());
        for (x, y, pixel) in saved.enumerate_pixels() {
            assert_eq!(pixel.0, expected.get_pixel(x, y).0, "像素 ({x}, {y}) 不一致");
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn preview_caps_long_side() {
        let img = RgbaImage::new(4000, 2000);
        let base = downscale(&img, PREVIEW_MAX_SIDE).unwrap();
        assert_eq!(base.width(), PREVIEW_MAX_SIDE);
        assert_eq!(base.height(), PREVIEW_MAX_SIDE / 2);
    }
}
