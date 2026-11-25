//! 图片缩放的几何计算：把原始像素尺寸映射到视口，并维护缩放挡位。

/// 最小显示比例（百分比，相对原始像素）。
pub const MIN_ZOOM: f32 = 5.0;
/// 最大显示比例（百分比，相对原始像素）。
pub const MAX_ZOOM: f32 = 800.0;

/// 等比适配视口：整体小于视口时保持原始像素尺寸，否则缩小到视口内。
pub fn fit_px(original: [usize; 2], avail_px: [f32; 2]) -> [f32; 2] {
    let (w, h) = (original[0] as f32, original[1] as f32);
    if w <= avail_px[0] && h <= avail_px[1] {
        [w, h]
    } else {
        let s = (avail_px[0] / w).min(avail_px[1] / h);
        [w * s, h * s]
    }
}

/// 纹理边长不能超过 GPU 上限，超限时等比缩小。
pub fn clamp_to_max(px: [f32; 2], max_side: usize) -> [f32; 2] {
    let longest = px[0].max(px[1]);
    if longest > max_side as f32 {
        let s = max_side as f32 / longest;
        [px[0] * s, px[1] * s]
    } else {
        px
    }
}

/// 适配视口时的显示比例（百分比，100% 表示原始像素 1:1 显示）。
pub fn fit_scale_pct(original: [usize; 2], avail_px: [f32; 2]) -> f32 {
    let fitted = fit_px(original, avail_px);
    let width = original[0].max(1) as f32;
    (fitted[0] / width * 100.0).clamp(MIN_ZOOM, 100.0)
}

/// 按显示比例换算纹理像素尺寸，并受 GPU 纹理上限约束。
pub fn scale_px(original: [usize; 2], pct: f32, max_side: usize) -> [f32; 2] {
    let s = pct / 100.0;
    clamp_to_max([original[0] as f32 * s, original[1] as f32 * s], max_side)
}

/// 缩放挡位：以 `anchor`（起始适配比例）为基准的等差阶梯，100% 单独占一档。
///
/// 例如基准 85%、步进 10% 时，放大依次为 85%→95%→100%→110%→120%。
pub fn zoom_gear(anchor: f32, current: f32, step: f32, dir: i32) -> f32 {
    let step = step.clamp(1.0, 100.0);
    const EPS: f32 = 0.5;
    let next = if dir > 0 {
        if current > 100.0 + EPS {
            current + step
        } else if (current - 100.0).abs() <= EPS {
            100.0 + step
        } else {
            let next = current + step;
            if next >= 100.0 - EPS { 100.0 } else { next }
        }
    } else if current > 100.0 + EPS {
        let next = current - step;
        if next <= 100.0 + EPS { 100.0 } else { next }
    } else if (current - 100.0).abs() <= EPS {
        // 100% 往下取基准阶梯中低于 100% 的最大一档。
        let steps = ((100.0 - anchor) / step).floor();
        let mut next = anchor + steps * step;
        if next >= 100.0 - EPS {
            next -= step;
        }
        next
    } else {
        current - step
    };
    next.clamp(MIN_ZOOM, MAX_ZOOM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_image_keeps_original_pixels() {
        assert_eq!(fit_px([120, 90], [800.0, 600.0]), [120.0, 90.0]);
    }

    #[test]
    fn large_image_scales_into_viewport() {
        let scaled = fit_px([3000, 2000], [600.0, 600.0]);
        assert_eq!(scaled, [600.0, 400.0]);
    }

    #[test]
    fn texture_side_is_capped() {
        assert_eq!(clamp_to_max([9000.0, 4500.0], 4096), [4096.0, 2048.0]);
        assert_eq!(clamp_to_max([900.0, 450.0], 4096), [900.0, 450.0]);
    }

    #[test]
    fn fit_scale_is_percentage_of_original_pixels() {
        assert_eq!(fit_scale_pct([120, 90], [800.0, 600.0]), 100.0);
        assert_eq!(fit_scale_pct([3000, 2000], [600.0, 600.0]), 20.0);
    }

    #[test]
    fn scale_px_follows_percentage_and_caps_side() {
        assert_eq!(scale_px([1000, 500], 50.0, 4096), [500.0, 250.0]);
        let capped = scale_px([9000, 9000], 200.0, 4096);
        assert!((capped[0] - 4096.0).abs() < 0.01 && (capped[1] - 4096.0).abs() < 0.01);
    }

    #[test]
    fn zoom_in_snaps_to_hundred_percent_gear() {
        let anchor = 85.0;
        let mut level = anchor;
        let mut seen = vec![level];
        for _ in 0..4 {
            level = zoom_gear(anchor, level, 10.0, 1);
            seen.push(level);
        }
        assert_eq!(seen, vec![85.0, 95.0, 100.0, 110.0, 120.0]);
    }

    #[test]
    fn zoom_out_returns_to_previous_gear() {
        let anchor = 85.0;
        assert_eq!(zoom_gear(anchor, 100.0, 10.0, -1), 95.0);
        assert_eq!(zoom_gear(anchor, 95.0, 10.0, -1), 85.0);
        assert_eq!(zoom_gear(anchor, 85.0, 10.0, -1), 75.0);
        assert_eq!(zoom_gear(anchor, 110.0, 10.0, -1), 100.0);
    }

    #[test]
    fn zoom_out_from_hundred_percent_uses_anchor_ladder() {
        assert_eq!(zoom_gear(100.0, 100.0, 10.0, -1), 90.0);
        // 基准 70% 的阶梯为 70/80/90/100，100% 往下取 90%。
        assert_eq!(zoom_gear(70.0, 100.0, 10.0, -1), 90.0);
    }
}
