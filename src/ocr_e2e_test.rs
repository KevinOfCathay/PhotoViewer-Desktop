//! 集成测试：用项目内的真实 ONNX 模型跑通检测→识别全流程。
//!
//! 需要 `models/` 目录下存在 det.onnx、rec.onnx、dict.txt。
//! 模型缺失时该测试自动跳过，避免在没有模型的机器上失败。
//!
//! 设置环境变量可验证识别准确率：
//! - `PV_OCR_TEST_IMAGE`：待识别的图片路径
//! - `PV_OCR_TEST_EXPECT`：期望出现在识别结果中的文本片段

use image::RgbaImage;

use crate::ocr::OcrEngine;
use crate::ocr_worker::{self, Msg};

/// 模型目录：优先取可执行文件旁的 models/，否则取当前目录下的 models/。
fn model_dir() -> std::path::PathBuf {
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let candidate = dir.join("models");
        if candidate.is_dir() {
            return candidate;
        }
    }
    // 测试二进制位于 target/debug 下，向上两级是项目根目录。
    if let Ok(exe) = std::env::current_exe()
        && let Some(root) = exe
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
    {
        let candidate = root.join("models");
        if candidate.is_dir() {
            return candidate;
        }
    }
    std::path::PathBuf::from("models")
}

/// 造一张含文字笔画的测试图（无字体依赖）。
fn sample_image() -> RgbaImage {
    let mut img = RgbaImage::new(640, 200);
    for y in 20..70 {
        for x in 20..400 {
            let value = ((x / 7 + y / 5) % 2) as u8 * 255;
            img.put_pixel(x, y, image::Rgba([value, value, value, 255]));
        }
    }
    img
}

#[test]
fn recognize_runs_end_to_end_with_real_models() {
    let dir = model_dir();
    if !dir.join("det.onnx").exists() {
        eprintln!("跳过：未找到模型目录 {}", dir.display());
        return;
    }
    let mut engine = OcrEngine::load(&dir).expect("加载模型失败");

    // 指定了真实图片时验证识别准确率，否则用合成图验证链路。
    let (image, expected) = match std::env::var("PV_OCR_TEST_IMAGE") {
        Ok(path) => {
            let image = image::open(&path)
                .unwrap_or_else(|e| panic!("打开测试图片失败 {path}: {e}"))
                .to_rgba8();
            (image, std::env::var("PV_OCR_TEST_EXPECT").ok())
        }
        Err(_) => (sample_image(), None),
    };

    let mut stages = Vec::new();
    let boxes = engine
        .recognize(&image, |fraction, label| {
            stages.push((fraction, label.to_string()));
        })
        .expect("识别失败");

    // 进度回调应被调用，且进度单调递增到 1.0。
    assert!(!stages.is_empty(), "进度回调未被调用");
    let last = stages.last().unwrap().0;
    assert!((last - 1.0).abs() < 0.01, "最终进度应为 1.0，实际 {last}");
    for pair in stages.windows(2) {
        assert!(
            pair[1].0 >= pair[0].0,
            "进度应单调递增：{} -> {}",
            pair[0].0,
            pair[1].0
        );
    }

    // 所有识别框都必须落在原图范围内。
    for item in &boxes {
        for (x, y) in item.quad {
            assert!(
                (0.0..=image.width() as f32).contains(&x)
                    && (0.0..=image.height() as f32).contains(&y),
                "框坐标越界：({x}, {y})"
            );
        }
    }

    let all: Vec<String> = boxes.iter().map(|b| b.text.clone()).collect();
    eprintln!("[e2e] 识别到 {} 个文本区域：{:?}", boxes.len(), all);

    if let Some(expected) = expected {
        assert!(
            !boxes.is_empty(),
            "期望识别出包含 {expected:?} 的文本，但没有任何结果"
        );
        let joined = all.join(" ");
        assert!(
            joined.contains(&expected),
            "识别结果应包含 {expected:?}，实际为 {joined:?}"
        );
    }
}

/// 覆盖后台工作线程的完整链路：提交任务 → 收到进度 → 收到结果。
#[test]
fn worker_thread_completes_recognition_task() {
    let dir = model_dir();
    if !dir.join("det.onnx").exists() {
        eprintln!("跳过：未找到模型目录 {}", dir.display());
        return;
    }
    let Ok(image_path) = std::env::var("PV_OCR_TEST_IMAGE") else {
        eprintln!("跳过：未设置 PV_OCR_TEST_IMAGE");
        return;
    };
    let image = image::open(&image_path)
        .unwrap_or_else(|e| panic!("打开测试图片失败 {image_path}: {e}"))
        .to_rgba8();

    let worker = ocr_worker::OcrWorker::spawn(dir);
    worker.recognize(1, std::path::PathBuf::from(&image_path), image);

    // 应陆续收到进度消息，最后是完成消息。
    let mut progress_count = 0usize;
    let mut last_fraction = 0.0f32;
    let mut result = None;
    for _ in 0..200 {
        match worker.msgs.recv_timeout(std::time::Duration::from_secs(2)) {
            Ok(Msg::Progress { fraction, .. }) => {
                assert!(fraction >= last_fraction, "进度应单调递增");
                last_fraction = fraction;
                progress_count += 1;
            }
            Ok(Msg::Done { token, result: r }) => {
                assert_eq!(token, 1);
                result = Some(r);
                break;
            }
            Err(_) => break,
        }
    }
    assert!(progress_count > 0, "应收到进度消息");
    let result = result.expect("应收到完成消息");
    let boxes = result.expect("识别不应失败");
    eprintln!("[e2e-worker] 收到 {progress_count} 条进度，识别到 {} 个区域", boxes.len());

    if let Ok(expected) = std::env::var("PV_OCR_TEST_EXPECT") {
        let joined: Vec<String> = boxes.iter().map(|b| b.text.clone()).collect();
        let joined = joined.join(" ");
        assert!(
            joined.contains(&expected),
            "识别结果应包含 {expected:?}，实际为 {joined:?}"
        );
    }
}
