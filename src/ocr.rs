//! 本地 OCR：基于 PaddleOCR 官方 PP-OCRv5 模型导出的 ONNX，用 ONNX Runtime 在 CPU 上推理。
//!
//! 流水线与官方 inference.yml 对齐：检测(DB) → 后处理 → 裁剪 → 识别(CTC) → 解码。
//! 前处理参数取自官方配置，务必保持一致，否则识别准确率会明显下降。

use std::path::Path;
use std::sync::{Arc, Mutex};

use image::RgbaImage;
use ort::session::Session;
use ort::value::TensorRef;


/// 官方 det 归一化参数（ImageNet mean/std，BGR 通道序）。
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];
/// 官方 det 长边缩放上限。
const DET_RESIZE_LONG: f32 = 960.0;
/// 官方 DB 后处理参数。
const DB_THRESH: f32 = 0.3;
const DB_BOX_THRESH: f32 = 0.6;
const DB_UNCLIP_RATIO: f32 = 1.5;
/// 官方 rec 输入高度。
const REC_HEIGHT: usize = 48;
/// 识别输入宽度上限。
const REC_MAX_WIDTH: usize = 3200;
/// rec 单批允许的最大列数（批内行数 × 批内最大宽度）。
///
/// 整批识别时窄行会被补到最宽行，CPU 上大张量的缓存命中率很差，
/// 实测比按此预算切批慢 1.7~2.4 倍，且切批不影响识别结果。
const REC_BATCH_COLUMNS: usize = 4096;
/// DirectML 预热用的输入边长（模型要求 32 的倍数）。
const DET_WARMUP_SIZE: usize = 64;

/// 一个文本区域的检测框与识别结果。
#[derive(Clone, Debug)]
pub struct TextBox {
    /// 原图坐标系下的四点（顺序：左上、右上、右下、左下）。
    pub quad: [(f32, f32); 4],
    /// 识别出的文本。
    pub text: String,
    /// 识别置信度。
    pub score: f32,
}

/// OCR 引擎：持有 det/rec 两个 ONNX 会话与字符字典。
pub struct OcrEngine {
    /// CPU 检测会话，始终可用，作为 GPU 不可用时的兜底。
    det: Session,
    /// DirectML 检测会话：后台线程创建就绪后放入，就绪前仍走 CPU。
    det_gpu: Arc<Mutex<Option<Session>>>,
    /// DirectML 运行期失败后置位，不再重试 GPU。
    det_gpu_disabled: bool,
    /// 是否已经打印过"检测走 GPU"，避免每次识别都刷日志。
    det_gpu_logged: bool,
    rec: Session,
    dict: Vec<String>,
}

impl OcrEngine {
    /// 从模型目录加载引擎。目录需包含 det.onnx、rec.onnx、dict.txt。
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        perf_scope!("OCR 引擎加载");
        let started = std::time::Instant::now();
        let det_path = model_dir.join("det.onnx");
        let rec_path = model_dir.join("rec.onnx");
        let dict_path = model_dir.join("dict.txt");
        for path in [&det_path, &rec_path, &dict_path] {
            if !path.exists() {
                return Err(format!("缺少模型文件：{}", path.display()));
            }
        }

        let env = ort::init()
            .with_name("photo-viewer-ocr")
            .build()
            .map_err(|e| format!("初始化 ONNX Runtime 失败：{e}"))?;
        let det = Session::builder(&env)
            .map_err(|e| format!("创建检测会话失败：{e}"))?
            .commit_from_file(&det_path)
            .map_err(|e| format!("加载检测模型失败：{e}"))?;
        let mut rec = Session::builder(&env)
            .map_err(|e| format!("创建识别会话失败：{e}"))?
            .commit_from_file(&rec_path)
            .map_err(|e| format!("加载识别模型失败：{e}"))?;

        let mut dict: Vec<String> = std::fs::read_to_string(&dict_path)
            .map_err(|e| format!("读取字典失败：{e}"))?
            .lines()
            .map(|line| line.trim_end_matches(['\r', '\n']).to_string())
            .collect();

        // 官方 rec 推理时会在字典末尾补一个空格（use_space_char），
        // 模型类别数 = 字典条目 + 空格 + blank，比字典多一个；缺了它空格就解不出来。
        let classes = rec_num_classes(&mut rec);
        if classes == Some(dict.len() + 2) && !dict.iter().any(|c| c == " ") {
            dict.push(" ".to_string());
        }
        eprintln!(
            "[ocr] rec 类别数 {:?}，字典条目 {}",
            classes,
            dict.len()
        );

        // 检测优先走 DirectML：D3D12 设备初始化要几百毫秒，放后台线程创建，
        // 首次识别先用 CPU，等 GPU 会话就绪后再自动切换。
        let det_gpu = Arc::new(Mutex::new(None));
        {
            let slot = Arc::clone(&det_gpu);
            let path = det_path.clone();
            let env = env.clone();
            std::thread::spawn(move || {
                let session = build_directml_session(&env, &path);
                if let Ok(mut guard) = slot.lock() {
                    *guard = session;
                }
            });
        }

        Ok(Self {
            det,
            det_gpu,
            det_gpu_disabled: false,
            det_gpu_logged: false,
            rec,
            dict,
        })
        .inspect(|_| {
            eprintln!("[ocr] 引擎加载完成，耗时 {} ms", started.elapsed().as_millis());
            perf_log!(
                "OCR 引擎就绪，模型加载合计 {:.2}ms",
                crate::perf::ms(started.elapsed())
            );
        })
    }

    /// 识别整张图片，返回所有文本区域。
    ///
    /// `progress` 用于回报阶段进度（0.0~1.0），供界面显示进度条。
    pub fn recognize(
        &mut self,
        image: &RgbaImage,
        mut progress: impl FnMut(f32, &str),
    ) -> Result<Vec<TextBox>, String> {
        let (w0, h0) = (image.width(), image.height());
        if w0 == 0 || h0 == 0 {
            return Ok(Vec::new());
        }

        // ---------- 检测 ----------
        progress(0.15, "检测文本区域");
        let det_started = std::time::Instant::now();
        let (nw, nh) = det_resize(w0, h0);
        let small = image::imageops::resize(
            image,
            nw,
            nh,
            image::imageops::FilterType::Triangle,
        );

        // 官方 det 是 BGR 通道序：索引 0 为蓝、2 为红。
        let mut det_in = ndarray::Array4::<f32>::zeros((1, 3, nh as usize, nw as usize));
        for (x, y, px) in small.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            let channels = [px[2], px[1], px[0]];
            for (c, &value) in channels.iter().enumerate() {
                let v = value as f32 / 255.0;
                det_in[[0, c, y, x]] = (v - DET_MEAN[c]) / DET_STD[c];
            }
        }

        // 检测优先走 DirectML：GPU 不可用或推理失败时自动回退 CPU。
        // 概率图在这里就拷出来，避免借用 GPU 会话导致后续借用冲突。
        let (hm, wm, prob): (usize, usize, Vec<f32>) = {
            let mut gpu = self.det_gpu.lock().unwrap_or_else(|err| err.into_inner());
            let infer_started = std::time::Instant::now();
            let gpu_out = match gpu.as_mut() {
                Some(session) if !self.det_gpu_disabled => {
                    match session.run(ort::inputs!["x" => TensorRef::from_array_view(&det_in)
                        .map_err(|e| format!("构造检测输入失败：{e}"))?])
                    {
                        Ok(out) => {
                            if !self.det_gpu_logged {
                                self.det_gpu_logged = true;
                                eprintln!("[ocr] 检测已切换到 DirectML");
                            }
                            Some(out)
                        }
                        Err(err) => {
                            eprintln!("[ocr] DirectML 检测失败，回退 CPU：{err}");
                            self.det_gpu_disabled = true;
                            None
                        }
                    }
                }
                _ => None,
            };
            let out = match gpu_out {
                Some(out) => out,
                None => self
                    .det
                    .run(ort::inputs!["x" => TensorRef::from_array_view(&det_in)
                        .map_err(|e| format!("构造检测输入失败：{e}"))?])
                    .map_err(|e| format!("检测推理失败：{e}"))?,
            };
            let infer_ms = infer_started.elapsed().as_millis();
            let (shape, prob) = out[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("读取检测输出失败：{e}"))?;
            eprintln!("[ocr] 检测推理 {infer_ms} ms");
            perf_log!("检测推理 {}ms", infer_ms);
            (shape[2] as usize, shape[3] as usize, prob.to_vec())
        };

        // ---------- DB 后处理 ----------
        progress(0.45, "分析文本框");
        let quads = db_postprocess(&prob, hm, wm, w0 as f32, h0 as f32);
        eprintln!(
            "[ocr] 检测 {} 个文本区域，耗时 {} ms",
            quads.len(),
            det_started.elapsed().as_millis()
        );
        perf_log!(
            "检测阶段 {} 个区域，耗时 {:.2}ms",
            quads.len(),
            crate::perf::ms(det_started.elapsed())
        );
        if quads.is_empty() {
            progress(1.0, "完成");
            return Ok(Vec::new());
        }

        // ---------- 裁剪 + 识别 ----------
        progress(0.55, "裁剪文本行");
        let mut crops: Vec<(image::RgbImage, usize)> = Vec::new();
        for (index, quad) in quads.iter().enumerate() {
            let Some(crop) = crop_quad(image, quad) else {
                eprintln!("[ocr] 第 {} 个框裁剪失败，跳过", index);
                continue;
            };
            crops.push((crop, index));
        }
        eprintln!("[ocr] 有效裁剪 {} 行", crops.len());
        perf_log!("裁剪 {} 行文本", crops.len());
        if crops.is_empty() {
            progress(1.0, "完成");
            return Ok(Vec::new());
        }

        // 按宽度分批识别：整批会把窄行补到最宽行，大张量在 CPU 上缓存命中率差。
        let widths: Vec<usize> = crops.iter().map(|(c, _)| c.width() as usize).collect();
        let batches = plan_rec_batches(&widths);

        progress(0.7, "识别文字");
        let started = std::time::Instant::now();
        let mut results: Vec<Option<TextBox>> = vec![None; crops.len()];
        for batch in &batches {
            self.recognize_batch(&crops, &quads, batch, &mut results)?;
        }
        eprintln!(
            "[ocr] 识别完成：{} 行，{} 批，耗时 {} ms",
            crops.len(),
            batches.len(),
            started.elapsed().as_millis()
        );
        perf_log!(
            "识别阶段 {} 行 / {} 批，耗时 {:.2}ms",
            crops.len(),
            batches.len(),
            crate::perf::ms(started.elapsed())
        );

        progress(0.95, "整理结果");
        let boxes: Vec<TextBox> = results.into_iter().flatten().collect();
        progress(1.0, "完成");
        Ok(boxes)
    }

    /// 识别一批文本行：构造输入张量 → 推理 → CTC 解码 → 写回结果。
    fn recognize_batch(
        &mut self,
        crops: &[(image::RgbImage, usize)],
        quads: &[[(f32, f32); 4]],
        batch: &[usize],
        results: &mut [Option<TextBox>],
    ) -> Result<(), String> {
        let max_w = batch
            .iter()
            .map(|i| crops[*i].0.width() as usize)
            .max()
            .unwrap_or(1);
        // 逐行写入扁平缓冲区，省掉 ndarray 逐元素下标的边界检查。
        let plane = REC_HEIGHT * max_w;
        let mut buf = vec![0.0f32; batch.len() * 3 * plane];
        for (row, &i) in batch.iter().enumerate() {
            let crop = &crops[i].0;
            for (x, y, px) in crop.enumerate_pixels() {
                let (x, y) = (x as usize, y as usize);
                let base = row * 3 * plane + y * max_w + x;
                for c in 0..3 {
                    buf[base + c * plane] = (px[c] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }
        let rec_in = ndarray::Array4::from_shape_vec((batch.len(), 3, REC_HEIGHT, max_w), buf)
            .map_err(|e| format!("构造识别输入失败：{e}"))?;

        let rec_out = self
            .rec
            .run(ort::inputs!["x" => TensorRef::from_array_view(&rec_in)
                .map_err(|e| format!("构造识别输入失败：{e}"))?])
            .map_err(|e| format!("识别推理失败：{e}"))?;
        let (rshape, rdata) = rec_out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("读取识别输出失败：{e}"))?;
        let num_classes = rshape[2] as usize;
        let time_steps = rshape[1] as usize;

        for (row, &i) in batch.iter().enumerate() {
            let start = row * time_steps * num_classes;
            let probs = &rdata[start..start + time_steps * num_classes];
            let (text, score) = ctc_decode(probs, time_steps, num_classes, &self.dict);
            let index = crops[i].1;
            let quad = quads[index];
            if text.is_empty() {
                eprintln!(
                    "[ocr] 第 {} 个框未识别出文字，框=({:.1},{:.1})-({:.1},{:.1})",
                    index, quad[0].0, quad[0].1, quad[2].0, quad[2].1
                );
            } else {
                eprintln!("[ocr] 第 {} 个框：{:.2} {:?}", index, score, text);
            }
            results[i] = Some(TextBox { quad, text, score });
        }
        Ok(())
    }
}

/// 按宽度把文本行分批：批内行数 × 批内最大宽度不超过预算。
///
/// 单行自身就超过预算时单独成批，保证每行都会被识别。
fn plan_rec_batches(widths: &[usize]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|i| widths[*i]);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_w = 0usize;
    for i in order {
        let w = widths[i];
        let next_w = current_w.max(w);
        if !current.is_empty() && (current.len() + 1) * next_w > REC_BATCH_COLUMNS {
            batches.push(std::mem::take(&mut current));
            current_w = 0;
        }
        current_w = current_w.max(w);
        current.push(i);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// 创建 DirectML 检测会话并预热；不可用时返回 None，由调用方回退 CPU。
fn build_directml_session(env: &ort::environment::Environment, det_path: &Path) -> Option<Session> {
    perf_scope!("DirectML 会话创建与预热");
    let builder = Session::builder(env).ok()?;
    let mut builder = match builder.with_execution_providers([ort::ep::DirectML::default().build()])
    {
        Ok(builder) => builder,
        Err(err) => {
            eprintln!("[ocr] DirectML 不可用，检测走 CPU：{err}");
            return None;
        }
    };
    let mut session = match builder.commit_from_file(det_path) {
        Ok(session) => session,
        Err(err) => {
            eprintln!("[ocr] DirectML 不可用，检测走 CPU：{err}");
            return None;
        }
    };
    // 预热：首次推理要编译 D3D12 管线，放在后台线程里做，不阻塞识别。
    let warm = ndarray::Array4::<f32>::zeros((1, 3, DET_WARMUP_SIZE, DET_WARMUP_SIZE));
    let warmed = session
        .run(ort::inputs!["x" => TensorRef::from_array_view(&warm).unwrap()])
        .is_ok();
    if warmed {
        eprintln!("[ocr] 检测启用 DirectML");
        Some(session)
    } else {
        eprintln!("[ocr] DirectML 预热失败，检测走 CPU");
        None
    }
}

/// 探测 rec 模型的输出类别数：输出形状为 [N, T, 类别数]。
fn rec_num_classes(rec: &mut Session) -> Option<usize> {
    let probe = ndarray::Array4::<f32>::zeros((1, 3, REC_HEIGHT, 8));
    let out = rec
        .run(ort::inputs!["x" => TensorRef::from_array_view(&probe).ok()?])
        .ok()?;
    let shape = out[0].try_extract_tensor::<f32>().ok()?.0;
    Some(shape[2] as usize)
}

/// det 长边缩放到 960，并把两边对齐到 32 的倍数。
/// 官方 det 模型的 Resize 节点对尺寸有硬约束，非 32 倍数会直接报错。
fn det_resize(w: u32, h: u32) -> (u32, u32) {
    let (w, h) = (w as f32, h as f32);
    let ratio = if w > h {
        DET_RESIZE_LONG / w
    } else {
        DET_RESIZE_LONG / h
    };
    let nw = ((w * ratio) as u32).max(32);
    let nh = ((h * ratio) as u32).max(32);
    let align = |v: u32| ((v as f32 / 32.0).round() as u32).max(1) * 32;
    (align(nw), align(nh))
}

/// DB 后处理：二值化 → 连通域 → 过滤 → unclip 扩张 → 映射回原图坐标。
fn db_postprocess(prob: &[f32], hm: usize, wm: usize, w0: f32, h0: f32) -> Vec<[(f32, f32); 4]> {
    let at = |y: usize, x: usize| prob[y * wm + x];
    let mut visited = vec![false; hm * wm];
    let mut quads = Vec::new();

    for sy in 0..hm {
        for sx in 0..wm {
            let idx = sy * wm + sx;
            if visited[idx] || at(sy, sx) < DB_THRESH {
                continue;
            }
            visited[idx] = true;
            let mut stack = vec![(sx, sy)];
            let mut min_x = sx;
            let mut max_x = sx;
            let mut min_y = sy;
            let mut max_y = sy;
            let mut sum = 0.0f32;
            let mut count = 0usize;
            while let Some((x, y)) = stack.pop() {
                let p = at(y, x);
                sum += p;
                count += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= wm as i32 || ny >= hm as i32 {
                        continue;
                    }
                    let nidx = ny as usize * wm + nx as usize;
                    if !visited[nidx] && at(ny as usize, nx as usize) >= DB_THRESH {
                        visited[nidx] = true;
                        stack.push((nx as usize, ny as usize));
                    }
                }
            }
            let bw = (max_x - min_x) as f32 + 1.0;
            let bh = (max_y - min_y) as f32 + 1.0;
            if bw < 3.0 || bh < 3.0 {
                continue;
            }
            if (sum / count as f32) < DB_BOX_THRESH {
                continue;
            }

            // 概率图上的框（半开区间）。
            let fx0 = min_x as f32;
            let fx1 = max_x as f32 + 1.0;
            let fy0 = min_y as f32;
            let fy1 = max_y as f32 + 1.0;

            // unclip：官方 DB 用 pyclipper 按周长归一化外扩，距离 = 面积 * ratio / 周长。
            // 早期版本误用 sqrt(面积 * ratio) / 2，细长文本行会被撑成巨大的框。
            let area = bw * bh;
            let perimeter = 2.0 * (bw + bh);
            let expand = area * DB_UNCLIP_RATIO / perimeter;

            // 概率图尺寸 → 原图尺寸。
            let sx = w0 / wm as f32;
            let sy = h0 / hm as f32;
            // unclip 扩张可能越出边界，需钳制到原图范围内。
            let x0 = ((fx0 - expand) * sx).max(0.0);
            let x1 = ((fx1 + expand) * sx).min(w0);
            let y0 = ((fy0 - expand) * sy).max(0.0);
            let y1 = ((fy1 + expand) * sy).min(h0);
            quads.push([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]);
        }
    }
    quads.sort_by(|a, b| a[0].1.partial_cmp(&b[0].1).unwrap_or(std::cmp::Ordering::Equal));
    quads
}

/// 按四点框裁剪出文本行，并缩放到识别模型需要的高度。
fn crop_quad(image: &RgbaImage, quad: &[(f32, f32); 4]) -> Option<image::RgbImage> {
    let min_x = quad.iter().map(|p| p.0).fold(f32::MAX, f32::min);
    let min_y = quad.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let max_x = quad.iter().map(|p| p.0).fold(f32::MIN, f32::max);
    let max_y = quad.iter().map(|p| p.1).fold(f32::MIN, f32::max);
    let (w, h) = (image.width() as f32, image.height() as f32);
    // 右下角为开区间边界，向上取整保证不裁掉最后一个像素。
    let x0 = min_x.max(0.0).min(w - 1.0) as u32;
    let y0 = min_y.max(0.0).min(h - 1.0) as u32;
    let x1 = (max_x.ceil().max(x0 as f32 + 1.0)).min(w) as u32;
    let y1 = (max_y.ceil().max(y0 as f32 + 1.0)).min(h) as u32;
    if x1 <= x0 + 1 || y1 <= y0 + 1 {
        return None;
    }
    // crop_imm 的后两个参数是宽高，不是右下角坐标。
    let crop = image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
    // 识别模型只需三通道，转成 RGB 节省后续内存拷贝。
    let crop = image::DynamicImage::ImageRgba8(crop).to_rgb8();
    let ratio = (x1 - x0) as f32 / (y1 - y0) as f32;
    // 官方 rec 按文本行宽高比确定输入宽度：宽 = 高 * 宽高比。
    let target_w = ((REC_HEIGHT as f32 * ratio).round() as u32).max(16);
    let target_w = target_w.min(REC_MAX_WIDTH as u32);
    let resized = image::imageops::resize(
        &crop,
        target_w,
        REC_HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );
    Some(resized)
}
/// CTC greedy 解码：blank 为 0，索引 i 对应字典第 i-1 项。
fn ctc_decode(probs: &[f32], time_steps: usize, num_classes: usize, dict: &[String]) -> (String, f32) {
    let mut text = String::new();
    let mut previous = 0usize;
    let mut conf_sum = 0.0f32;
    let mut conf_count = 0usize;
    for t in 0..time_steps {
        let row = &probs[t * num_classes..(t + 1) * num_classes];
        let Some((index, &confidence)) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        else {
            continue;
        };
        if index != 0 && index != previous && index <= dict.len() {
            text.push_str(&dict[index - 1]);
            conf_sum += confidence;
            conf_count += 1;
        }
        previous = index;
    }
    let score = if conf_count > 0 {
        conf_sum / conf_count as f32
    } else {
        0.0
    };
    (text, score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_resize_aligns_to_multiple_of_32() {
        let (w, h) = det_resize(900, 200);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        assert!(w >= h, "长边应缩放到 960 附近并保持宽高比");
    }

    #[test]
    fn det_resize_handles_portrait() {
        let (w, h) = det_resize(200, 900);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        assert!(h >= w);
    }

    #[test]
    fn ctc_decode_skips_blank_and_repeats() {
        // 序列：1(a) 1(a) 0(blank) 2(b) 0 0 —— 应解出 "ab"。
        let dict = vec!["a".to_string(), "b".to_string()];
        let num_classes = 3;
        let time_steps = 6;
        let mut probs = vec![0.0f32; time_steps * num_classes];
        let set = |probs: &mut Vec<f32>, t: usize, index: usize| {
            probs[t * num_classes + index] = 1.0;
        };
        set(&mut probs, 0, 1);
        set(&mut probs, 1, 1);
        set(&mut probs, 2, 0);
        set(&mut probs, 3, 2);
        set(&mut probs, 4, 0);
        set(&mut probs, 5, 0);
        let (text, score) = ctc_decode(&probs, time_steps, num_classes, &dict);
        assert_eq!(text, "ab");
        assert!(score > 0.99);
    }

    #[test]
    fn ctc_decode_ignores_out_of_range_index() {
        let dict = vec!["a".to_string()];
        let num_classes = 4;
        let probs = vec![0.0, 0.0, 0.0, 1.0];
        let (text, _) = ctc_decode(&probs, 1, num_classes, &dict);
        assert_eq!(text, "");
    }

    #[test]
    fn db_postprocess_finds_bright_regions_and_maps_to_source() {
        // 构造 16x32 的概率图，中间 8x8 是高亮区域。
        let hm = 16;
        let wm = 32;
        let mut prob = vec![0.0f32; hm * wm];
        for y in 4..12 {
            for x in 12..20 {
                prob[y * wm + x] = 0.9;
            }
        }
        let quads = db_postprocess(&prob, hm, wm, 640.0, 320.0);
        assert_eq!(quads.len(), 1);
        let quad = quads[0];
        // 映射后应落在原图坐标范围内。
        assert!(quad.iter().all(|p| p.0 >= 0.0 && p.0 <= 640.0 && p.1 >= 0.0 && p.1 <= 320.0));
        // 高亮区域在概率图中央，映射到原图也应偏中央。
        let cx = (quad[0].0 + quad[2].0) / 2.0;
        assert!((cx - 320.0).abs() < 40.0, "框中心应接近原图中心，实际 {cx}");
    }

    #[test]
    fn plan_rec_batches_covers_every_line_exactly_once() {
        let widths = vec![100, 2500, 300, 3200, 80, 1200, 640];
        let batches = plan_rec_batches(&widths);
        let mut seen: Vec<usize> = batches.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..widths.len()).collect::<Vec<_>>(), "每行都要被分到某一批");
    }

    #[test]
    fn plan_rec_batches_respects_column_budget() {
        let widths = vec![100, 2500, 300, 3200, 80, 1200, 640];
        for batch in plan_rec_batches(&widths) {
            let max_w = batch.iter().map(|i| widths[*i]).max().unwrap();
            if batch.len() > 1 {
                assert!(
                    batch.len() * max_w <= REC_BATCH_COLUMNS,
                    "批次列数超预算：{} 行 × {} 列",
                    batch.len(),
                    max_w
                );
            }
        }
    }

    #[test]
    fn plan_rec_batches_keeps_narrow_lines_together() {
        // 窄行应聚成一批，避免被补齐到最宽行造成的算力浪费。
        let widths = vec![100, 120, 140, 3000];
        let batches = plan_rec_batches(&widths);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3]);
    }

    #[test]
    fn plan_rec_batches_handles_empty_input() {
        assert!(plan_rec_batches(&[]).is_empty());
    }

    #[test]
    fn db_postprocess_expands_thin_text_line_only_slightly() {
        // 细长文本行：概率图上 200x4 的高亮条，外扩应按 面积*ratio/周长 计算，
        // 而不是 sqrt(面积*ratio)/2（那会把框撑成巨大的方块）。
        let hm = 16;
        let wm = 256;
        let mut prob = vec![0.0f32; hm * wm];
        for y in 6..10 {
            for x in 10..210 {
                prob[y * wm + x] = 0.9;
            }
        }
        let quads = db_postprocess(&prob, hm, wm, 2560.0, 160.0);
        assert_eq!(quads.len(), 1);
        let quad = quads[0];
        let width = quad[2].0 - quad[0].0;
        let height = quad[2].1 - quad[0].1;
        // 外扩量 = 面积 * 1.5 / 周长 = 800 * 1.5 / 408 ≈ 2.94（概率图像素），
        // 概率图→原图缩放为 10 倍，故高度 ≈ (4 + 2 * 2.94) * 10。
        assert!(
            (height - 98.8).abs() < 1.0,
            "细长文本行的高度不应被撑大，实际 {height}"
        );
        assert!(
            (width - 2058.8).abs() < 2.0,
            "细长文本行的宽度不应被撑大，实际 {width}"
        );
    }

    #[test]
    fn crop_quad_crops_the_requested_region() {
        // 8x8 纯色图，取 4x2 的框：裁剪区域应为 4x2，再按官方规则缩放到高 48。
        let image = RgbaImage::from_pixel(8, 8, image::Rgba([10, 20, 30, 255]));
        let quad = [(2.0, 2.0), (6.0, 2.0), (6.0, 4.0), (2.0, 4.0)];
        let crop = crop_quad(&image, &quad).expect("裁剪失败");
        assert_eq!(crop.height(), REC_HEIGHT as u32);
        // 宽高比 4/2 = 2 → 宽度 = 48 * 2
        assert_eq!(crop.width(), 96);
    }

    #[test]
    fn crop_quad_resizes_to_rec_height() {
        let image = RgbaImage::from_pixel(100, 20, image::Rgba([10, 20, 30, 255]));
        let quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)];
        let crop = crop_quad(&image, &quad).expect("裁剪失败");
        assert_eq!(crop.height(), REC_HEIGHT as u32);
        // 宽度按宽高比 = 100/20 = 5 → 48 * 5 = 240
        assert_eq!(crop.width(), 240);
    }
}
