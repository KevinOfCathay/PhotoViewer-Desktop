//! OCR 后台工作线程：在独立线程里跑 ONNX 推理，避免阻塞界面。

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};

use image::RgbaImage;

use crate::ocr::{OcrEngine, TextBox};

/// 发送给工作线程的任务。
pub enum Job {
    /// 识别指定图片。
    Recognize { token: u64, path: PathBuf, image: RgbaImage },
}

/// 工作线程回传的结果。
pub enum Msg {
    /// 识别过程中的阶段进度。
    Progress {
        token: u64,
        fraction: f32,
        label: String,
    },
    /// 识别完成，失败时携带可展示的错误信息。
    Done {
        token: u64,
        result: Result<Vec<TextBox>, String>,
    },
}

/// OCR 后台加载器：串行处理识别任务。
pub struct OcrWorker {
    jobs: Sender<Job>,
    /// 后台线程回传的结果。
    pub msgs: Receiver<Msg>,
}

impl OcrWorker {
    /// 启动工作线程。引擎延迟到首个任务时才加载，避免启动即占用内存。
    pub fn spawn(model_dir: PathBuf) -> Self {
        let (job_tx, job_rx) = channel::<Job>();
        let (msg_tx, msg_rx) = channel::<Msg>();
        std::thread::spawn(move || {
            let mut engine: Option<OcrEngine> = None;
            while let Ok(job) = job_rx.recv() {
                match job {
                    Job::Recognize { token, path, image } => {
                        // 标签表达式放在宏内，release 版本不会被求值。
                        perf_scope!(format!(
                            "识别任务 {}",
                            path.file_name()
                                .map(|name| name.to_string_lossy().into_owned())
                                .unwrap_or_else(|| path.to_string_lossy().into_owned())
                        ));
                        if engine.is_none() {
                            let _ = msg_tx.send(Msg::Progress {
                                token,
                                fraction: 0.05,
                                label: "加载模型".to_string(),
                            });
                            match OcrEngine::load(&model_dir) {
                                Ok(loaded) => engine = Some(loaded),
                                Err(err) => {
                                    eprintln!("[ocr] 加载模型失败: {err}");
                                    let _ = msg_tx.send(Msg::Done {
                                        token,
                                        result: Err(err),
                                    });
                                    continue;
                                }
                            }
                        }
                        let (tx, token) = (&msg_tx, token);
                        let result = engine
                            .as_mut()
                            .expect("引擎已加载")
                            .recognize(&image, |fraction, label| {
                                let _ = tx.send(Msg::Progress {
                                    token,
                                    fraction,
                                    label: label.to_string(),
                                });
                            })
                            .map_err(|err| {
                                eprintln!("[ocr] 识别 {:?} 失败: {err}", path);
                                err
                            });
                        if let Ok(boxes) = &result {
                            eprintln!("[ocr] {:?} 识别到 {} 个文本区域", path, boxes.len());
                        }
                        let _ = msg_tx.send(Msg::Done { token, result });
                    }
                }
            }
        });
        Self {
            jobs: job_tx,
            msgs: msg_rx,
        }
    }

    /// 提交识别任务。
    pub fn recognize(&self, token: u64, path: PathBuf, image: RgbaImage) {
        let _ = self.jobs.send(Job::Recognize { token, path, image });
    }
}

/// 默认识别模型目录。按以下顺序查找首个存在的目录：
/// 1. 可执行文件旁的 `models/`
/// 2. 可执行文件上两级（target/release → 项目根）的 `models/`
/// 3. 当前工作目录的 `models/`
pub fn default_model_dir() -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("models"));
        // target/release/photo-viewer.exe → 上两级是项目根目录。
        if let Some(root) = dir.parent().and_then(|p| p.parent()) {
            candidates.push(root.join("models"));
        }
    }
    candidates.push(Path::new("models").to_path_buf());
    candidates
        .into_iter()
        .find(|dir| dir.join("det.onnx").exists())
        .unwrap_or_else(|| Path::new("models").to_path_buf())
}
