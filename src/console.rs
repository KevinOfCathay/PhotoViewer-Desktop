//! 无控制台时把 stderr 重定向到日志文件，保证 eprintln! 诊断日志不丢失。

#[cfg(windows)]
use std::path::{Path, PathBuf};

/// 日志文件大小上限，超过后下次启动重新记录，避免无限增长。
#[cfg(windows)]
const MAX_LOG_BYTES: u64 = 4 * 1024 * 1024;

/// 启动最早期调用：进程没有可写 stderr 时改写到日志文件。
#[cfg(windows)]
pub fn redirect_stderr_to_log() {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Console::{GetStdHandle, SetStdHandle, STD_ERROR_HANDLE};

    let current = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
    if !current.is_null() && current != INVALID_HANDLE_VALUE {
        // 已继承到有效 stderr（从终端启动），保持终端输出。
        return;
    }
    let Some(path) = log_path() else { return };
    let Ok(file) = open_log(&path) else { return };
    unsafe { SetStdHandle(STD_ERROR_HANDLE, file.as_raw_handle()) };
    // 句柄必须存活到进程结束，否则后续写入会失败。
    std::mem::forget(file);
    eprintln!("[console] stderr 已重定向到 {}", path.display());
}

/// 非 Windows 平台没有控制台概念，无需处理。
#[cfg(not(windows))]
pub fn redirect_stderr_to_log() {}

/// 日志文件路径：%LOCALAPPDATA%\photo-viewer\photo-viewer.log。
#[cfg(windows)]
fn log_path() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")?;
    let dir = PathBuf::from(base).join("photo-viewer");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("photo-viewer.log"))
}

/// 以追加方式打开日志文件，超限时先清空。
#[cfg(windows)]
fn open_log(path: &Path) -> std::io::Result<std::fs::File> {
    use std::fs::{File, OpenOptions};

    if let Ok(meta) = std::fs::metadata(path)
        && meta.len() > MAX_LOG_BYTES
    {
        let _ = File::create(path);
    }
    OpenOptions::new().create(true).append(true).open(path)
}
