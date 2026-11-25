//! 性能基准日志：仅 debug 版本输出。
//!
//! 每条日志带上「启动至今」的时间，用来定位从进程启动到各个操作的耗时。
//! release 版本由 `#[cfg(debug_assertions)]` 编译期剔除，不产生任何输出与格式化开销。

use std::sync::OnceLock;
use std::time::Instant;
#[cfg(debug_assertions)]
use std::time::Duration;

/// 进程计时起点，首次访问时锚定。
static START: OnceLock<Instant> = OnceLock::new();

/// 锚定计时起点，需在 main 最开始调用。
pub fn init() {
    START.get_or_init(Instant::now);
}

/// 毫秒数，保留两位小数以便观察亚毫秒级操作。
#[cfg(debug_assertions)]
pub fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// 启动至今的时间。
#[cfg(debug_assertions)]
pub fn uptime() -> Duration {
    START.get_or_init(Instant::now).elapsed()
}

/// 输出一条性能日志，自动带上启动至今的时间。
#[cfg(debug_assertions)]
pub fn emit(args: std::fmt::Arguments<'_>) {
    eprintln!("[perf] +{}ms {}", uptime().as_millis(), args);
}

/// 作用域计时器：离开作用域时打印本段耗时。
#[cfg(debug_assertions)]
pub struct Timer {
    label: String,
    start: Instant,
}

#[cfg(debug_assertions)]
impl Timer {
    /// 开始计时。
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            start: Instant::now(),
        }
    }

    /// 已经过的时间。
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(debug_assertions)]
impl Drop for Timer {
    fn drop(&mut self) {
        emit(format_args!("{}: {:.2}ms", self.label, ms(self.elapsed())));
    }
}

/// 打一条性能日志；release 版本不输出任何内容。
macro_rules! perf_log {
    ($($arg:tt)*) => {{
        #[cfg(debug_assertions)]
        $crate::perf::emit(::core::format_args!($($arg)*));
    }};
}

/// 作用域计时：进入时开始计时，离开当前作用域时打印耗时；release 版本既不计时也不输出。
///
/// 展开成一个 `let` 绑定，计时器随所在作用域结束而析构，因此必须写在语句位置。
macro_rules! perf_scope {
    ($label:expr) => {
        #[cfg(debug_assertions)]
        let _perf_timer = $crate::perf::Timer::new($label);
    };
}

/// 端到端耗时打点：起点取出后算出距今耗时，追加到消息末尾；release 版本整段剔除。
macro_rules! perf_since {
    ($started:expr, $fmt:literal, $($arg:expr),* $(,)?) => {{
        #[cfg(debug_assertions)]
        {
            let started = $started.take();
            if let Some(started) = started {
                $crate::perf::emit(::core::format_args!(
                    concat!($fmt, "，端到端 {:.2}ms"),
                    $($arg,)* $crate::perf::ms(started.elapsed())
                ));
            }
        }
    }};
}
