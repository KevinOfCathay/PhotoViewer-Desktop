//! 构建脚本：把应用图标嵌入 Windows exe 的 PE 资源。
//!
//! Windows 资源管理器在显示文件关联图标时，会从注册表中记录的
//! `DefaultIcon` 路径读取图标资源。若 exe 内没有图标资源，图标就会
//! 退化成系统默认的「应用程序」图标。因此这里把 `assets/icon.ico`
//! 编译进 exe 的资源表（资源 ID 1，即 IDI_ICON）。

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let mut res = winres::WindowsResource::new();
    res.set_icon("assets/icon.ico");
    res.set("FileDescription", "Photo Viewer");
    res.set("ProductName", "Photo Viewer");
    res.set("FileVersion", "0.1.0");
    res.set("ProductVersion", "0.1.0");
    res.set("LegalCopyright", "");
    res.set("CompanyName", "");

    if let Err(err) = res.compile() {
        // 资源编译失败不应中断构建，程序仍可运行，只是缺少图标。
        panic!("failed to embed icon resource: {err}");
    }
}
