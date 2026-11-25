//! 同目录图片枚举与自然排序。

use std::cmp::Ordering;
use std::path::{Path, PathBuf};


/// image crate 默认特性可解码的图片扩展名。
const EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif", "ico", "qoi", "pnm", "ppm", "pgm",
    "tga", "dds", "ff", "farbfeld", "hdr", "exr",
];

/// 列出目录下的图片，按文件名自然排序（img2 排在 img10 前）。
pub fn list_images(dir: &Path) -> Vec<PathBuf> {
    perf_scope!("枚举目录图片");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_image(path))
        .collect();
    files.sort_by(|a, b| natural_cmp(&file_name(a), &file_name(b)));
    files
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// 自然排序：数字段按数值比较，其余按文本比较。
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let mut left = tokens(a).into_iter();
    let mut right = tokens(b).into_iter();
    loop {
        match (left.next(), right.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => {
                let ord = match (x, y) {
                    (Token::Num(n), Token::Num(m)) => n.cmp(&m),
                    (Token::Text(s), Token::Text(t)) => s.cmp(&t),
                    (Token::Num(_), Token::Text(_)) => Ordering::Less,
                    (Token::Text(_), Token::Num(_)) => Ordering::Greater,
                };
                if ord != Ordering::Equal {
                    return ord;
                }
            }
        }
    }
}

enum Token {
    Num(u128),
    Text(String),
}

/// 把文件名切成数字段与文本段，数字段去掉前导零后按数值比较。
fn tokens(s: &str) -> Vec<Token> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        if bytes[i].is_ascii_digit() {
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let digits = s[start..i].trim_start_matches('0');
            let value = if digits.is_empty() {
                0
            } else {
                digits.parse::<u128>().unwrap_or(u128::MAX)
            };
            out.push(Token::Num(value));
        } else {
            while i < bytes.len() && !bytes[i].is_ascii_digit() {
                i += 1;
            }
            out.push(Token::Text(s[start..i].to_lowercase()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(names: &[&str]) -> Vec<String> {
        let mut list: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        list.sort_by(|a, b| natural_cmp(a, b));
        list
    }

    #[test]
    fn natural_sort_keeps_numeric_order() {
        assert_eq!(
            sorted(&["img10.png", "img2.png", "img1.png"]),
            ["img1.png", "img2.png", "img10.png"]
        );
        assert_eq!(sorted(&["a2b", "a10b", "a1b"]), ["a1b", "a2b", "a10b"]);
    }

    #[test]
    fn list_images_filters_and_sorts() {
        let dir = std::env::temp_dir().join("photo-viewer-gallery-test");
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["b.png", "a10.png", "a2.png", "note.txt", "c.JPEG"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }

        let files = list_images(&dir);
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["a2.png", "a10.png", "b.png", "c.JPEG"]);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
