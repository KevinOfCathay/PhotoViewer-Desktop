//! OCR 结果缓存：按图片路径缓存识别结果，避免来回切换时重复识别。
//!
//! 缓存只存在于内存中，程序退出即释放。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ocr::TextBox;

/// 默认缓存上限（张图片）。
pub const DEFAULT_CACHE_CAP: usize = 100;

/// OCR 结果的 LRU 缓存。
pub struct OcrCache {
    map: HashMap<PathBuf, Vec<TextBox>>,
    /// 由旧到新的使用顺序。
    order: Vec<PathBuf>,
    cap: usize,
}

impl OcrCache {
    pub fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: Vec::new(),
            cap,
        }
    }

    /// 命中缓存时返回结果并刷新使用顺序。
    pub fn get(&mut self, path: &Path) -> Option<&Vec<TextBox>> {
        if !self.map.contains_key(path) {
            return None;
        }
        self.touch(path);
        self.map.get(path)
    }

    /// 写入缓存，超出容量时淘汰最久未使用的图片。
    pub fn put(&mut self, path: PathBuf, boxes: Vec<TextBox>) {
        if self.map.insert(path.clone(), boxes).is_none() {
            self.order.push(path);
        } else {
            self.touch(&path);
        }
        while self.order.len() > self.cap {
            let oldest = self.order.remove(0);
            self.map.remove(&oldest);
        }
    }

    /// 删除缓存中的结果：图片被编辑后旧坐标失效。
    pub fn remove(&mut self, path: &Path) {
        self.map.remove(path);
        self.order.retain(|p| p != path);
    }

    /// 调整容量，超出部分立即淘汰。
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap;
        while self.order.len() > self.cap {
            let oldest = self.order.remove(0);
            self.map.remove(&oldest);
        }
    }

    /// 当前缓存的图片数量。
    pub fn len(&self) -> usize {
        self.map.len()
    }

    fn touch(&mut self, path: &Path) {
        if let Some(pos) = self.order.iter().position(|p| p == path) {
            let path = self.order.remove(pos);
            self.order.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxes(count: usize) -> Vec<TextBox> {
        (0..count)
            .map(|i| TextBox {
                quad: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                text: format!("text{i}"),
                score: 0.9,
            })
            .collect()
    }

    #[test]
    fn get_returns_result_and_refreshes_order() {
        let mut cache = OcrCache::new(2);
        let a = PathBuf::from("a.png");
        let b = PathBuf::from("b.png");
        cache.put(a.clone(), boxes(1));
        cache.put(b.clone(), boxes(1));

        assert!(cache.get(&a).is_some());
        // a 被访问过，淘汰时应先淘汰 b。
        cache.put(PathBuf::from("c.png"), boxes(1));
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&b).is_none());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn set_cap_evicts_immediately() {
        let mut cache = OcrCache::new(3);
        for name in ["a.png", "b.png", "c.png"] {
            cache.put(PathBuf::from(name), boxes(1));
        }
        cache.set_cap(1);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&PathBuf::from("c.png")).is_some());
    }
}
