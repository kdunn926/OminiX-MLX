//! Memory management utilities for MLX on Apple Silicon.
//!
//! Provides safe wrappers around `mlx_sys` memory introspection and limit-setting
//! functions. On Apple Silicon, unified memory means GPU memory IS system memory;
//! `DeviceInfo::memory_size` is the total physical RAM.

/// Snapshot of current MLX memory state.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Bytes currently held by live MLX arrays (cannot be reclaimed without freeing the array).
    pub active_bytes: usize,
    /// Bytes in MLX's reclaimable cache (can be freed by `clear_cache()`).
    pub cache_bytes: usize,
    /// Peak bytes since the last `reset_peak()` call.
    pub peak_bytes: usize,
    /// Current MLX allocator soft limit (0 = unlimited).
    pub memory_limit_bytes: usize,
}

impl MemoryStats {
    /// Resident bytes = active + cache (total currently allocated from Metal).
    pub fn resident_bytes(&self) -> usize {
        self.active_bytes + self.cache_bytes
    }

    /// Fraction of memory_limit currently resident (0.0 if limit is 0).
    pub fn resident_fraction(&self) -> f64 {
        if self.memory_limit_bytes == 0 {
            0.0
        } else {
            self.resident_bytes() as f64 / self.memory_limit_bytes as f64
        }
    }
}

/// Key Metal/MLX device limits for Apple Silicon.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct DeviceInfo {
    /// Total physical RAM (= unified GPU memory on Apple Silicon).
    pub memory_size: usize,
    /// Metal's recommended maximum wired working-set size.
    pub max_recommended_working_set_size: usize,
    /// Largest single contiguous buffer Metal will allow.
    pub max_buffer_length: usize,
}

/// Read current MLX memory statistics.
///
/// Returns `None` if any MLX call fails (should be extremely rare).
pub fn get_memory_stats() -> Option<MemoryStats> {
    unsafe {
        let mut active: usize = 0;
        let mut cache: usize = 0;
        let mut peak: usize = 0;
        let mut limit: usize = 0;
        if mlx_sys::mlx_get_active_memory(&mut active) != 0 {
            return None;
        }
        if mlx_sys::mlx_get_cache_memory(&mut cache) != 0 {
            return None;
        }
        if mlx_sys::mlx_get_peak_memory(&mut peak) != 0 {
            return None;
        }
        // mlx_get_memory_limit returns the current soft cap (0 = unlimited)
        let _ = mlx_sys::mlx_get_memory_limit(&mut limit);
        Some(MemoryStats {
            active_bytes: active,
            cache_bytes: cache,
            peak_bytes: peak,
            memory_limit_bytes: limit,
        })
    }
}

/// Return Metal device information (memory sizes, buffer limits).
///
/// mlx-c 0.6 replaced the typed `mlx_metal_device_info` struct with a generic
/// key-value lookup on a `mlx_device_info` object. Missing keys return 0 — on
/// Apple Silicon all three are populated by the Metal backend.
pub fn get_device_info() -> DeviceInfo {
    unsafe {
        let mut dev = mlx_sys::mlx_device_new();
        if mlx_sys::mlx_get_default_device(&mut dev) != 0 {
            mlx_sys::mlx_device_free(dev);
            return DeviceInfo::default();
        }
        let info = mlx_sys::mlx_device_info_new();
        let mut info_local = info;
        if mlx_sys::mlx_device_info_get(&mut info_local, dev) != 0 {
            mlx_sys::mlx_device_info_free(info);
            mlx_sys::mlx_device_free(dev);
            return DeviceInfo::default();
        }
        let memory_size = read_size_key(info_local, c"memory_size");
        let max_recommended_working_set_size =
            read_size_key(info_local, c"max_recommended_working_set_size");
        let max_buffer_length = read_size_key(info_local, c"max_buffer_length");
        mlx_sys::mlx_device_info_free(info_local);
        mlx_sys::mlx_device_free(dev);
        DeviceInfo {
            memory_size,
            max_recommended_working_set_size,
            max_buffer_length,
        }
    }
}

unsafe fn read_size_key(info: mlx_sys::mlx_device_info, key: &core::ffi::CStr) -> usize {
    let mut value: usize = 0;
    if mlx_sys::mlx_device_info_get_size(&mut value, info, key.as_ptr()) != 0 {
        return 0;
    }
    value
}


/// Free MLX's reclaimable tensor cache.
///
/// Does not touch live arrays. Returns `Ok(())` on success.
pub fn clear_cache() -> Result<(), String> {
    let rc = unsafe { mlx_sys::mlx_clear_cache() };
    if rc != 0 {
        Err("mlx_clear_cache failed".to_string())
    } else {
        Ok(())
    }
}

/// Set the MLX allocator soft limit (bytes).
///
/// When the allocator would exceed this limit, MLX flushes its cache and, if still over,
/// returns an allocation error instead of committing more Metal memory. This makes OOM
/// recoverable rather than fatal.
///
/// Returns the *previous* limit on success.
pub fn set_memory_limit(limit: usize) -> Result<usize, String> {
    let mut prev: usize = 0;
    let rc = unsafe { mlx_sys::mlx_set_memory_limit(&mut prev, limit) };
    if rc != 0 {
        Err(format!("mlx_set_memory_limit({limit}) failed"))
    } else {
        Ok(prev)
    }
}

/// Set the MLX reclaimable cache size limit (bytes).
///
/// MLX will flush its cache rather than grow it past this limit.
///
/// Returns the *previous* limit on success.
pub fn set_cache_limit(limit: usize) -> Result<usize, String> {
    let mut prev: usize = 0;
    let rc = unsafe { mlx_sys::mlx_set_cache_limit(&mut prev, limit) };
    if rc != 0 {
        Err(format!("mlx_set_cache_limit({limit}) failed"))
    } else {
        Ok(prev)
    }
}

/// Set the Metal wired memory limit (bytes).
///
/// Controls how much memory Metal may keep wired (non-pageable). Should be set to
/// `DeviceInfo::max_recommended_working_set_size` for best balance of performance
/// and system stability.
///
/// Returns the *previous* limit on success.
pub fn set_wired_limit(limit: usize) -> Result<usize, String> {
    let mut prev: usize = 0;
    let rc = unsafe { mlx_sys::mlx_set_wired_limit(&mut prev, limit) };
    if rc != 0 {
        Err(format!("mlx_set_wired_limit({limit}) failed"))
    } else {
        Ok(prev)
    }
}

/// Reset the peak memory counter to zero.
pub fn reset_peak_memory() -> Result<(), String> {
    let rc = unsafe { mlx_sys::mlx_reset_peak_memory() };
    if rc != 0 {
        Err("mlx_reset_peak_memory failed".to_string())
    } else {
        Ok(())
    }
}

/// Flush the MLX cache only when `active + cache > limit * threshold`.
///
/// Returns `true` if a flush was performed.
pub fn flush_cache_if_needed(limit: usize, threshold: f64) -> bool {
    let Some(stats) = get_memory_stats() else {
        return false;
    };
    if stats.resident_bytes() as f64 > limit as f64 * threshold {
        let _ = clear_cache();
        return true;
    }
    false
}
