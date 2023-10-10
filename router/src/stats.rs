
#[derive(Clone, Debug)]
pub(crate) struct ServerStats {
    kv_cache_usage: f32,
}

impl ServerStats {
    pub(crate) fn new() -> Self {
        Self { kv_cache_usage: 0.0 }
    }

    pub(crate) fn log_stats(&self) {
        // log kv_cache_usage with precision of 2
        tracing::info!("KV cache usage: {:.2}%", self.kv_cache_usage * 100.0);
        // tracing::info!("KV cache usage: {.2}%", self.kv_cache_usage);
    }

    pub(crate) fn update_kv_cache_usage(&mut self, kv_cache_usage: f32) {
        self.kv_cache_usage = kv_cache_usage;
    }
}