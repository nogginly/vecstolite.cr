require "./vector_embedder"

module Vecstolite
  # Memory size convenience constants for `cache_max_bytes`.
  # Example: `CacheMode.lru(256 * Vecstolite::MB)`
  KB = 1_024_i64
  MB = 1_024_i64 * KB
  GB = 1_024_i64 * MB
end
