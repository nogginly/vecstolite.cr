require "./vector_embedder"

# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

module Vecstolite
  # Memory size convenience constants for `cache_max_bytes`.
  # Example: `SQLitePayloadVectorStore.create(path, embedder, cache_max_bytes: 256 * Vecstolite::MB)`
  KB = 1_024_i64
  MB = 1_024_i64 * KB
  GB = 1_024_i64 * MB

  module VectorSearchResult
    abstract def text : String
    abstract def score : Float32
  end

  # Base definition for a vector store
  module VectorStore(M)
    DEFAULT_K = 5

    # Add and index `text`, with optional `meta` data (not embedded or indexed).
    abstract def add(text : String, meta : M? = nil) : Nil

    # Search for `k` entries that are most similar to `query`.
    abstract def search(query : String, k : Int32 = DEFAULT_K) : Array

    # Total number of entries in the store.
    abstract def size : Int32

    # Empty the store.
    abstract def clear : Nil
  end
end
