require "./vector_embedder"

# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

module Vecstolite
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
