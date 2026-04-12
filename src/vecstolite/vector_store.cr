require "./vector_embedder"

# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

module Vecstolite
  module VectorStore
    DEFAULT_K = 5

    # One stored item in a vector store: the original text plus its embedding vector and optional extra string.
    record Entry, text : String, vector : Embedding, extra : String?

    # A single search result returned by `VectorStore#search`.
    record SearchResult, text : String, score : Float32, extra : String?

    # Add and index `text`, with optional `extra` data (not embedded or indexed).
    abstract def add(text : String, extra : String? = nil) : Nil

    # Search for `k` entries that are most similar to `query`.
    abstract def search(query : String, k : Int32 = DEFAULT_K) : Array(SearchResult)

    # Total number of entries in the store.
    abstract def size : Int32

    # Empty the store.
    abstract def clear : Nil
  end
end
