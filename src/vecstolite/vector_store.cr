require "./vector_embedder"

# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

module Vecstolite
  module VectorStore
    # One stored item: the original text plus its embedding vector.
    record Entry, text : String, vector : Embedding, extra : String?

    # A single search result returned from VectorStore#search.
    record SearchResult, text : String, score : Float32, extra : String?

    # Add and index `text`, with optional `extra` data (not embedded or indexed)
    abstract def add(text : String, extra : String? = nil) : Nil

    # Search for `k` entries that are most similar to `query`
    abstract def search(query : String, k : Int32 = 5) : Array(SearchResult)

    abstract def size : Int32
    abstract def clear : Nil
  end
end
