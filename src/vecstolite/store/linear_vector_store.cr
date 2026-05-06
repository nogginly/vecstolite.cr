require "../vector_store"

module Vecstolite
  # An in-memory vector store with linear search .
  #
  # Usage:
  # ```
  # store = Vecstolite::LinearVectorStore.new(embedder)
  # store.add("The sky is blue.")
  # store.add("Crystal is fast.")
  # results = store.search("colour of the sky", k: 3)
  # ```
  class LinearVectorStore
    include VectorStore(String)

    # One stored item in a vector store: the original text plus its embedding vector and optional extra string.
    record Entry, text : String, vector : Embedding, extra : String?

    # A single search result returned by `VectorStore#search`.
    record SearchResult, text : String, score : Float32, extra : String? do
      include VectorSearchResult
    end

    @entries : Array(Entry)
    @embedder : VectorEmbedder

    # Create a vector store that uses the given `embedder`
    def initialize(@embedder)
      @entries = [] of Entry
    end

    # Add and index `text`, with optional `meta` data (not embedded or indexed)
    def add(text : String, meta : String? = nil) : Nil
      vector = @embedder.embed(text)
      @entries << Entry.new(text, vector, meta)
    end

    # Return the top-`k` entries most similar to `query`, sorted descending.
    def search(query : String, k : Int32 = DEFAULT_K) : Array(VectorSearchResult)
      return [] of VectorSearchResult if @entries.empty?

      query_vec = @embedder.embed(query)

      @entries
        .map { |entry| SearchResult.new(entry.text, cosine_similarity(query_vec, entry.vector), entry.extra) }
        .sort_by! { |result| -result.score }
        .first(k)
        .map(&.as(VectorSearchResult))
    end

    # Total number of stored chunks.
    def size : Int32
      @entries.size
    end

    # Remove all entries.
    def clear : Nil
      @entries.clear
    end

    # Cosine similarity between two unit vectors (dot product).
    # Assumes both vectors are already L2-normalised by the embedder.
    private def cosine_similarity(a : Embedding, b : Embedding) : Float32
      raise ArgumentError.new("Vector length mismatch") if a.size != b.size

      # dot product: a.zip(b).sum { |x, y| x * y }
      VECM.dot(a, b)
    end
  end
end
