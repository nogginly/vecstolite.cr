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
    include VectorStore

    @entries : Array(Entry)
    @embedder : VectorEmbedder

    # Create a vector store that uses the given `embedder`
    def initialize(@embedder)
      @entries = [] of Entry
    end

    # Add and index `text`, with optional `extra` data (not embedded or indexed)
    def add(text : String, extra : String? = nil) : Nil
      vector = @embedder.embed(text)
      @entries << Entry.new(text, vector, extra)
    end

    # Return the top-`k` entries most similar to `query`, sorted descending.
    def search(query : String, k : Int32 = DEFAULT_K) : Array(SearchResult)
      return [] of SearchResult if @entries.empty?

      query_vec = @embedder.embed(query)

      @entries
        .map { |entry| SearchResult.new(entry.text, cosine_similarity(query_vec, entry.vector), entry.extra) }
        .sort_by! { |result| -result.score }
        .first(k)
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
