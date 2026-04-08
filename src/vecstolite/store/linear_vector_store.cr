require "../vector_store"

module Vecstolite
  class LinearVectorStore
    include VectorStore

    @entries : Array(Entry)
    @embedder : VectorEmbedder

    # Create a vector store that uses the given `embedder`
    def initialize(@embedder)
      @entries = [] of Entry
    end

    # Add `text` to the vector store. The embedding is computed once and cached.
    def add(text : String) : Nil
      vector = @embedder.embed(text)
      @entries << Entry.new(text, vector)
    end

    # Return the top-`k` entries most similar to `query`, sorted descending.
    def search(query : String, k : Int32 = 5) : Array(SearchResult)
      return [] of SearchResult if @entries.empty?

      query_vec = @embedder.embed(query)

      @entries
        .map { |e| SearchResult.new(e.text, cosine_similarity(query_vec, e.vector)) }
        .sort_by! { |r| -r.score }
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
