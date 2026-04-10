require "../vector_embedder"

module Vecstolite
  # A minimal lexical TF-style bag-of-words embedder. Useful for testing.
  struct LexicalEmbedder
    include VectorEmbedder

    DEFAULT_DIMS = 1024

    getter dimensions : Int32

    # Needed to get consistent hashes across runs
    private HASHER = Crystal::Hasher.new(1, 1)

    def initialize(@dimensions = DEFAULT_DIMS)
    end

    # Unique name for the embedding model
    def model_name : String
      "vecstolite/lexical"
    end

    # Returns a fixed-length Float32 vector for *text*.
    def embed(text : String) : Embedding
      vec = Embedding.new(dimensions, 0.0_f32)
      tokens(text).each do |token|
        idx = token.hash(HASHER).result.abs % dimensions
        vec[idx] += 1.0_f32
      end
      normalize(vec)
    end

    # Simple word-like tokens split out of the text
    private def tokens(text : String) : Array(String)
      text.downcase.scan(/[a-z']+/).map(&.[0])
    end

    private def normalize(vec : Embedding) : Embedding
      magnitude = Math.sqrt(VECM.sum(vec))
      return vec if magnitude == 0.0_f32

      vec.map { |x| x / magnitude }
    end
  end
end
