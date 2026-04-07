require "../vector_embedder"

module Vecstolite
  # A minimal lexical TF-style bag-of-words embedder. Useful for testing.
  struct LexicalEmbedder
    include VectorEmbedder

    VOCAB_SIZE = 1024

    # Returns a fixed-length Float32 vector for *text*.
    def embed(text : String) : Embedding
      vec = Embedding.new(VOCAB_SIZE, 0.0_f32)
      tokens(text).each do |token|
        idx = token.hash.abs % VOCAB_SIZE
        vec[idx] += 1.0_f32
      end
      normalize(vec)
    end

    # Returns the number of dimensions
    def dimensions : Int32
      VOCAB_SIZE
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
