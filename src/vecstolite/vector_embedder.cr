require "simd"

module Vecstolite
  # Leverage SIMD vector embedding maths.
  # :nodoc:
  VECM = SIMD.scalar # This is faster with latest Crystal due to LLVM upgrade.

  # The vector embedding is a slice of Float32 values
  alias Embedding = Slice(Float32)

  module VectorEmbedder
    # Unique name for the embedding model
    abstract def model_name : String

    # Returns a fixed-length Float32 vector embedding of the given text.
    #
    # The result must be L2-normalised: every distance in this shard is a dot
    # product, which only equals cosine similarity for unit vectors. Use
    # `l2_normalize!` unless the model already guarantees it.
    abstract def embed(text : String) : Embedding

    # Returns the number of dimensions `embed` actually produces. A model
    # whose output is truncated must report the truncated size, since a store
    # writes this into its metadata and validates every vector against it.
    abstract def dimensions : Int32

    # Embeds several texts. Implementations backed by a remote service should
    # override this to make one request instead of one per text.
    def embed_all(texts : Array(String)) : Array(Embedding)
      texts.map { |text| embed(text) }
    end

    # Scales *vector* to unit length in place, and returns it. A zero vector
    # is left alone: it has no direction to preserve.
    def l2_normalize!(vector : Embedding) : Embedding
      magnitude = Math.sqrt(VECM.dot(vector, vector))
      return vector if magnitude == 0.0_f32

      vector.size.times { |i| vector[i] /= magnitude }
      vector
    end
  end
end
