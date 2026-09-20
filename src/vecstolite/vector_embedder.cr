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

    # Returns a fixed-length Float32 vector embedding of the given text
    abstract def embed(text : String) : Embedding

    # Returns the number of dimensions
    abstract def dimensions : Int32

    # Embeds several texts. Implementations backed by a remote service should
    # override this to make one request instead of one per text.
    def embed_all(texts : Array(String)) : Array(Embedding)
      texts.map { |text| embed(text) }
    end
  end
end
