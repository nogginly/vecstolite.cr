require "simd"

module Vecstolite
  # Leverage SIMD vector embedding maths.
  {% if flag?(:aarch64) %}
    # :nodoc:
    VECM = SIMD::NEON.new
  {% else %}
    # :nodoc:
    VECM = SIMD.instance
  {% end %}

  # The vector embedding is a slice of Float32 values
  alias Embedding = Slice(Float32)

  module VectorEmbedder
    # Unique name for the embedding model
    abstract def model_name : String

    # Returns a fixed-length Float32 vector embedding of the given text
    abstract def embed(text : String) : Embedding

    # Returns the number of dimensions
    abstract def dimensions : Int32
  end
end
