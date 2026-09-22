require "../vector_embedder"

module Vecstolite
  module Index
    # One node in a navigable graph: the vector it stands for, and the
    # neighbours it links to at each layer.
    #
    # Nodes are reference types, so a strategy holding one and mutating its
    # neighbour lists must hand it back to the node cache with `write_back`
    # for the change to reach storage.
    class Node
      property vector : Embedding

      # `neighbours[layer]` holds the graph positions linked at that layer.
      property neighbours : Array(Array(Int32))

      def initialize(@vector : Embedding, max_layer : Int32)
        @neighbours = Array(Array(Int32)).new(max_layer + 1) { [] of Int32 }
      end
    end
  end
end
