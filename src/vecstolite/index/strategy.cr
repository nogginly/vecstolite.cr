require "../vector_embedder"

module Vecstolite
  # Search strategies over the stored vectors.
  #
  # A strategy answers nearest-neighbour queries in terms of stable entry ids:
  # graph positions, where a strategy uses them at all, never leave this layer.
  #
  # - `Index::Flat` scans every live vector. Exact, O(n) per query, no graph to
  #   build, persist or rebuild.
  # - `Index::HNSW` walks a navigable small-world graph. Approximate, fast at
  #   scale, and pays for it with the graph it maintains.
  module Index
    # One search result: the entry that matched and its cosine similarity,
    # where 1.0 is identical.
    record Hit, entry_id : Int64, score : Float32

    abstract class Strategy
      # Indexes *vector* under the stable *entry_id*.
      abstract def add(entry_id : Int64, vector : Embedding) : Nil

      # Returns up to *k* hits, nearest first. *ef* is the beam width for
      # strategies that search approximately, and is ignored by exact ones.
      # When *allowed* is given, only those entry ids may appear in the result.
      abstract def search(vector : Embedding,
                          k : Int32,
                          ef : Int32 = DEFAULT_EF_SEARCH,
                          allowed : Set(Int64)? = nil) : Array(Hit)

      # How many entries the strategy currently indexes, tombstones included.
      abstract def size : Int32

      # Identifies the strategy in stored metadata, so a store can tell when it
      # has been asked to open with a different one.
      abstract def kind : Symbol

      # True when nothing is held only in memory.
      abstract def fully_persisted? : Bool

      # Writes anything held only in memory. Callers run this inside the
      # transaction that records the accompanying metadata.
      abstract def flush : Nil

      # Discards all indexed state, leaving the strategy ready to be rebuilt
      # from the stored vectors.
      abstract def clear : Nil

      DEFAULT_EF_SEARCH = 50

      # Angular distance: 1 - cosine similarity. Both vectors must be
      # L2-normalised, which is what makes the dot product sufficient.
      protected def distance(a : Embedding, b : Embedding) : Float32
        1.0_f32 - VECM.dot(a, b)
      end
    end
  end
end
