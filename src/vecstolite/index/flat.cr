require "./strategy"
require "../store/repository"
require "./binary_heap"

module Vecstolite
  module Index
    # Exact nearest-neighbour search by scanning every live vector.
    #
    # There is no graph, so nothing is built at insert time, nothing is
    # persisted beyond the vectors themselves, and nothing needs rebuilding
    # after an unclean exit. Queries cost one pass over `vecsto_vectors`.
    #
    # Two uses: small stores, where a graph costs more than it saves; and as
    # the oracle a recall harness measures `HNSW` against.
    class Flat < Strategy
      def initialize(@repo : Repository)
      end

      # Vectors are already in the repository, so indexing is nothing.
      def add(entry_id : Int64, vector : Embedding) : Nil
      end

      def search(vector : Embedding,
                 k : Int32,
                 ef : Int32 = DEFAULT_EF_SEARCH,
                 allowed : Set(Int64)? = nil) : Array(Hit)
        return [] of Hit if k <= 0

        # Keep the k best seen so far in a max-heap keyed by distance, so the
        # worst is always the one to evict.
        worst_first = BinaryHeap(Hit).new { |this, that| this.score <= that.score }
        @repo.each_live_vector do |entry_id, candidate|
          next if allowed && !allowed.includes?(entry_id)

          score = 1.0_f32 - distance(vector, candidate)
          if worst_first.size < k
            worst_first.push(Hit.new(entry_id, score))
          elsif score > worst_first.peek.score
            worst_first.pop
            worst_first.push(Hit.new(entry_id, score))
          end
        end

        results = Array(Hit).new(worst_first.size)
        while !worst_first.empty?
          results << worst_first.pop
        end
        results.reverse!
      end

      def size : Int32
        @repo.live_count
      end

      def kind : Symbol
        :flat
      end

      def fully_persisted? : Bool
        true
      end

      def flush : Nil
      end

      def clear : Nil
      end
    end
  end
end
