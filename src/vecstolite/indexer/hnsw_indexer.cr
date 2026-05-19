require "./binary_heap"
require "../vector_embedder"

module Vecstolite
  # :nodoc:
  # Hierarchical Navigable Small World (HNSW) approximate nearest neighbour index.
  # Drop-in companion to vector_store.cr — use Index when your store exceeds
  # ~50k entries and linear scan latency becomes a problem.
  #
  # Usage:
  #   index = Index.new(dims: 1536)
  #   index.add(id: 0, vector: vec0)
  #   index.add(id: 1, vector: vec1)
  #   results = index.search(query_vec, k: 5)
  #   results.each { |r| puts "id=#{r.id}  score=#{r.score.round(4)}" }
  #
  # The index stores integer IDs only — map them back to your Entry objects in
  # VectorStore via a simple Array or Hash.
  # ---------------------------------------------------------------------------
  # Tuning knobs
  # ---------------------------------------------------------------------------
  #
  # M              — max neighbours per node per layer (default 16).
  #                  Higher → better recall, ~M² memory per node, slower inserts.
  # EF_CONSTRUCTION — beam width when inserting a new node (default 200).
  #                  Higher → better graph quality, slower inserts.
  # EF_SEARCH       — beam width during queries (default 50, override at call site).
  #                  Higher → better recall, slower queries.
  #
  # Practical starting point for 1536-dim embeddings:
  #   M=16, ef_construction=200, ef_search=50 → ~95% recall at ~1ms/query (100k entries)
  #   M=32, ef_construction=400, ef_search=100 → ~99% recall at ~3ms/query
  module HNSW
    # One candidate in the greedy search beam — tracked in a max-heap by distance.
    private struct Candidate
      include Comparable(Candidate)

      property id : Int32
      property dist : Float32 # lower = closer (1 - cosine_similarity)

      def initialize(@id, @dist)
      end

      # Max-heap: the "largest" candidate is the most distant.
      def <=>(other : Candidate)
        dist <=> other.dist
      end
    end

    # One node in the HNSW graph.
    class HNSWNode
      property vector : Embedding
      # neighbours[layer] = array of neighbour IDs at that layer
      property neighbours : Array(Array(Int32))

      def initialize(@vector : Embedding, max_layer : Int32, m : Int32)
        @neighbours = Array(Array(Int32)).new(max_layer + 1) { [] of Int32 }
      end
    end

    # NodeStore is the abstract interface for storing HNSW graph nodes
    # (vector + neighbour lists). It is the single source of truth for all
    # node data — replacing the former split between @entry_embeddings and
    # the index's own @nodes array.
    #
    # Phase 1 provides only MemoryNodeStore.
    # Phases 2–3 will add LRUNodeStore and DiskNodeStore.
    #
    # HNSWNode is a class (reference type), so `get` returns a mutable
    # reference. Back-edge mutations during `add` are reflected in the store
    # directly without an explicit write-back — valid for in-memory stores.
    # Future disk-backed stores will need explicit update calls instead.
    module NodeStore
      abstract def get(id : Int32) : HNSWNode
      abstract def append(node : HNSWNode) : Nil
      abstract def size : Int32
    end

    # In-memory NodeStore: nodes stored in a flat Array indexed by entry ID.
    class MemoryNodeStore
      include NodeStore

      def initialize
        @nodes = [] of HNSWNode
      end

      def get(id : Int32) : HNSWNode
        @nodes[id]
      end

      def append(node : HNSWNode) : Nil
        @nodes << node
      end

      def size : Int32
        @nodes.size
      end
    end

    # A result returned from Index#search.
    record AnnResult, id : Int32, score : Float32

    class Index
      DEFAULT_M               =  16
      DEFAULT_EF_CONSTRUCTION = 200
      DEFAULT_EF_SEARCH       =  50

      @node_store : NodeStore
      @entry_point : Int32 # ID of the current graph entry point
      @max_layer : Int32   # highest layer currently in the graph
      @m : Int32
      @m_max0 : Int32 # layer-0 allows 2× neighbours (standard HNSW)
      @ef_construction : Int32
      @ml : Float64 # level generation factor = 1 / ln(M)
      @rng : Random

      def initialize(
        dims : Int32,
        m : Int32 = DEFAULT_M,
        ef_construction : Int32 = DEFAULT_EF_CONSTRUCTION,
        seed : Int32? = nil,
      )
        @dims = dims
        @m = m
        @m_max0 = m * 2
        @ef_construction = ef_construction
        @ml = 1.0 / Math.log(m.to_f64)
        @node_store = MemoryNodeStore.new
        @entry_point = -1
        @max_layer = -1
        @rng = seed ? Random.new(seed) : Random.new
      end

      # -------------------------------------------------------------------------
      # Public API
      # -------------------------------------------------------------------------

      # Insert a vector with integer *id*.  *id* must equal @node_store.size (i.e. add
      # sequentially starting from 0, matching VectorStore's entry indices).
      def add(id : Int32, vector : Embedding) : Nil
        raise ArgumentError.new("Wrong dims: #{vector.size} != #{@dims}") if vector.size != @dims
        raise ArgumentError.new("id must be sequential (expected #{@node_store.size}, got #{id})") if id != @node_store.size

        # Precalculate the reduced ef for higher layers
        reduced_ef_higher_layers = [@ef_construction // 4, @m].max

        node_layer = random_layer
        node = HNSWNode.new(vector, node_layer, @m)
        @node_store.append(node)

        if @entry_point == -1
          # First node — becomes the entry point at layer 0.
          @entry_point = id
          @max_layer = node_layer
          return
        end

        ep = @entry_point

        # Phase 1: descend from @max_layer down to node_layer + 1 (greedy, 1 candidate).
        (@max_layer).downto(node_layer + 1) do |layer|
          ep = greedy_descend(vector, ep, layer)
        end

        # Phase 2: from node_layer down to 0, do a beam search and wire neighbours.
        [node_layer, @max_layer].min.downto(0) do |layer|
          m_at_layer = layer == 0 ? @m_max0 : @m
          ef_at_layer = layer == 0 ? @ef_construction : reduced_ef_higher_layers

          candidates = search_layer(vector, ep, ef_at_layer, layer)
          neighbours = select_neighbours(vector, candidates, m_at_layer)

          node.neighbours[layer] = neighbours.map(&.id)
          ep = neighbours.first.id unless neighbours.empty?

          # Wire back-edges (mutual connections).
          neighbours.each do |neighbor|
            nb_node = @node_store.get(neighbor.id)
            unless nb_node.neighbours[layer].includes?(id)
              nb_node.neighbours[layer] << id
              # Prune if over limit.
              if nb_node.neighbours[layer].size > m_at_layer
                nb_node.neighbours[layer] = prune_neighbours(nb_node.vector, nb_node.neighbours[layer], m_at_layer)
              end
            end
          end
        end

        # Update entry point if the new node reaches a higher layer.
        if node_layer > @max_layer
          @max_layer = node_layer
          @entry_point = id
        end
      end

      # Restore an index from previously exported graph data without re-inserting vectors.
      # Yields each sequential node id; the caller supplies the vector and neighbour lists
      # for that node. Returns a fully wired Index.
      #
      # Example:
      #   index = HNSW::Index.restore(dims: 1024, m: 16, ef_construction: 200,
      #                               entry_point: ep, max_layer: ml,
      #                               node_count: n) do |id|
      #     { vectors[id], neighbours[id] }
      #   end
      def self.restore(dims : Int32, m : Int32, ef_construction : Int32,
                       entry_point : Int32, max_layer : Int32,
                       node_count : Int32,
                       & : Int32 -> {Embedding, Array(Array(Int32))}) : Index
        index = new(dims: dims, m: m, ef_construction: ef_construction)
        node_count.times do |id|
          vector, neighbours = yield id
          node = HNSWNode.new(vector, 0, m)
          node.neighbours = neighbours
          index.@node_store.append(node)
        end
        index.reset_with(entry_point, max_layer)
        index
      end

      # Export the graph topology by yielding (node_id, neighbours) for every node,
      # where neighbours is an Array per layer of neighbour ids.
      # Returns graph-level metadata the caller needs to persist alongside the nodes.
      #
      # Example:
      #   meta = index.export do |node_id, neighbours|
      #     # persist node_id and neighbours ...
      #   end
      #   # persist meta[:entry_point] and meta[:max_layer]
      def export(& : Int32, Array(Array(Int32)) ->) : NamedTuple(entry_point: Int32, max_layer: Int32)
        @node_store.size.times do |id|
          yield id, @node_store.get(id).neighbours
        end
        {entry_point: @entry_point, max_layer: @max_layer}
      end

      # Return the *k* approximate nearest neighbours for *query*.
      # *ef* controls recall vs speed — higher is slower but more accurate.
      def search(query : Embedding, k : Int32 = 5, ef : Int32 = DEFAULT_EF_SEARCH) : Array(AnnResult)
        raise ArgumentError.new("Index is empty") if @entry_point == -1
        raise ArgumentError.new("Wrong dims: #{query.size} != #{@dims}") if query.size != @dims

        ep = @entry_point

        # Descend to layer 1 greedily.
        @max_layer.downto(1) do |layer|
          ep = greedy_descend(query, ep, layer)
        end

        # Full beam search at layer 0.
        candidates = search_layer(query, ep, [ef, k].max, 0)

        candidates
          .first(k)
          .map { |candidate| AnnResult.new(candidate.id, 1.0_f32 - candidate.dist) } # convert dist back to similarity
      end

      def size : Int32
        @node_store.size
      end

      protected def reset_with(entry_point : Int32, max_layer : Int32) : Nil
        @entry_point = entry_point
        @max_layer = max_layer
      end

      # Draw a random layer for a new node.  Layer 0 is most common.
      private def random_layer : Int32
        layer = 0
        while @rng.rand < (1.0 / Math::E) && layer < 32
          layer += 1
        end
        layer
      end

      # Greedy single-step descent: follow the neighbour at *layer* that minimises
      # distance to *query*, repeat until no improvement.
      private def greedy_descend(query : Embedding, start : Int32, layer : Int32) : Int32
        best = start
        best_d = distance(query, @node_store.get(start).vector)

        loop do
          changed = false
          @node_store.get(best).neighbours[layer].each do |nb_id|
            d = distance(query, @node_store.get(nb_id).vector)
            if d < best_d
              best = nb_id
              best_d = d
              changed = true
            end
          end
          break unless changed
        end

        best
      end

      # Beam search at a single layer.  Returns up to *ef* candidates sorted by
      # distance ascending (nearest first).
      #
      # Uses two BinaryHeap instances:
      #   candidates  — min-heap: always expand the nearest unvisited node next
      #   dynamic_set — max-heap: evict the most distant when over capacity
      private def search_layer(
        query : Embedding,
        entry : Int32,
        ef : Int32,
        layer : Int32,
      ) : Array(Candidate)
        visited = Set(Int32).new
        visited << entry

        entry_dist = distance(query, @node_store.get(entry).vector)
        seed = Candidate.new(entry, entry_dist)

        candidates = BinaryHeap(Candidate).new { |this, that| this.dist <= that.dist }  # min-heap
        dynamic_set = BinaryHeap(Candidate).new { |this, that| this.dist >= that.dist } # max-heap

        candidates.push(seed)
        dynamic_set.push(seed)

        until candidates.empty?
          best = candidates.pop                      # O(log n) — nearest candidate
          break if best.dist > dynamic_set.peek.dist # O(1)     — worst in result set

          @node_store.get(best.id).neighbours[layer].each do |nb_id|
            next if visited.includes?(nb_id)
            visited << nb_id

            d = distance(query, @node_store.get(nb_id).vector)

            if d < dynamic_set.peek.dist || dynamic_set.size < ef
              c = Candidate.new(nb_id, d)
              candidates.push(c)                       # O(log n)
              dynamic_set.push(c)                      # O(log n)
              dynamic_set.pop if dynamic_set.size > ef # O(log n) — evict farthest
            end
          end
        end

        # Drain max-heap into result array, then reverse for ascending order.
        result = Array(Candidate).new(dynamic_set.size)
        until dynamic_set.empty?
          result << dynamic_set.pop
        end
        result.reverse!
        result
      end

      # Simple heuristic: take the *m* nearest candidates.
      private def select_neighbours(
        _query : Embedding,
        candidates : Array(Candidate),
        m : Int32,
      ) : Array(Candidate)
        candidates.first(m)
      end

      # Prune a neighbour list back to *m* by keeping the nearest *m*.
      private def prune_neighbours(
        base_vec : Embedding,
        neighbour_ids : Array(Int32),
        m : Int32,
      ) : Array(Int32)
        neighbour_ids
          .map { |nb_id| {nb_id, distance(base_vec, @node_store.get(nb_id).vector)} }
          .sort_by! { |_, dist| dist }
          .first(m)
          .map { |nb_id, _| nb_id }
      end

      # Angular distance: 1 - cosine_similarity.
      # Both vectors must be L2-normalised
      private def distance(a : Embedding, b : Embedding) : Float32
        1.0_f32 - VECM.dot(a, b)
      end
    end
  end
end
