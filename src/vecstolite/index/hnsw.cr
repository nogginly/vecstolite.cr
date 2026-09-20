require "./strategy"
require "../store/node_cache"
require "../indexer/binary_heap"

module Vecstolite
  module Index
    # Approximate nearest-neighbour search over a Hierarchical Navigable Small
    # World graph.
    #
    # Nodes live in a `NodeCache`, addressed by contiguous graph positions
    # (`ord`). Those positions never leave this class: `add` takes a stable
    # entry id, `search` returns them, and the mapping is the node cache's.
    #
    # Tuning:
    #
    # - *m* — neighbours per node per layer (layer 0 allows 2×). Higher means
    #   better recall, more memory per node and slower inserts.
    # - *ef_construction* — beam width while inserting. Higher means a better
    #   graph and slower ingest.
    # - *ef* at search time — beam width per query. Higher means better recall
    #   and slower queries.
    class HNSW < Strategy
      DEFAULT_M               =  16
      DEFAULT_EF_CONSTRUCTION = 200

      # One node in the search beam, ordered by distance (lower is nearer).
      record Candidate, ord : Int32, dist : Float32

      # The stored node type, kept fully qualified: `HNSW` inside this class
      # refers to the class itself, not to the graph module.
      alias Node = ::Vecstolite::HNSW::HNSWNode

      getter entry_point : Int32
      getter max_layer : Int32

      @cache : NodeCache::Strategy
      @dims : Int32
      @m : Int32
      @m_max0 : Int32
      @ef_construction : Int32
      @ml : Float64
      @rng : Random

      def initialize(@cache : NodeCache::Strategy,
                     dims : Int32,
                     m : Int32 = DEFAULT_M,
                     ef_construction : Int32 = DEFAULT_EF_CONSTRUCTION,
                     seed : Int32? = nil,
                     entry_point : Int32 = -1,
                     max_layer : Int32 = -1)
        @dims = dims
        @m = m
        @m_max0 = m * 2
        @ef_construction = ef_construction
        @ml = 1.0 / Math.log(m.to_f64)
        @entry_point = entry_point
        @max_layer = max_layer
        @rng = seed ? Random.new(seed) : Random.new
      end

      # Restores graph-level state read from metadata, without touching nodes.
      def reset_with(entry_point : Int32, max_layer : Int32) : Nil
        @entry_point = entry_point
        @max_layer = max_layer
      end

      def add(entry_id : Int64, vector : Embedding) : Nil
        raise ArgumentError.new("Wrong dims: #{vector.size} != #{@dims}") if vector.size != @dims

        reduced_ef_higher_layers = [@ef_construction // 4, @m].max
        node_layer = random_layer
        node = Node.new(vector, node_layer, @m)
        ord = @cache.append(node, entry_id)

        if @entry_point == -1
          @entry_point = ord
          @max_layer = node_layer
          return
        end

        ep = @entry_point

        # Descend greedily through the layers above the new node.
        @max_layer.downto(node_layer + 1) do |layer|
          ep = greedy_descend(vector, ep, layer)
        end

        # Beam search and wire neighbours from the new node's top layer down.
        [node_layer, @max_layer].min.downto(0) do |layer|
          m_at_layer = layer == 0 ? @m_max0 : @m
          ef_at_layer = layer == 0 ? @ef_construction : reduced_ef_higher_layers

          candidates = search_layer(vector, ep, ef_at_layer, layer)
          neighbours = select_neighbours(candidates, m_at_layer)

          node.neighbours[layer] = neighbours.map(&.ord)
          ep = neighbours.first.ord unless neighbours.empty?

          neighbours.each do |neighbour|
            nb_node = @cache.get(neighbour.ord)
            next if nb_node.neighbours[layer].includes?(ord)

            nb_node.neighbours[layer] << ord
            if nb_node.neighbours[layer].size > m_at_layer
              nb_node.neighbours[layer] =
                prune_neighbours(nb_node.vector, nb_node.neighbours[layer], m_at_layer)
            end
            @cache.write_back(neighbour.ord, nb_node)
          end
        end

        @cache.write_back(ord, node)

        if node_layer > @max_layer
          @max_layer = node_layer
          @entry_point = ord
        end
      end

      def search(vector : Embedding,
                 k : Int32,
                 ef : Int32 = DEFAULT_EF_SEARCH,
                 allowed : Set(Int64)? = nil) : Array(Hit)
        return [] of Hit if k <= 0 || @entry_point == -1
        raise ArgumentError.new("Wrong dims: #{vector.size} != #{@dims}") if vector.size != @dims

        ep = @entry_point
        @max_layer.downto(1) do |layer|
          ep = greedy_descend(vector, ep, layer)
        end

        hits = [] of Hit
        search_layer(vector, ep, [ef, k].max, 0).each do |candidate|
          entry_id = @cache.entry_id(candidate.ord)
          next if allowed && !allowed.includes?(entry_id)

          hits << Hit.new(entry_id, 1.0_f32 - candidate.dist)
          break if hits.size == k
        end
        hits
      end

      def size : Int32
        @cache.size
      end

      def kind : Symbol
        :hnsw
      end

      def fully_persisted? : Bool
        @cache.fully_persisted?
      end

      def flush : Nil
        @cache.flush
      end

      def clear : Nil
        @cache.clear
        @entry_point = -1
        @max_layer = -1
      end

      # -------------------------------------------------------------------------

      # Draws a layer for a new node. Layer 0 is the common case; each further
      # layer is exponentially less likely.
      private def random_layer : Int32
        layer = 0
        while @rng.rand < (1.0 / Math::E) && layer < 32
          layer += 1
        end
        layer
      end

      # Follows the neighbour at *layer* that most reduces distance to *query*,
      # repeating until no neighbour improves on the current node.
      private def greedy_descend(query : Embedding, start : Int32, layer : Int32) : Int32
        best = start
        best_dist = distance(query, @cache.get(start).vector)

        loop do
          changed = false
          @cache.get(best).neighbours[layer].each do |nb_ord|
            dist = distance(query, @cache.get(nb_ord).vector)
            next unless dist < best_dist

            best = nb_ord
            best_dist = dist
            changed = true
          end
          break unless changed
        end

        best
      end

      # Beam search at one layer, returning up to *ef* candidates nearest first.
      #
      # Two heaps: a min-heap of candidates still to expand, and a max-heap of
      # the best found so far, whose most distant member is evicted once the
      # beam is full.
      private def search_layer(query : Embedding,
                               entry : Int32,
                               ef : Int32,
                               layer : Int32) : Array(Candidate)
        visited = Set(Int32).new
        visited << entry

        seed = Candidate.new(entry, distance(query, @cache.get(entry).vector))
        candidates = BinaryHeap(Candidate).new { |this, that| this.dist <= that.dist }
        beam = BinaryHeap(Candidate).new { |this, that| this.dist >= that.dist }
        candidates.push(seed)
        beam.push(seed)

        until candidates.empty?
          best = candidates.pop
          break if best.dist > beam.peek.dist

          @cache.get(best.ord).neighbours[layer].each do |nb_ord|
            next if visited.includes?(nb_ord)
            visited << nb_ord

            dist = distance(query, @cache.get(nb_ord).vector)
            next unless dist < beam.peek.dist || beam.size < ef

            candidate = Candidate.new(nb_ord, dist)
            candidates.push(candidate)
            beam.push(candidate)
            beam.pop if beam.size > ef
          end
        end

        result = Array(Candidate).new(beam.size)
        while !beam.empty?
          result << beam.pop
        end
        result.reverse!
      end

      # Chooses up to *m* neighbours from *candidates*, which must arrive
      # sorted nearest first.
      #
      # Taking simply the *m* nearest leaves a node in a dense cluster with
      # every edge pointing inward, and a search that arrives there has no
      # road out. This keeps a candidate only when it is closer to the node
      # being linked than to any neighbour already chosen, so the selected
      # edges spread out in different directions rather than piling into one
      # neighbourhood.
      #
      # Candidates rejected that way are held back and used to fill any
      # remaining slots, so a node never ends up with fewer edges than nearest
      # *m* would have given it.
      private def select_neighbours(candidates : Array(Candidate),
                                    m : Int32) : Array(Candidate)
        return candidates if candidates.size <= m

        selected = Array(Candidate).new(m)
        selected_vectors = Array(Embedding).new(m)
        discarded = Array(Candidate).new

        candidates.each do |candidate|
          break if selected.size >= m

          candidate_vector = @cache.get(candidate.ord).vector
          closer_to_node = selected_vectors.all? do |chosen|
            candidate.dist < distance(candidate_vector, chosen)
          end

          if closer_to_node
            selected << candidate
            selected_vectors << candidate_vector
          else
            discarded << candidate
          end
        end

        discarded.each do |candidate|
          break if selected.size >= m
          selected << candidate
        end

        selected
      end

      # Prunes a neighbour list back to *m*, applying the same diversity rule
      # as `select_neighbours` so pruning cannot undo what selection achieved.
      private def prune_neighbours(base_vec : Embedding,
                                   neighbour_ords : Array(Int32),
                                   m : Int32) : Array(Int32)
        candidates = neighbour_ords
          .map { |nb_ord| Candidate.new(nb_ord, distance(base_vec, @cache.get(nb_ord).vector)) }
          .sort_by!(&.dist)

        select_neighbours(candidates, m).map(&.ord)
      end
    end
  end
end
