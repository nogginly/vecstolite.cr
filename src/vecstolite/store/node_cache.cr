require "./repository"
require "../indexer/hnsw_node_store"

module Vecstolite
  # Holds graph nodes for an index strategy, backed by `Repository`.
  #
  # Three strategies trade memory against durability:
  #
  # - `Memory` keeps every node in RAM and writes the graph only on `flush`.
  #   Fastest, but an unclean exit costs a rebuild from vectors.
  # - `LRU` keeps a bounded set in RAM and writes through on every mutation.
  # - `Disk` caches nothing and writes through on every mutation.
  #
  # Nodes are addressed by `ord`, the contiguous graph position. Each one also
  # carries the stable `entry_id` it stands for, which is what callers resolve
  # search results against.
  module NodeCache
    abstract class Strategy
      abstract def get(ord : Int32) : HNSW::HNSWNode

      # Adds *node* as the next graph position, returning its `ord`.
      abstract def append(node : HNSW::HNSWNode, entry_id : Int64) : Int32

      # Persists mutated neighbour lists for *ord*.
      abstract def write_back(ord : Int32, node : HNSW::HNSWNode) : Nil

      abstract def entry_id(ord : Int32) : Int64
      abstract def size : Int32
      abstract def each_node(& : Int32, Int64, HNSW::HNSWNode ->) : Nil

      # True when every mutation is already on disk, so only graph metadata
      # needs writing at close.
      abstract def fully_persisted? : Bool

      # Writes anything held only in memory. A no-op for write-through
      # strategies.
      abstract def flush : Nil

      # Discards every node, in memory and in the repository, so the graph can
      # be rebuilt from the stored vectors.
      abstract def clear : Nil

      abstract def mode : Symbol

      def stats : NamedTuple(mode: Symbol, cached: Int32, bytes: Int64,
        hits: Int64, misses: Int64, evictions: Int64)
        {mode: mode, cached: 0, bytes: 0_i64, hits: 0_i64, misses: 0_i64, evictions: 0_i64}
      end

      # Bytes a node occupies in RAM: vector, packed neighbours, and object
      # overhead.
      protected def node_bytes(node : HNSW::HNSWNode) : Int64
        (node.vector.bytesize + node.neighbours.sum(&.size) * 4 + 128).to_i64
      end
    end

    # Keeps every node in RAM. `flush` writes the whole graph in one pass, so
    # a crash before it leaves `graph_saved = 0` and forces a rebuild.
    class Memory < Strategy
      @nodes : Array(HNSW::HNSWNode)
      @entry_ids : Array(Int64)

      def initialize(@repo : Repository)
        @nodes = [] of HNSW::HNSWNode
        @entry_ids = [] of Int64
      end

      # Loads every persisted node into memory, in `ord` order.
      def load : Nil
        @nodes.clear
        @entry_ids.clear
        @repo.each_node do |row|
          node = HNSW::HNSWNode.new(row.vector, 0, 0)
          node.neighbours = row.neighbours
          @nodes << node
          @entry_ids << row.entry_id
        end
      end

      def get(ord : Int32) : HNSW::HNSWNode
        @nodes[ord]
      end

      def append(node : HNSW::HNSWNode, entry_id : Int64) : Int32
        @nodes << node
        @entry_ids << entry_id
        @nodes.size - 1
      end

      def write_back(ord : Int32, node : HNSW::HNSWNode) : Nil
        @nodes[ord] = node
      end

      def entry_id(ord : Int32) : Int64
        @entry_ids[ord]
      end

      def size : Int32
        @nodes.size
      end

      def each_node(& : Int32, Int64, HNSW::HNSWNode ->) : Nil
        @nodes.each_with_index do |node, ord|
          yield ord, @entry_ids[ord], node
        end
      end

      def fully_persisted? : Bool
        false
      end

      # Rewrites the whole graph. Callers run this inside the transaction that
      # also records the graph metadata.
      def flush : Nil
        @repo.clear_nodes
        each_node do |ord, entry_id, node|
          @repo.insert_node(ord, entry_id, node.neighbours)
        end
      end

      def clear : Nil
        @nodes.clear
        @entry_ids.clear
        @repo.clear_nodes
      end

      def mode : Symbol
        :memory
      end

      def stats : NamedTuple(mode: Symbol, cached: Int32, bytes: Int64,
        hits: Int64, misses: Int64, evictions: Int64)
        {
          mode:      mode,
          cached:    @nodes.size,
          bytes:     @nodes.sum(0_i64) { |node| node_bytes(node) },
          hits:      0_i64,
          misses:    0_i64,
          evictions: 0_i64,
        }
      end
    end

    # Reads every node from the repository and caches nothing. Constant memory,
    # one query per node visit.
    class Disk < Strategy
      @total : Int32
      @misses : Int64

      def initialize(@repo : Repository)
        @total = @repo.node_count
        @misses = 0_i64
      end

      def get(ord : Int32) : HNSW::HNSWNode
        @misses += 1
        row = @repo.node(ord) || raise Repository::Error.new("Node #{ord} not found.")
        node = HNSW::HNSWNode.new(row.vector, 0, 0)
        node.neighbours = row.neighbours
        node
      end

      def append(node : HNSW::HNSWNode, entry_id : Int64) : Int32
        ord = @total
        @total += 1
        @repo.insert_node(ord, entry_id, node.neighbours)
        ord
      end

      def write_back(ord : Int32, node : HNSW::HNSWNode) : Nil
        @repo.update_neighbours(ord, node.neighbours)
      end

      def entry_id(ord : Int32) : Int64
        @repo.entry_id_for(ord) || raise Repository::Error.new("Node #{ord} not found.")
      end

      def size : Int32
        @total
      end

      def each_node(& : Int32, Int64, HNSW::HNSWNode ->) : Nil
        @repo.each_node do |row|
          node = HNSW::HNSWNode.new(row.vector, 0, 0)
          node.neighbours = row.neighbours
          yield row.ord, row.entry_id, node
        end
      end

      def fully_persisted? : Bool
        true
      end

      def flush : Nil
      end

      def clear : Nil
        @repo.clear_nodes
        @total = 0
      end

      def mode : Symbol
        :disk
      end

      def stats : NamedTuple(mode: Symbol, cached: Int32, bytes: Int64,
        hits: Int64, misses: Int64, evictions: Int64)
        {mode: mode, cached: 0, bytes: 0_i64, hits: 0_i64, misses: @misses, evictions: 0_i64}
      end
    end

    # Write-through cache with a byte budget, evicting least-recently-used
    # nodes. The database is always current; the cache only saves reads.
    class LRU < Strategy
      private class Slot
        property ord : Int32
        property node : HNSW::HNSWNode
        property bytes : Int64
        property prev : Slot?
        property succ : Slot?

        def initialize(@ord, @node, @bytes)
        end
      end

      @cache = {} of Int32 => Slot
      @head : Slot? = nil
      @tail : Slot? = nil
      @current_bytes = 0_i64
      @hits = 0_i64
      @misses = 0_i64
      @evictions = 0_i64
      @total : Int32

      def initialize(@repo : Repository, @max_bytes : Int64)
        @total = @repo.node_count
      end

      def get(ord : Int32) : HNSW::HNSWNode
        if slot = @cache[ord]?
          promote(slot)
          @hits += 1
          return slot.node
        end

        @misses += 1
        row = @repo.node(ord) || raise Repository::Error.new("Node #{ord} not found.")
        node = HNSW::HNSWNode.new(row.vector, 0, 0)
        node.neighbours = row.neighbours
        admit(ord, node)
        node
      end

      def append(node : HNSW::HNSWNode, entry_id : Int64) : Int32
        ord = @total
        @total += 1
        @repo.insert_node(ord, entry_id, node.neighbours)
        admit(ord, node)
        ord
      end

      # Persists *node*, then makes it the cached object for *ord*. A caller
      # can hold a node across an eviction, during which a second object for
      # the same ord may be read back and cached; leaving that copy in place
      # would let a later write_back persist its stale neighbours over this
      # one, silently stripping edges from the graph.
      def write_back(ord : Int32, node : HNSW::HNSWNode) : Nil
        @repo.update_neighbours(ord, node.neighbours)
        if slot = @cache[ord]?
          bytes = node_bytes(node)
          @current_bytes += bytes - slot.bytes
          slot.bytes = bytes
          slot.node = node unless slot.node.same?(node)
          promote(slot)
        else
          admit(ord, node)
        end
      end

      def entry_id(ord : Int32) : Int64
        @repo.entry_id_for(ord) || raise Repository::Error.new("Node #{ord} not found.")
      end

      def size : Int32
        @total
      end

      def each_node(& : Int32, Int64, HNSW::HNSWNode ->) : Nil
        @repo.each_node do |row|
          node = HNSW::HNSWNode.new(row.vector, 0, 0)
          node.neighbours = row.neighbours
          yield row.ord, row.entry_id, node
        end
      end

      def fully_persisted? : Bool
        true
      end

      def flush : Nil
      end

      def clear : Nil
        @repo.clear_nodes
        @cache.clear
        @head = nil
        @tail = nil
        @current_bytes = 0_i64
        @total = 0
      end

      def mode : Symbol
        :lru
      end

      def stats : NamedTuple(mode: Symbol, cached: Int32, bytes: Int64,
        hits: Int64, misses: Int64, evictions: Int64)
        {
          mode:      mode,
          cached:    @cache.size,
          bytes:     @current_bytes,
          hits:      @hits,
          misses:    @misses,
          evictions: @evictions,
        }
      end

      # -------------------------------------------------------------------------
      private def admit(ord : Int32, node : HNSW::HNSWNode) : Nil
        bytes = node_bytes(node)
        while @current_bytes + bytes > @max_bytes && @tail
          evict_tail
        end
        slot = Slot.new(ord, node, bytes)
        @cache[ord] = slot
        @current_bytes += bytes
        push_head(slot)
      end

      private def evict_tail : Nil
        tail = @tail
        return unless tail
        unlink(tail)
        @cache.delete(tail.ord)
        @current_bytes -= tail.bytes
        @evictions += 1
      end

      private def promote(slot : Slot) : Nil
        return if slot.same?(@head)
        unlink(slot)
        push_head(slot)
      end

      private def push_head(slot : Slot) : Nil
        slot.succ = @head
        slot.prev = nil
        if head = @head
          head.prev = slot
        end
        @head = slot
        @tail ||= slot
      end

      private def unlink(slot : Slot) : Nil
        if prev = slot.prev
          prev.succ = slot.succ
        else
          @head = slot.succ
        end
        if succ = slot.succ
          succ.prev = slot.prev
        else
          @tail = slot.prev
        end
        slot.prev = nil
        slot.succ = nil
      end
    end
  end
end
