require "../vector_embedder"

module Vecstolite
  module HNSW
    # One node in the HNSW graph: a vector and per-layer neighbour lists.
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
    # node data, replacing the former split between `@entry_embeddings` and
    # the index's own node array.
    #
    # Available implementations:
    #   - `MemoryNodeStore`  — full index in memory (default; fastest)
    #   - `LRUNodeStore`     — fixed memory budget with LRU eviction (sqlite3_node_store.cr)
    #   - `DiskNodeStore`    — no cache; every access reads from SQLite (sqlite3_node_store.cr)
    #
    # Note on mutability: `HNSWNode` is a reference type, so `get` returns a
    # mutable reference. For `MemoryNodeStore` and `LRUNodeStore`, back-edge
    # mutations during `Index#add` are reflected in the store directly.
    # `DiskNodeStore` requires an explicit `write_back` call after mutation.
    module NodeStore
      # Retrieve the node for *id*. Must raise if *id* is out of range.
      abstract def get(id : Int32) : HNSWNode

      # Append a newly created node. Called once per `Index#add`, in ID order.
      abstract def append(node : HNSWNode) : Nil

      # Persist any mutation to *node* (e.g. back-edge wiring after an `add`).
      # No-op for memory-backed stores; DB write for disk-backed stores.
      abstract def write_back(id : Int32, node : HNSWNode) : Nil

      # Number of nodes stored.
      abstract def size : Int32

      # Restore from previously persisted state during `open`.
      # Called by `HNSW::Index.restore` instead of repeated `append` calls.
      # For MemoryNodeStore: loads all nodes by calling the block for each id.
      # For disk-backed stores: sets the size counter; data already in DB.
      abstract def restore_from_persisted(count : Int32, & : Int32 -> {Embedding, Array(Array(Int32))}) : Nil

      # Iterate all nodes. Used by `save_graph` for MemoryNodeStore.
      abstract def each_node(& : Int32, HNSWNode ->) : Nil

      # True if nodes are written through to disk on every mutation (LRU/Disk).
      # False for MemoryNodeStore — `save_graph` must flush them explicitly.
      abstract def fully_persisted? : Bool

      # Observability — used by the store's `stats` method.
      abstract def cache_mode : Symbol
      abstract def cache_size : Int32
      abstract def cache_hits : Int64
      abstract def cache_misses : Int64
      abstract def cache_evictions : Int64
    end

    # In-memory NodeStore: nodes held in a flat Array indexed by entry ID.
    # Default for all store types when no `cache_max_bytes` is specified
    # via the legacy API. `SQLitePayloadVectorStore` defaults to `LRUNodeStore`.
    class MemoryNodeStore
      include NodeStore

      protected def initialize
        @nodes = [] of HNSWNode
      end

      def get(id : Int32) : HNSWNode
        @nodes[id]
      end

      def append(node : HNSWNode) : Nil
        @nodes << node
      end

      # No-op: HNSWNode is a reference type; mutations are reflected in-place.
      def write_back(id : Int32, node : HNSWNode) : Nil
      end

      def size : Int32
        @nodes.size
      end

      def restore_from_persisted(count : Int32, & : Int32 -> {Embedding, Array(Array(Int32))}) : Nil
        count.times do |id|
          vector, neighbours = yield id
          node = HNSWNode.new(vector, 0, 0) # neighbours overridden below
          node.neighbours = neighbours
          @nodes << node
        end
      end

      def each_node(& : Int32, HNSWNode ->) : Nil
        @nodes.each_with_index { |node, id| yield id, node }
      end

      def fully_persisted? : Bool
        false
      end

      def cache_mode : Symbol
        :memory
      end

      def cache_size : Int32
        @nodes.size
      end

      def cache_hits : Int64
        0_i64
      end

      def cache_misses : Int64
        0_i64
      end

      def cache_evictions : Int64
        0_i64
      end
    end
  end
end
