require "../vector_embedder"

module Vecstolite
  module HNSW
    # LRU cache for HNSW nodes with fixed memory budget.
    # Supports pinning nodes (e.g., entry point + top layers) that are never evicted.
    class NodeCache
      # A cached node with access time for LRU ordering.
      private record CacheEntry, node : HNSWNode, size : Int32, accessed_at : Time

      @cache : Hash(Int32, CacheEntry)
      @max_bytes : Int32
      @bytes_used : Int32
      @pinned : Set(Int32)

      def initialize(@max_bytes : Int32, pinned_ids : Array(Int32) = [] of Int32)
        @cache = {} of Int32 => CacheEntry
        @bytes_used = 0
        @pinned = Set.new(pinned_ids)
      end

      # Get a cached node, updating its access time (LRU).
      # Returns nil if not cached.
      def get?(id : Int32) : HNSWNode?
        if entry = @cache[id]?
          @cache[id] = CacheEntry.new(entry.node, entry.size, Time.utc)
          return entry.node
        end
        nil
      end

      # Put a node in the cache. Evicts LRU nodes if over budget.
      def put(id : Int32, node : HNSWNode) : Nil
        size = calculate_node_size(node)

        # If this node already exists, subtract its old size first
        if existing = @cache[id]?
          @bytes_used -= existing.size
        end

        # Evict LRU nodes until there's space
        while @bytes_used + size > @max_bytes && @cache.size > 0
          evict_one
        end

        @cache[id] = CacheEntry.new(node, size, Time.utc)
        @bytes_used += size
      end

      # Invalidate a cached node (remove it, will reload on next access).
      def invalidate(id : Int32) : Nil
        if entry = @cache.delete(id)
          @bytes_used -= entry.size
        end
      end

      # Clear all cached nodes except pinned ones.
      def clear_unpinned : Nil
        to_delete = @cache.keys.reject { |id| @pinned.includes?(id) }
        to_delete.each do |id|
          invalidate(id)
        end
      end

      # Pin a node so it's never evicted.
      def pin(id : Int32) : Nil
        @pinned << id
      end

      # Unpin a node so it can be evicted.
      def unpin(id : Int32) : Nil
        @pinned.delete(id)
      end

      # Total bytes currently in cache.
      def bytes_used : Int32
        @bytes_used
      end

      # Number of cached nodes.
      def size : Int32
        @cache.size
      end

      # Check if a node is cached.
      def cached?(id : Int32) : Bool
        @cache.has_key?(id)
      end

      # Private: Calculate memory footprint of a node
      private def calculate_node_size(node : HNSWNode) : Int32
        # Vector: Float32 = 4 bytes per element
        vector_bytes = node.vector.size * 4

        # Neighbors: Array(Array(Int32)) — each Int32 = 4 bytes
        neighbor_bytes = node.neighbours.sum { |layer| layer.size * 4 }

        # Overhead for the HNSWNode struct and internal arrays
        # (vector object, array objects, allocator overhead)
        overhead = 220

        vector_bytes + neighbor_bytes + overhead
      end

      # Private: Evict the least-recently-used non-pinned node
      private def evict_one : Nil
        # Find the oldest non-pinned node
        oldest_id = @cache.keys
          .reject { |id| @pinned.includes?(id) }
          .min_by? { |id| @cache[id].accessed_at }

        return unless oldest_id

        if entry = @cache.delete(oldest_id)
          @bytes_used -= entry.size
        end
      end
    end
  end
end
