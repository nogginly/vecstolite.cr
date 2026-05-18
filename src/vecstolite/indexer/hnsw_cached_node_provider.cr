require "./hnsw_indexer"

module Vecstolite
  module HNSW
    # Node provider that uses a cache with a pluggable loader for misses.
    # The loader is called when a node is not in cache, allowing the caller
    # to supply the source (DB, network, etc).
    class CachedNodeProvider < NodeProvider
      # Loader proc signature: given a node ID, return the HNSWNode
      alias Loader = Int32 -> HNSWNode

      @cache : NodeCache
      @loader : Loader

      def initialize(@cache : NodeCache, &loader : Loader)
        @loader = loader
      end

      # Get a node from cache, or load it if not cached.
      def get(id : Int32) : HNSWNode
        # Check cache first (hit: O(1), updates LRU)
        if cached = @cache.get?(id)
          return cached
        end

        # Cache miss: load from source
        node = @loader.call(id)

        # Store in cache (may trigger eviction)
        @cache.put(id, node)

        node
      end
    end
  end
end
