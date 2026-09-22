require "./node_cache"
require "./repository"

module Vecstolite
  # Chooses how graph nodes are held in memory.
  #
  # ```
  # CacheMode.lru(256 * Vecstolite::MB) # bounded, written through
  # CacheMode.memory                    # all in RAM, written at close
  # CacheMode.disk                      # nothing cached
  # ```
  #
  # `memory` is the fastest and the least durable: nodes reach disk only when
  # the store is closed, so an unclean exit costs a rebuild from the stored
  # vectors. Prefer it for batch work that ends with a clean close, and `lru`
  # for long-lived or frequently restarted stores.
  struct CacheMode
    DEFAULT_MAX_BYTES = 256_i64 * MB

    getter kind : Symbol
    getter max_bytes : Int64

    protected def initialize(@kind : Symbol, @max_bytes : Int64)
    end

    def self.memory : CacheMode
      new(:memory, 0_i64)
    end

    def self.disk : CacheMode
      new(:disk, 0_i64)
    end

    def self.lru(max_bytes : Int64 = DEFAULT_MAX_BYTES) : CacheMode
      raise ArgumentError.new("Cache budget must be positive.") unless max_bytes > 0
      new(:lru, max_bytes)
    end

    def build(repo : Repository) : NodeCache::Strategy
      case @kind
      when :memory then NodeCache::Memory.new(repo)
      when :lru    then NodeCache::LRU.new(repo, @max_bytes)
      else              NodeCache::Disk.new(repo)
      end
    end
  end
end
