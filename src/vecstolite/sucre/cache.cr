# A generic key/value cache
#
# This cache stores values along with the timestamp of their last access.
# It is useful for scenarios where data freshness or hit/miss tracking is required.
#
# @example
#   cache = Cache(String, Int32).new
#   cache.put("one", 1)
#   val = cache.get("one")   # => 1
#
# Internally, each entry is wrapped in a `Cached` struct which holds the
# value and the last read time. Accessing a key via `get` or `get?` updates
# the access timestamp.
#
# The cache supports optional TTL (time-to-live) expiration.
# If a TTL is set, entries older than the span are evicted on access.
# It also records the last access time for hit/miss tracking.
# :nodoc:
class Cache(K, V)
  # Encapsulates a cached value and its last read timestamp.
  struct Cached(V)
    # The stored entry.
    getter entry : V
    # The instant when the entry was last accessed.
    getter last_read : Time::Instant

    # Creates a new cached entry and records the current time.
    def initialize(@entry)
      @last_read = Time.instant
    end

    # Updates the `last_read` timestamp to now.
    def touch!
      @last_read = Time.instant
    end
  end

  @cache : Hash(K, Cached(V))
  @ttl : Time::Span?

  # Initializes an empty cache.
  # @param ttl [Time::Span?] optional TTL; if set, entries expire after the time span
  # @return [Cache(K,V)] a new cache instance
  def initialize(@ttl = nil)
    @cache = {} of K => Cached(V)
  end

  # Stores the given entry under `id`, replacing any existing value.
  # Returns the stored entry.
  #
  # @param id [K] the key used to identify the entry
  # @param entry [V] the value to cache
  # @return [V] the cached value
  def put(id : K, entry : V)
    @cache[id] = Cached(V).new(entry)
    entry
  end

  # Retrieves the entry for `id` if present, otherwise returns `nil`.
  # If TTL is configured, an entry older than the time span is evicted and `nil` is returned.
  # Accessing a valid entry updates its last read timestamp.
  #
  # @param id [K] the key for the requested entry
  # @return [V?] the cached value or `nil` if not found or expired
  def get?(id : K) : V?
    if hit = @cache[id]?
      expired = (ttl = @ttl) && (Time.instant - hit.last_read > ttl)
      if expired
        @cache.delete(id)
        nil
      else
        hit.touch!
        hit.entry
      end
    end
  end

  # Retrieves the entry for `id` if present; otherwise yields to the
  # block to compute the value, caches it, and returns the result.
  #
  # @param id [K] the key
  # @yield [id] compute the value if missing
  # @return [V] the cached or freshly computed value
  def get(id : K, &) : V
    get?(id) || put(id, yield id)
  end

  # Number of entries stored in the cache.
  #
  # @return [Int] the cache size
  def size
    @cache.size
  end

  # Force expired entries to be removed from the cache
  def purge_expired
    return unless ttl = @ttl
    now = Time.instant
    @cache.each { |k, v| @cache.delete(k) if now - v.last_read > ttl }
  end
end
