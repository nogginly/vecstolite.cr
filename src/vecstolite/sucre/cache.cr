# A generic key/value cache
# :nodoc:
class Cache(K, V)
  struct Cached(V)
    getter entry : V
    getter last_read : Time::Instant

    def initialize(@entry)
      @last_read = Time.instant
    end

    # Update `last_read` time
    def touch!
      @last_read = Time.instant
    end
  end

  @cache : Hash(K, Cached(V))

  def initialize
    @cache = {} of K => Cached(V)
  end

  def put(id : K, entry : V)
    @cache[id] = Cached(V).new(entry)
    entry
  end

  def get?(id : K) : V?
    if hit = @cache[id]?
      hit.touch!
      hit.entry
    end
  end

  # Returns cached valye, or yields `id` to retrieve Entry to put in cache
  def get(id : K, &) : V
    get?(id) || put(id, yield id)
  end

  def size
    @cache.size
  end
end
