require "./repository"
require "./cache_mode"
require "../index/config"

module Vecstolite
  # A vector store backed by SQLite, on disk or in memory.
  #
  # Entries are text plus an embedding, with optional metadata (`M`) and an
  # optional shared payload (`P`). Both types must round-trip through JSON —
  # any `JSON::Serializable` type, `Hash` or primitive will do.
  #
  # ```
  # store = Vecstolite::Store(Meta, Payload).open("notes.db", embedder)
  # id = store.add("The sky is blue.", meta: Meta.new("en"))
  # store.search("colour of the sky", k: 3)
  # store.close
  # ```
  #
  # The block form closes the store — flushing the graph — even if the block
  # raises:
  #
  # ```
  # Vecstolite::Store(Meta, Payload).open("notes.db", embedder) do |store|
  #   store.add("The sky is blue.")
  # end
  # ```
  #
  # Ids returned by `add` are stable: they survive compaction and are never
  # reused. A store is single-threaded; serialise concurrent access yourself.
  class Store(M, P)
    class Error < Exception; end

    DEFAULT_K         =  5
    DEFAULT_EF_SEARCH = 50

    # How many times a search widens its beam to replace tombstoned hits
    # before returning a short result rather than scanning the whole graph.
    MAX_OVERSAMPLE_ROUNDS = 3

    # An entry as stored. `text` and `meta` are absent on a deleted entry
    # whose space has been released but whose graph node still routes.
    record Entry(M, P),
      id : Int64,
      key : String?,
      text : String?,
      meta : M?,
      payload_id : Int64?,
      payload : P?

    # One search result, scored by cosine similarity where 1.0 is identical.
    record SearchResult(M, P),
      id : Int64,
      key : String?,
      text : String,
      score : Float32,
      meta : M?,
      payload_id : Int64?,
      payload : P?

    getter path : String
    getter embedder : VectorEmbedder
    getter? closed : Bool

    @repo : Repository
    @cache : NodeCache::Strategy
    @index : Index::Strategy

    # Opens the store at *path*, or `":memory:"` for a private, unshared
    # database that vanishes with the process.
    #
    # Raises `Error` if the database was written by a different embedder or
    # with a different vector dimension, unless *verify_embedder* is false.
    def self.open(path : String,
                  embedder : VectorEmbedder,
                  index : Index::Config = Index.hnsw,
                  cache : CacheMode = CacheMode.lru,
                  readonly : Bool = false,
                  create_if_missing : Bool = true,
                  verify_embedder : Bool = true,
                  page_cache_bytes : Int64 = Repository::DEFAULT_PAGE_CACHE_BYTES) : self
      new(path, embedder, index, cache, readonly, create_if_missing, verify_embedder, page_cache_bytes)
    end

    # Opens the store, yields it, and closes it even if the block raises.
    def self.open(path : String,
                  embedder : VectorEmbedder,
                  index : Index::Config = Index.hnsw,
                  cache : CacheMode = CacheMode.lru,
                  readonly : Bool = false,
                  create_if_missing : Bool = true,
                  verify_embedder : Bool = true,
                  page_cache_bytes : Int64 = Repository::DEFAULT_PAGE_CACHE_BYTES, &)
      store = open(path, embedder, index, cache, readonly, create_if_missing, verify_embedder, page_cache_bytes)
      begin
        yield store
      ensure
        store.close
      end
    end

    protected def initialize(@path : String,
                             @embedder : VectorEmbedder,
                             index_config : Index::Config,
                             cache_mode : CacheMode,
                             @readonly : Bool,
                             create_if_missing : Bool,
                             verify_embedder : Bool,
                             page_cache_bytes : Int64)
      @closed = false
      @repo = Repository.open(@path,
        dimensions: @embedder.dimensions,
        readonly: @readonly,
        create_if_missing: create_if_missing,
        page_cache_bytes: page_cache_bytes)

      # Assigned before the guarded section below: Crystal does not treat
      # assignments inside a begin/rescue body as definite.
      @cache = cache_mode.build(@repo)
      @index = index_config.build(@repo, @cache)

      begin
        check_embedder if verify_embedder
        restore_index(index_config)
      rescue ex
        @repo.close
        @closed = true
        raise ex
      end
    end

    # Flushes anything held in memory and closes the database.
    def close : Nil
      return if @closed
      unless @readonly
        @repo.transaction do
          @index.flush
          @repo.set_graph_meta(@index.entry_point, @index.max_layer, graph_saved: true)
        end
      end
      @repo.close
      @closed = true
    end

    # -------------------------------------------------------------------------
    # Entries
    # -------------------------------------------------------------------------

    # Adds *text*, returning its stable id. Supply *vector* to skip embedding,
    # and *key* to give the entry an identifier of your own.
    def add(text : String,
            meta : M? = nil,
            payload_id : Int64? = nil,
            key : String? = nil,
            vector : Embedding? = nil) : Int64
      writable!
      embedding = vector || @embedder.embed(text)

      new_id = 0_i64
      guarded do
        @repo.transaction do
          new_id = @repo.insert_entry(text, embedding, meta.try(&.to_json), payload_id, key)
          @index.add(new_id, embedding)
          record_graph_state
        end
      end
      new_id
    end

    # Adds several entries in one transaction. Embedding happens before the
    # transaction opens, so a slow embedder never holds the write lock.
    #
    # ```
    # store.bulk do |batch|
    #   batch.add("The sky is blue.", meta: Meta.new("en"))
    # end
    # ```
    def bulk(& : Batch(M, P) ->) : Nil
      writable!
      batch = Batch(M, P).new
      yield batch
      return if batch.empty?

      prepared = batch.prepare(@embedder)
      guarded do
        @repo.transaction do
          prepared.each do |item|
            id = @repo.insert_entry(item.text, item.vector, item.meta.try(&.to_json), item.payload_id, item.key)
            @index.add(id, item.vector)
          end
          record_graph_state
        end
      end
    end

    def get(id : Int64) : Entry(M, P)?
      readable!
      @repo.entry(id).try { |row| to_entry(row) }
    end

    def get_by_key(key : String) : Entry(M, P)?
      readable!
      @repo.entry_by_key(key).try { |row| to_entry(row) }
    end

    # Live entries, tombstones excluded.
    def size : Int32
      readable!
      @repo.live_count
    end

    # Every entry row, tombstones included.
    def total : Int32
      readable!
      @repo.entry_count
    end

    # Deletes an entry, returning false if it was absent or already deleted.
    #
    # The entry is tombstoned rather than removed: its graph node stays in
    # place as a routing waypoint, because deleting it would sever paths
    # between the live entries it connects. Its text, metadata and payload
    # link are released immediately; `compact!` reclaims the rest.
    def delete(id : Int64) : Bool
      writable!
      deleted = false
      @repo.transaction { deleted = @repo.tombstone(id) }
      deleted
    end

    def delete_by_key(key : String) : Bool
      writable!
      row = @repo.entry_by_key(key)
      return false if row.nil? || row.deleted
      delete(row.id)
    end

    # Deletes a payload and every entry referencing it, returning how many
    # entries went. Detaching the entries first is what keeps the foreign key
    # satisfied without a second deleted flag on the payload row.
    def delete_payload(id : Int64) : Int32
      writable!
      count = 0
      @repo.transaction do
        count = @repo.tombstone_by_payload(id)
        @repo.delete_payload(id)
      end
      count
    end

    # Reclaims the space and graph slots held by deleted entries.
    #
    # Tombstoned rows are purged and the graph is rebuilt from the survivors,
    # in one transaction: a failure part way leaves the store exactly as it
    # was. Entry ids and keys are untouched, so anything holding an id keeps
    # working — only positions inside the graph move, and those were never
    # visible.
    #
    # Deliberately manual. A batch of deletions then one `compact!` costs one
    # rebuild; compacting after each deletion would cost one apiece.
    def compact! : Nil
      writable!
      return if tombstones == 0

      guarded do
        @repo.transaction do
          # The graph goes first: its nodes reference the entry rows about to
          # be purged, and every position is invalid after a rebuild anyway.
          @index.clear
          @repo.purge_tombstoned
          @repo.each_live_vector { |entry_id, vector| @index.add(entry_id, vector) }
          @index.flush
          # Flushed inside this transaction, so the graph is on disk whatever
          # the cache's usual write-through behaviour.
          @repo.set_graph_meta(@index.entry_point, @index.max_layer, graph_saved: true)
        end
      end
    end

    # Entries deleted but not yet compacted away.
    def tombstones : Int32
      readable!
      @repo.entry_count - @repo.live_count
    end

    # -------------------------------------------------------------------------
    # Payloads
    # -------------------------------------------------------------------------

    def add_payload(payload : P) : Int64
      writable!
      @repo.insert_payload(payload.to_json)
    end

    def get_payload(id : Int64) : P?
      readable!
      @repo.payload(id).try { |json| P.from_json(json) }
    end

    # Replaces a payload's content. Entries are not re-embedded: the embedding
    # comes from the entry's text, not from its payload.
    def update_payload(id : Int64, payload : P) : Bool
      writable!
      @repo.update_payload(id, payload.to_json)
    end

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    # Returns up to *k* entries nearest to *query*, nearest first.
    def search(query : String,
               k : Int32 = DEFAULT_K,
               ef_search : Int32 = DEFAULT_EF_SEARCH) : Array(SearchResult(M, P))
      search_vector(@embedder.embed(query), k, ef_search)
    end

    # Searches with an embedding you already have.
    def search_vector(query : Embedding,
                      k : Int32 = DEFAULT_K,
                      ef_search : Int32 = DEFAULT_EF_SEARCH) : Array(SearchResult(M, P))
      readable!
      return [] of SearchResult(M, P) if k <= 0 || @index.size == 0

      # A graph keeps tombstoned nodes as routing waypoints, so a search can
      # return entries that are no longer live. Ask for more and drop them,
      # widening geometrically. The cap matters: without it a heavily
      # tombstoned store escalates to scanning the whole graph on every query,
      # which is a latency cliff rather than a slow answer. Running `compact!`
      # is what restores full results.
      results = [] of SearchResult(M, P)
      request = k
      MAX_OVERSAMPLE_ROUNDS.times do
        request = Math.min(request, @index.size)
        results = resolve(@index.search(query, k: request, ef: ef_search), k)
        break if results.size >= k || request >= @index.size
        request *= 4
      end
      results
    end

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    def stats
      readable!
      cache_stats = @cache.stats
      {
        entries:         @repo.live_count,
        tombstones:      @repo.entry_count - @repo.live_count,
        payloads:        @repo.payload_count,
        indexed_nodes:   @repo.node_count,
        index:           @index.kind,
        dimensions:      @repo.dimensions,
        embedder:        @embedder.model_name,
        cache_mode:      cache_stats[:mode],
        cache_bytes:     cache_stats[:bytes],
        cache_hits:      cache_stats[:hits],
        cache_misses:    cache_stats[:misses],
        cache_evictions: cache_stats[:evictions],
      }
    end

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    # Collects entries so a bulk block can embed them all at once.
    class Batch(M, P)
      record Item(M),
        text : String,
        meta : M?,
        payload_id : Int64?,
        key : String?,
        vector : Embedding

      @pending = [] of {String, M?, Int64?, String?, Embedding?}

      def add(text : String,
              meta : M? = nil,
              payload_id : Int64? = nil,
              key : String? = nil,
              vector : Embedding? = nil) : Nil
        @pending << {text, meta, payload_id, key, vector}
      end

      def empty? : Bool
        @pending.empty?
      end

      def size : Int32
        @pending.size
      end

      # :nodoc:
      # Embeds everything still lacking a vector, in one call to the embedder.
      def prepare(embedder : VectorEmbedder) : Array(Item(M))
        missing_positions = [] of Int32
        missing_texts = [] of String
        @pending.each_with_index do |pending, position|
          next unless pending[4].nil?
          missing_positions << position
          missing_texts << pending[0]
        end

        embedded = embedder.embed_all(missing_texts)
        vectors = {} of Int32 => Embedding
        missing_positions.each_with_index { |position, i| vectors[position] = embedded[i] }

        @pending.map_with_index do |pending, position|
          text, meta, payload_id, key, vector = pending
          Item(M).new(text, meta, payload_id, key, vector || vectors[position])
        end
      end
    end

    # Turns hits into results, dropping any whose entry is tombstoned, and
    # fetching each payload once however many hits share it.
    private def resolve(hits : Array(Index::Hit), k : Int32) : Array(SearchResult(M, P))
      payloads = {} of Int64 => P?
      results = [] of SearchResult(M, P)

      hits.each do |hit|
        break if results.size >= k
        row = @repo.entry(hit.entry_id)
        next if row.nil? || row.deleted
        text = row.text
        next if text.nil?

        payload = row.payload_id.try do |pid|
          payloads.fetch(pid) { payloads[pid] = get_payload(pid) }
        end

        results << SearchResult(M, P).new(
          id: row.id, key: row.key, text: text, score: hit.score,
          meta: row.meta.try { |json| M.from_json(json) },
          payload_id: row.payload_id, payload: payload
        )
      end

      results
    end

    private def to_entry(row : Repository::EntryRow) : Entry(M, P)
      Entry(M, P).new(
        id: row.id,
        key: row.key,
        text: row.text,
        meta: row.meta.try { |json| M.from_json(json) },
        payload_id: row.payload_id,
        payload: row.payload_id.try { |pid| get_payload(pid) }
      )
    end

    # Records the graph's entry point and height in the same transaction as
    # the nodes they describe, so a crash cannot leave metadata pointing at a
    # graph that was never committed.
    private def record_graph_state : Nil
      @repo.set_graph_meta(@index.entry_point, @index.max_layer,
        graph_saved: @index.fully_persisted?)
    end

    # A failed transaction rolls back the rows but not the strategy's own
    # state, so the index is rebuilt from what actually survived.
    private def guarded(&)
      yield
    rescue ex
      rebuild_index
      raise ex
    end

    private def rebuild_index : Nil
      @index.clear
      @repo.transaction do
        @repo.each_live_vector { |entry_id, vector| @index.add(entry_id, vector) }
        record_graph_state
      end
    end

    private def restore_index(config : Index::Config) : Nil
      stored_kind = @repo.meta_text("index_kind")
      @repo.set_meta("index_kind", 1, config.kind.to_s) unless @readonly || stored_kind == config.kind.to_s

      unless config.graph?
        # A graph left by a previous strategy would go stale as entries are
        # added, so it is discarded rather than left to mislead a later open.
        if @repo.node_count > 0 && !@readonly
          @repo.transaction do
            @repo.clear_nodes
            @repo.set_graph_meta(-1, -1, graph_saved: false)
          end
        end
        return
      end

      return if @repo.entry_count == 0

      meta = @repo.graph_meta
      if stored_kind == config.kind.to_s && meta[:graph_saved] && @repo.node_count > 0
        @cache.load
        @index.reset_with(meta[:entry_point], meta[:max_layer])
        return
      end

      # Either the graph was never persisted, or a different strategy wrote
      # this database. The vectors are the source of truth either way.
      raise Error.new("Index must be rebuilt, but the store is readonly.") if @readonly
      rebuild_index
    end

    private def check_embedder : Nil
      stored = @repo.embedder_name
      if stored.nil?
        @repo.embedder_name = @embedder.model_name unless @readonly
        return
      end

      return if stored == @embedder.model_name
      raise Error.new(
        "Database was written with embedder '#{stored}' but '#{@embedder.model_name}' " \
        "was supplied. Vectors from different models are not comparable; pass " \
        "verify_embedder: false only if you know they are."
      )
    end

    private def readable! : Nil
      raise Error.new("Store is closed.") if @closed
    end

    private def writable! : Nil
      readable!
      raise Error.new("Store is readonly.") if @readonly
    end
  end
end
