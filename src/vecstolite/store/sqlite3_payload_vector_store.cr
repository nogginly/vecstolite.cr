require "sqlite3"
require "json"

require "../indexer/*"
require "../sucre/cache"
require "./sqlite3_node_store"

module Vecstolite
  # A SQLite-backed vector store with typed per-embedding metadata (`M`) and
  # shared payload objects (`P`). Both `M` and `P` must support `to_json` and
  # `T.from_json(string)` — i.e. any `JSON::Serializable` type, `Hash`, or
  # `JSON::Any`.
  #
  # Conceptual model:
  #   - `meta`    — optional data *about* the embedding (language code, page offset, …)
  #   - `payload` — the primary content returned by search (translation pair, page text, …)
  #                 Many embeddings may reference the same payload (many-to-one).
  #
  # Memory modes (controlled by `cache_max_bytes`):
  #   - **Smart cache** (default, 512 MB): LRU-evicted node cache; balanced memory/speed.
  #   - **No cache** (`cache_max_bytes: nil`): every traversal hop reads from disk; minimal RAM.
  #   - **Eager load** (call `load_all_in_memory!` after open): full index in RAM; fastest search.
  #
  # Thread safety: **single-threaded only**. For concurrent access, serialise
  # externally with a `Mutex` per store instance.
  #
  # Usage:
  # ```
  # store = Vecstolite::SQLitePayloadVectorStore(Lang, Translation).create("store.db", embedder)
  # pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
  # store.add("The sky is blue", meta: Lang.new("en"), payload_id: pid)
  # store.add("Le ciel est bleu", meta: Lang.new("fr"), payload_id: pid)
  # results = store.search("colour of the sky", k: 3)
  # store.close
  # ```
  class SQLitePayloadVectorStore(M, P)
    include IndexedVectorStore(M)

    class Error < Exception; end

    SCHEMA_VERSION          = 3
    DEFAULT_CACHE_MAX_BYTES = 512_i64 * MB
    DEFAULT_K               =   5
    DEFAULT_M               =  16
    DEFAULT_EF_CONSTRUCTION = 200
    DEFAULT_EF_SEARCH       =  50

    # Result returned by `search`. Carries the matched embedding text, similarity
    # score, optional per-embedding metadata, and the optional resolved payload.
    record SearchResult(M, P),
      text : String,
      score : Float32,
      meta : M?,
      payload : P? do
      include VectorSearchResult
    end

    # Yielded by `bulk_add` to allow many entries and payloads to be added
    # within a single database transaction, reducing per-add fsync overhead.
    # The API mirrors the store's own methods — any developer familiar with
    # the store needs no extra learning.
    class Batch(M, P)
      protected def initialize(@store : SQLitePayloadVectorStore(M, P))
      end

      # Add and index *text*. Mirrors `SQLitePayloadVectorStore#add`.
      def add(text : String, meta : M? = nil, payload_id : Int64? = nil) : Nil
        @store.add_within_bulk(text, meta, payload_id)
      end

      # Store a shared payload and return its id. Mirrors `SQLitePayloadVectorStore#add_payload`.
      # The insert is part of the enclosing `bulk_add` transaction — if the block
      # raises, the payload is rolled back along with all entries.
      def add_payload(payload : P) : Int64
        @store.add_payload_within_bulk(payload)
      end
    end

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    # Open an existing store at *path*.
    #
    # *cache_max_bytes* controls the node cache budget:
    #   - Integer (default 512 MB): LRU cache capped at that many bytes.
    #   - `nil`: no cache; every traversal hop reads from disk.
    #
    # Call `load_all_in_memory!` after `open` to switch to full in-memory mode.
    def self.open(path : String,
                  embedder : VectorEmbedder,
                  readonly = false,
                  cache_max_bytes : Int64? = DEFAULT_CACHE_MAX_BYTES,
                  cache_ttl : Time::Span? = nil,
                  cache_purge_period : Time::Span? = nil) : self
      raise Error.new("Database '#{path}' does not exist.") unless File.exists?(path)
      # DB.connect rather than DB.open: SQLite is single-writer; a connection pool
      # offers no benefit and complicates transaction safety — all exec calls inside
      # a transaction block must share the same connection. A single DB::Connection
      # guarantees this without threading tx.connection through every call site
      # (including deep into NodeStore). See DEVELOPMENT.md § Design Decisions.
      db = DB.connect("sqlite3://#{path}")
      store = new(embedder, db, path,
        readonly: readonly,
        cache_max_bytes: cache_max_bytes,
        cache_ttl: cache_ttl,
        cache_purge_period: cache_purge_period)
      store.bootstrap
      store
    end

    # Create a new store at *path*.
    #
    # *cache_max_bytes* controls the node cache budget (see `open` for details).
    # *hnsw_seed* fixes the random seed for HNSW layer assignment, producing a
    # reproducible graph topology. Useful for tests and debugging. Omit (or pass
    # `nil`) for the default entropy-seeded behaviour.
    def self.create(path : String,
                    embedder : VectorEmbedder,
                    m = DEFAULT_M,
                    ef_construction = DEFAULT_EF_CONSTRUCTION,
                    cache_max_bytes : Int64? = DEFAULT_CACHE_MAX_BYTES,
                    hnsw_seed : Int32? = nil,
                    cache_ttl : Time::Span? = nil,
                    cache_purge_period : Time::Span? = nil) : self
      raise Error.new("Database '#{path}' already exists.") if File.exists?(path)
      # See open() above for why DB.connect is used instead of DB.open.
      db = DB.connect("sqlite3://#{path}")
      store = new(embedder, db, path,
        readonly: false,
        m: m, ef_construction: ef_construction,
        cache_max_bytes: cache_max_bytes,
        hnsw_seed: hnsw_seed,
        cache_ttl: cache_ttl,
        cache_purge_period: cache_purge_period)
      store.bootstrap
      store
    end

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    # Flush the graph (if needed) and close the backing database.
    def close : Nil
      return if @closed
      save_graph unless @readonly
      @db.close
      @closed = true
    end

    def clear : Nil
      raise Error.new("Clearing SQLitePayloadVectorStore not yet supported.")
    end

    # -------------------------------------------------------------------------
    # Graph persistence
    # -------------------------------------------------------------------------

    # Ensure the HNSW graph topology is durable.
    #
    # For `LRUNodeStore`/`DiskNodeStore`: nodes are already persisted; only
    # updates the `entry_point`, `max_layer`, and `graph_saved` meta keys.
    #
    # For `MemoryNodeStore`: writes the full neighbour lists to `vecsto_nodes`,
    # then updates meta.
    def save_graph : Nil
      @db.transaction do
        unless @node_store.fully_persisted?
          # MemoryNodeStore path: flush all neighbour lists to DB.
          @db.exec "DELETE FROM #{TABLE_NODES}"
          @node_store.each_node do |id, node|
            @db.exec "INSERT INTO #{TABLE_NODES} (id, neighbours) VALUES (?, ?)",
              id, NodePacking.pack(node.neighbours)
          end
        end
        update_graph_meta
      end
    end

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    # Returns a `NamedTuple` with:
    # - `cache_mode`      — `:memory`, `:lru`, or `:disk`
    # - `cache_nodes`     — nodes currently in the LRU cache (0 for others)
    # - `cache_capacity`  — estimated max nodes for the configured budget
    # - `cache_hits`      — cumulative LRU cache hits
    # - `cache_misses`    — cumulative cache misses (disk reads)
    # - `cache_evictions` — cumulative LRU evictions
    # - `embeddings`      — total entries in the index
    # - `indexed_nodes`   — nodes in the HNSW index
    # - `cached_entries`  — entries in the text/meta lazy cache
    def stats : NamedTuple
      dims = @embedder.dimensions
      bytes_per_node = (dims * 4 + 128).to_i64
      capacity = case max_bytes = @cache_max_bytes
                 in Int64 then (max_bytes / bytes_per_node).to_i32
                 in Nil   then 0
                 end
      {
        cache_mode:      @node_store.cache_mode,
        cache_nodes:     @node_store.cache_size,
        cache_capacity:  capacity,
        cache_hits:      @node_store.cache_hits,
        cache_misses:    @node_store.cache_misses,
        cache_evictions: @node_store.cache_evictions,
        embeddings:      @index.size,
        indexed_nodes:   @index.size,
        cached_entries:  @entry_cache.size,
      }
    end

    # -------------------------------------------------------------------------
    # Payload management
    # -------------------------------------------------------------------------

    # Store a shared payload and return its id for use with `add`.
    def add_payload(payload : P) : Int64
      raise Error.new("Store is closed") if @closed
      raise Error.new("Store is readonly") if @readonly
      payload_id = Int64.new(0)
      @db.transaction do
        result = @db.exec "INSERT INTO #{TABLE_PAYLOADS} (content) VALUES (?)", payload.to_json
        payload_id = result.last_insert_id
      end
      payload_id
    end

    # Retrieve a payload by id. Returns nil if not found.
    def get_payload(payload_id : Int64) : P?
      result = nil
      @db.query("SELECT content FROM #{TABLE_PAYLOADS} WHERE id = ?", payload_id) do |result_set|
        result_set.each { result = P.from_json(result_set.read(String)) }
      end
      result
    end

    # Called by Batch#add_payload — skips the transaction wrapper.
    protected def add_payload_within_bulk(payload : P) : Int64
      result = @db.exec "INSERT INTO #{TABLE_PAYLOADS} (content) VALUES (?)", payload.to_json
      result.last_insert_id
    end

    # -------------------------------------------------------------------------
    # Mutations
    # -------------------------------------------------------------------------

    def add(text : String, extra : M? = nil) : Nil
      add(text, meta: extra)
    end

    # Add and index *text* within its own transaction.
    # - *meta*       — optional metadata about this embedding (e.g. language, offset).
    # - *payload_id* — optional id of a shared payload created via `add_payload`.
    def add(text : String, meta : M? = nil, payload_id : Int64? = nil) : Nil
      raise Error.new("Store is closed") if @closed
      raise Error.new("Store is readonly") if @readonly
      purge_expired_from_cache

      vector = @embedder.embed(text)
      id = @index.size
      begin
        @db.transaction do
          @db.exec "INSERT INTO #{TABLE_ENTRIES} (id, text, vector, meta, payload_id) VALUES (?, ?, ?, ?, ?)",
            id, text, pack_vector(vector), meta.try(&.to_json), payload_id
          @index.add(id: id, vector: vector)
          update_graph_meta if @node_store.fully_persisted?
        end
      rescue ex
        # DB transaction rolled back. For LRU/Disk stores, in-memory cache
        # entries for back-edge neighbours may have been mutated before the
        # failure — rebuild to restore consistency.
        rebuild_after_rollback(id) if @node_store.fully_persisted?
        raise ex
      end
      @entry_cache.put(id, CachedEntry(M).new(text, meta.as(M?), payload_id))
    end

    # Add many entries and payloads within a **single database transaction**,
    # dramatically reducing per-entry fsync overhead for bulk ingest.
    #
    # The block receives a `Batch` whose `add` and `add_payload` methods mirror
    # the store's own API. If the block raises, the transaction is rolled back
    # and the store is left unchanged — no orphaned payload rows, no partial index.
    #
    # Example:
    # ```
    # store.bulk_add do |batch|
    #   inputs.each do |input|
    #     pid = batch.add_payload(input.translation)
    #     batch.add(input.en, meta: Lang.new("en"), payload_id: pid)
    #     batch.add(input.fr, meta: Lang.new("fr"), payload_id: pid)
    #   end
    # end
    # ```
    def bulk_add(& : Batch(M, P) ->) : Nil
      raise Error.new("Store is closed") if @closed
      raise Error.new("Store is readonly") if @readonly
      purge_expired_from_cache

      id_before = @index.size
      batch = Batch(M, P).new(self)
      begin
        @db.transaction do
          yield batch
          update_graph_meta if @node_store.fully_persisted?
        end
      rescue ex
        # DB transaction rolled back; restore in-memory state to match.
        rebuild_after_rollback(id_before)
        raise ex
      end
    end

    # Load the entire index into memory, switching from LRU or Disk mode to
    # full in-memory storage. Subsequent searches hit no disk.
    #
    # This is a one-time startup cost linear in index size. The caller is
    # responsible for ensuring sufficient RAM. Calling this when already in
    # memory mode is a no-op.
    def load_all_in_memory! : Nil
      return if @node_store.is_a?(HNSW::MemoryNodeStore)

      node_count = @node_store.size
      return if node_count == 0

      vectors = Array(Embedding).new(node_count)
      @db.query("SELECT vector FROM #{TABLE_ENTRIES} WHERE deleted = 0 ORDER BY id") do |result_set|
        result_set.each { vectors << unpack_vector(result_set.read(Bytes)) }
      end

      all_neighbours = Array(Array(Array(Int32))).new(node_count) { [] of Array(Int32) }
      @db.query("SELECT id, neighbours FROM #{TABLE_NODES} ORDER BY id") do |result_set|
        result_set.each do
          nid = result_set.read(Int32)
          blob = result_set.read(Bytes)
          all_neighbours[nid] = NodePacking.unpack(blob) if nid < node_count
        end
      end

      mem_store = HNSW::MemoryNodeStore.new
      @index = HNSW::Index.restore(
        dims: @embedder.dimensions,
        m: @m,
        ef_construction: @ef_construction,
        entry_point: @index.entry_point,
        max_layer: @index.max_layer,
        node_count: node_count,
        node_store: mem_store
      ) { |id| {vectors[id], all_neighbours[id]} }
      @node_store = mem_store
    end

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def search(query : String, k : Int32 = DEFAULT_K, ef_search : Int32 = DEFAULT_EF_SEARCH) : Array
      raise Error.new("Store is closed") if @closed
      return [] of SearchResult(M, P) if @index.size == 0
      purge_expired_from_cache

      query_vec = @embedder.embed(query)
      @index.search(query_vec, k: k, ef: ef_search).map do |result|
        entry = get_entry(result.id)
        payload = entry.payload_id.try { |pid| get_payload(pid) }
        SearchResult.new(entry.text, result.score, entry.meta, payload)
      end
    end

    def size : Int32
      @index.size
    end

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    private TABLE_META     = "vecsto_meta"
    private TABLE_ENTRIES  = "vecsto_entries"
    private TABLE_PAYLOADS = "vecsto_payloads"
    private TABLE_NODES    = "vecsto_nodes"

    private record CachedEntry(M),
      text : String,
      meta : M?,
      payload_id : Int64?

    @readonly : Bool
    @db : DB::Connection
    @path : String
    @embedder : VectorEmbedder
    @node_store : HNSW::NodeStore
    @index : HNSW::Index
    @closed : Bool
    @m : Int32
    @ef_construction : Int32
    @cache_max_bytes : Int64?
    @hnsw_seed : Int32?

    @entry_cache : Cache(Int32, CachedEntry(M))
    @cache_last_purged = Time.instant
    @cache_purge_period : Time::Span?

    private def initialize(@embedder : VectorEmbedder, @db : DB::Connection, @path : String,
                           @readonly, @m : Int32 = DEFAULT_M,
                           @ef_construction : Int32 = DEFAULT_EF_CONSTRUCTION,
                           cache_max_bytes : Int64? = DEFAULT_CACHE_MAX_BYTES,
                           hnsw_seed : Int32? = nil,
                           cache_ttl : Time::Span? = nil,
                           @cache_purge_period : Time::Span? = nil)
      @cache_max_bytes = cache_max_bytes
      @hnsw_seed = hnsw_seed
      @entry_cache = Cache(Int32, CachedEntry(M)).new(cache_ttl)
      @node_store = make_node_store
      @index = new_index
      @closed = false
    end

    protected def bootstrap : Nil
      unless @readonly
        @db.exec "PRAGMA journal_mode = WAL"
        @db.exec "PRAGMA synchronous  = FULL"
        @db.exec "PRAGMA foreign_keys = ON"

        @db.exec <<-SQL
          CREATE TABLE IF NOT EXISTS #{TABLE_META} (
            key   TEXT    PRIMARY KEY,
            value INTEGER NOT NULL,
            text  TEXT    DEFAULT NULL
          )
        SQL

        @db.exec <<-SQL
          CREATE TABLE IF NOT EXISTS #{TABLE_PAYLOADS} (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT    NOT NULL
          )
        SQL

        @db.exec <<-SQL
          CREATE TABLE IF NOT EXISTS #{TABLE_ENTRIES} (
            id         INTEGER PRIMARY KEY,
            text       TEXT    NOT NULL,
            vector     BLOB    NOT NULL,
            meta       TEXT    DEFAULT NULL,
            payload_id INTEGER DEFAULT NULL REFERENCES #{TABLE_PAYLOADS}(id),
            deleted    INTEGER NOT NULL DEFAULT 0
          )
        SQL

        @db.exec <<-SQL
          CREATE TABLE IF NOT EXISTS #{TABLE_NODES} (
            id         INTEGER PRIMARY KEY,
            neighbours BLOB NOT NULL
          )
        SQL

        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('schema_version', ?, NULL)", SCHEMA_VERSION
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('m',              ?, NULL)", @m
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('ef_construction',?, NULL)", @ef_construction
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('graph_saved',    ?, NULL)", 0
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('entry_point',    ?, NULL)", -1
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('max_layer',      ?, NULL)", -1
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('dimensions',     ?, NULL)", @embedder.dimensions
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('embedder',      1, ?)", @embedder.model_name
      end

      meta = read_meta
      existing_version = meta["schema_version"]? || 0
      if existing_version > 0 && existing_version < SCHEMA_VERSION
        raise Error.new(
          "Database schema version #{existing_version} is not supported " \
          "(current: #{SCHEMA_VERSION}). Please recreate the database " \
          "(re-run your ingest pipeline)."
        )
      end

      @m = meta["m"]? || DEFAULT_M
      @ef_construction = meta["ef_construction"]? || DEFAULT_EF_CONSTRUCTION
      @node_store = make_node_store # rebuild with correct dims from embedder
      @index = new_index

      load_from_db(meta)
    end

    # Called by Batch#add — skips the transaction wrapper (uses bulk_add's transaction).
    protected def add_within_bulk(text : String, meta : M? = nil, payload_id : Int64? = nil) : Nil
      vector = @embedder.embed(text)
      id = @index.size
      @db.exec "INSERT INTO #{TABLE_ENTRIES} (id, text, vector, meta, payload_id) VALUES (?, ?, ?, ?, ?)",
        id, text, pack_vector(vector), meta.try(&.to_json), payload_id
      @index.add(id: id, vector: vector)
      @entry_cache.put(id, CachedEntry(M).new(text, meta.as(M?), payload_id))
    end

    private def load_from_db(meta : Hash(String, Int32)) : Nil
      return if (meta["entry_point"]? || -1) == -1

      if (meta["graph_saved"]? || 0) == 1
        restore_graph_from_db(meta)
      else
        # MemoryNodeStore only: graph wasn't saved before last close; rebuild.
        rebuild_index_from_entries
      end
    end

    private def restore_graph_from_db(meta : Hash(String, Int32)) : Nil
      node_count = 0
      @db.scalar("SELECT COUNT(*) FROM #{TABLE_ENTRIES} WHERE deleted = 0").tap do |v|
        node_count = v.as(Int64).to_i32
      end
      return if node_count == 0

      @index = HNSW::Index.restore(
        dims: @embedder.dimensions,
        m: @m,
        ef_construction: @ef_construction,
        entry_point: meta["entry_point"]? || -1,
        max_layer: meta["max_layer"]? || -1,
        node_count: node_count,
        node_store: @node_store
      ) do |id|
        # Called only by MemoryNodeStore#restore_from_persisted.
        vector = uninitialized Embedding
        neighbours = uninitialized Array(Array(Int32))
        @db.query(
          "SELECT e.vector, n.neighbours FROM #{TABLE_ENTRIES} e
           JOIN #{TABLE_NODES} n ON e.id = n.id
           WHERE e.id = ? AND e.deleted = 0", id
        ) do |result_set|
          result_set.each do
            vector = unpack_vector(result_set.read(Bytes))
            neighbours = NodePacking.unpack(result_set.read(Bytes))
          end
        end
        {vector, neighbours}
      end
    end

    private def rebuild_after_rollback(id_before : Int32) : Nil
      # Evict rolled-back entries from the text/meta cache.
      @index.size.times { |id| @entry_cache.delete(id) if id >= id_before }

      if @node_store.is_a?(HNSW::MemoryNodeStore)
        # Extract surviving vectors before discarding the old store.
        vectors = Array(Embedding).new(id_before) { |i| @node_store.get(i).vector }
        fresh_store = HNSW::MemoryNodeStore.new
        fresh_index = HNSW::Index.new(
          dims: @embedder.dimensions, m: @m, ef_construction: @ef_construction,
          node_store: fresh_store, seed: @hnsw_seed)
        vectors.each_with_index { |v, i| fresh_index.add(id: i, vector: v) }
        @node_store = fresh_store
        @index = fresh_index
      else
        # LRU/Disk: DB was already rolled back to id_before entries.
        # Build a fresh node store and a correctly-sized HNSW index directly —
        # restore_from_persisted sets the count; the block is never called for
        # disk-backed stores so the dummy return value is irrelevant.
        fresh_store = make_node_store
        meta = read_meta
        entry_point = meta["entry_point"]? || -1
        max_layer = meta["max_layer"]? || -1
        fresh_index = HNSW::Index.restore(
          dims: @embedder.dimensions,
          m: @m,
          ef_construction: @ef_construction,
          entry_point: (entry_point >= 0 && id_before > 0) ? entry_point : -1,
          max_layer: (max_layer >= 0 && id_before > 0) ? max_layer : -1,
          node_count: id_before,
          node_store: fresh_store
        ) { |_| {Embedding.new(@embedder.dimensions) { 0_f32 }, [] of Array(Int32)} }
        @node_store = fresh_store
        @index = fresh_index
      end
    end

    private def rebuild_index_from_entries : Nil
      @index = new_index
      @db.query("SELECT vector FROM #{TABLE_ENTRIES} WHERE deleted = 0 ORDER BY id") do |result_set|
        id = 0
        result_set.each do
          @index.add(id: id, vector: unpack_vector(result_set.read(Bytes)))
          id += 1
        end
      end
    end

    private def get_entry(id : Int32) : CachedEntry(M)
      @entry_cache.get(id) { retrieve_entry(id) }
    end

    private def retrieve_entry(id : Int32) : CachedEntry(M)
      entry = nil
      @db.query(
        "SELECT id, text, meta, payload_id FROM #{TABLE_ENTRIES} WHERE id = ? AND deleted = 0", id
      ) do |result_set|
        result_set.each do
          _id = result_set.read(Int32)
          text = result_set.read(String)
          meta_json = result_set.read(String?)
          payload_id = result_set.read(Int64?)
          entry = CachedEntry.new(text, meta_json.try { |j| M.from_json(j) }, payload_id)
        end
      end
      entry || raise Error.new("Unexpected error fetching entry (id = #{id}) from DB.")
    end

    private def update_graph_meta : Nil
      @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('entry_point', ?, NULL)", @index.entry_point
      @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('max_layer',   ?, NULL)", @index.max_layer
      @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('graph_saved', ?, NULL)", 1
    end

    private def make_node_store : HNSW::NodeStore
      case max_bytes = @cache_max_bytes
      in Int64
        LRUNodeStore.new(@db, @embedder.dimensions, TABLE_ENTRIES, TABLE_NODES, max_bytes)
      in Nil
        DiskNodeStore.new(@db, @embedder.dimensions, TABLE_ENTRIES, TABLE_NODES)
      end
    end

    private def new_index : HNSW::Index
      HNSW::Index.new(dims: @embedder.dimensions, m: @m, ef_construction: @ef_construction,
        node_store: @node_store, seed: @hnsw_seed)
    end

    private def purge_expired_from_cache : Nil
      if purge_delay = @cache_purge_period
        if (now = Time.instant) - @cache_last_purged > purge_delay
          @cache_last_purged = now
          @entry_cache.purge_expired
        end
      end
    end

    private def read_meta : Hash(String, Int32)
      meta = {} of String => Int32
      @db.query("SELECT key, value FROM #{TABLE_META}") do |result_set|
        result_set.each { meta[result_set.read(String)] = result_set.read(Int32) }
      end
      meta
    end

    private def pack_vector(vec : Embedding) : Bytes
      buf = Bytes.new(vec.size * 4)
      vec.each_with_index { |v, i| IO::ByteFormat::LittleEndian.encode(v, buf[i * 4, 4]) }
      buf
    end

    private def unpack_vector(blob : Bytes) : Embedding
      Embedding.new(@embedder.dimensions) do |i|
        IO::ByteFormat::LittleEndian.decode(Float32, blob[i * 4, 4])
      end
    end
  end
end
