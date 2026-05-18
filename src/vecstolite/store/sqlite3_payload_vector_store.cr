require "sqlite3"
require "json"

require "../indexer/*"
require "../sucre/cache"

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
  # Design:
  #   - Entries (text + vector + optional meta + optional payload_id) written on every `add`.
  #   - Payloads live in a separate table; created via `add_payload`.
  #   - HNSW graph lives in memory; flushed to SQLite on `close` or `save_graph`.
  #   - On `open`, entries and vectors are loaded eagerly; text/meta/payload fetched lazily.
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
  #
  class SQLitePayloadVectorStore(M, P)
    include IndexedVectorStore(M)

    class Error < Exception; end

    SCHEMA_VERSION          =   2
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

    # Open an existing store at `path`.
    # Optional `cache_max_bytes` enables smart node caching for large indices
    # (set to nil to disable caching and load full index into memory on each open).
    def self.open(path : String,
                  embedder : VectorEmbedder,
                  readonly = false,
                  cache_ttl : Time::Span? = nil,
                  cache_purge_period : Time::Span? = nil,
                  cache_max_bytes : Int32? = 536_870_912) : self
      raise Error.new("Database '#{path}' does not exist.") unless File.exists?(path)

      db = DB.open("sqlite3://#{path}")
      store = new(embedder, db, path,
        readonly: readonly,
        cache_ttl: cache_ttl,
        cache_purge_period: cache_purge_period,
        cache_max_bytes: cache_max_bytes)
      store.bootstrap
      store
    end

    # Create a new store at `path`.
    # Optional `cache_max_bytes` enables smart node caching for large indices
    # (set to nil to disable caching; use `load_all_in_memory!` to eager-load instead).
    def self.create(path : String,
                    embedder : VectorEmbedder,
                    m = DEFAULT_M,
                    ef_construction = DEFAULT_EF_CONSTRUCTION,
                    cache_ttl : Time::Span? = nil,
                    cache_purge_period : Time::Span? = nil,
                    cache_max_bytes : Int32? = 536_870_912) : self
      raise Error.new("Database '#{path}' already exists.") if File.exists?(path)

      db = DB.open("sqlite3://#{path}")
      store = new(embedder, db, path, m, ef_construction,
        readonly: false,
        cache_ttl: cache_ttl,
        cache_purge_period: cache_purge_period,
        cache_max_bytes: cache_max_bytes)
      store.bootstrap
      store
    end

    # Flush graph and close the backing database.
    def close : Nil
      return if @closed
      save_graph unless @readonly
      @db.close
      @closed = true
    end

    def clear : Nil
      raise Error.new("Clearing SQLitePayloadVectorStore not yet supported.")
    end

    private TABLE_GRAPH_NODES = "vecsto_graph_nodes"
    private TABLE_GRAPH_EDGES = "vecsto_graph_edges"
    private TABLE_META        = "vecsto_meta"
    private TABLE_ENTRIES     = "vecsto_entries"
    private TABLE_PAYLOADS    = "vecsto_payloads"
    private INDEX_EDGES       = "vecsto_idx_edges"

    # Flush the in-memory HNSW graph topology to SQLite.
    def save_graph : Nil
      @db.transaction do
        @db.exec "DELETE FROM #{TABLE_GRAPH_NODES}"
        @db.exec "DELETE FROM #{TABLE_GRAPH_EDGES}"

        meta = @index.export do |node_id, neighbours|
          @db.exec "INSERT INTO #{TABLE_GRAPH_NODES} VALUES (?, ?)",
            node_id, neighbours.size

          neighbours.each_with_index do |layer_neighbours, layer|
            layer_neighbours.each do |nb_id|
              @db.exec "INSERT INTO #{TABLE_GRAPH_EDGES} VALUES (?, ?, ?)",
                node_id, layer, nb_id
            end
          end
        end

        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('entry_point', ?, NULL)", meta[:entry_point]
        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('max_layer',   ?, NULL)", meta[:max_layer]
        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('graph_saved', ?, NULL)", 1
      end

      # Now that graph is saved to DB, switch to CachedNodeProvider if caching is enabled
      setup_node_provider if @node_cache
    end

    # Returns stats `NamedTuple`:
    # - `embeddings`    — total entries loaded
    # - `indexed_nodes` — nodes in the HNSW index (should match `embeddings`)
    # - `cached`        — entries currently in the lazy cache
    def stats : NamedTuple
      {cached: @entry_cache.size, embeddings: @entry_embeddings.size, indexed_nodes: @index.size}
    end

    # Eagerly load the entire HNSW graph into memory, disabling node caching.
    # Useful for datasets that fit in RAM and benefit from direct array access.
    # After calling this, the index will use DirectNodeProvider instead of CachedNodeProvider.
    def load_all_in_memory! : Nil
      raise Error.new("Store is closed") if @closed
      return if @node_cache.nil? # Already loaded or caching disabled

                # Disable the cache and switch back to DirectNodeProvider
                # (All nodes are now in memory via @index.nodes)
      @node_cache = nil
      @index.set_node_provider(HNSW::DirectNodeProvider.new(@index.nodes))
    end

    # Setup the node provider for the HNSW index based on cache configuration.
    # If caching is enabled AND graph has been persisted, creates a CachedNodeProvider with a DB loader.
    # Otherwise, uses the DirectNodeProvider for in-memory access.
    private def setup_node_provider : Nil
      # Check if graph has been saved to DB (nodes are available to load)
      graph_saved = @db.query_one?("SELECT value FROM #{TABLE_META} WHERE key = 'graph_saved'", &.read(Int32)) == 1

      if graph_saved && (cache = @node_cache)
        # Create a loader that fetches nodes from the DB
        loader = ->(id : Int32) do
          node_id = id
          layer_count = 0
          neighbours = [] of Array(Int32)

          # Load the node's layer count
          @db.query_one(
            "SELECT layer_count FROM #{TABLE_GRAPH_NODES} WHERE id = ?",
            node_id
          ) do |rs|
            layer_count = rs.read(Int32)
          end

          # Initialize empty neighbor lists
          neighbours = Array(Array(Int32)).new(layer_count) { [] of Int32 }

          # Load neighbors for each layer
          @db.query(
            "SELECT layer, neighbour_id FROM #{TABLE_GRAPH_EDGES} WHERE node_id = ? ORDER BY layer",
            node_id
          ) do |rs|
            rs.each do
              layer = rs.read(Int32)
              neighbour_id = rs.read(Int32)
              neighbours[layer] << neighbour_id
            end
          end

          # Construct the HNSWNode with vector and neighbors
          vector = @entry_embeddings[node_id].vector
          node = HNSW::HNSWNode.new(vector, layer_count - 1, @m)
          node.neighbours = neighbours
          node
        end

        provider = HNSW::CachedNodeProvider.new(cache, &loader)
        @index.set_node_provider(provider)
      else
        # No caching or graph not yet saved: use DirectNodeProvider for in-memory array access
        provider = HNSW::DirectNodeProvider.new(@index.nodes)
        @index.set_node_provider(provider)
      end
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
        result_set.each do
          result = P.from_json(result_set.read(String))
        end
      end
      result
    end

    # -------------------------------------------------------------------------
    # Mutations
    # -------------------------------------------------------------------------

    def add(text : String, extra : M? = nil) : Nil
      add(text, meta: extra)
    end

    # Add and index `text`.
    # - `meta`       — optional metadata about this embedding (e.g. language, offset).
    # - `payload_id` — optional id of a shared payload created via `add_payload`.
    def add(text : String, meta : M? = nil, payload_id : Int64? = nil) : Nil
      raise Error.new("Store is closed") if @closed
      raise Error.new("Store is readonly") if @readonly

      purge_expired_from_cache

      vector = @embedder.embed(text)
      id = @entry_embeddings.size

      @db.transaction do
        @db.exec "INSERT INTO #{TABLE_ENTRIES} (id, text, vector, meta, payload_id) VALUES (?, ?, ?, ?, ?)",
          id, text, pack_vector(vector), meta.try(&.to_json), payload_id
      end

      @entry_cache.put(id, CachedEntry.new(text, vector, meta, payload_id))
      @entry_embeddings << EntryVector.new(vector)
      @index.add(id: id, vector: vector)
    end

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def search(query : String, k : Int32 = DEFAULT_K, ef_search : Int32 = DEFAULT_EF_SEARCH) : Array
      raise Error.new("Store is closed") if @closed

      return [] of SearchResult(M, P) if @entry_embeddings.empty?

      purge_expired_from_cache

      query_vec = @embedder.embed(query)
      @index.search(query_vec, k: k, ef: ef_search).map do |result|
        entry = get_entry(result.id)
        payload = entry.payload_id.try { |pid| get_payload(pid) }
        SearchResult.new(entry.text, result.score, entry.meta, payload)
      end
    end

    def size : Int32
      @entry_embeddings.size
    end

    # -------------------------------------------------------------------------

    private record EntryVector, vector : Embedding

    private record CachedEntry(M),
      text : String,
      vector : Embedding,
      meta : M?,
      payload_id : Int64?

    @readonly : Bool
    @db : DB::Database
    @path : String
    @entry_embeddings : Array(EntryVector)
    @embedder : VectorEmbedder
    @index : HNSW::Index
    @closed : Bool
    @m : Int32
    @ef_construction : Int32

    @entry_cache : Cache(Int32, CachedEntry(M))
    @cache_last_purged = Time.instant
    @cache_purge_period : Time::Span?
    @node_cache : HNSW::NodeCache?

    private def initialize(@embedder : VectorEmbedder, @db : DB::Database, @path : String,
                           @m : Int32, @ef_construction : Int32, @readonly,
                           cache_ttl : Time::Span? = nil,
                           @cache_purge_period : Time::Span? = nil,
                           cache_max_bytes : Int32? = nil)
      @entry_cache = Cache(Int32, CachedEntry(M)).new(cache_ttl)
      @entry_embeddings = [] of EntryVector
      @index = new_index
      @node_cache = cache_max_bytes ? HNSW::NodeCache.new(cache_max_bytes) : nil
      @closed = false
    end

    private def initialize(@embedder : VectorEmbedder, @db : DB::Database,
                           @path : String, @readonly,
                           cache_ttl : Time::Span? = nil,
                           @cache_purge_period : Time::Span? = nil,
                           cache_max_bytes : Int32? = nil)
      @entry_cache = Cache(Int32, CachedEntry(M)).new(cache_ttl)
      @entry_embeddings = [] of EntryVector
      @m = DEFAULT_M
      @ef_construction = DEFAULT_EF_CONSTRUCTION
      @index = new_index
      @node_cache = cache_max_bytes ? HNSW::NodeCache.new(cache_max_bytes) : nil
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
      CREATE TABLE IF NOT EXISTS #{TABLE_GRAPH_NODES} (
        id          INTEGER PRIMARY KEY,
        layer_count INTEGER NOT NULL
      )
      SQL

        @db.exec <<-SQL
      CREATE TABLE IF NOT EXISTS #{TABLE_GRAPH_EDGES} (
        node_id      INTEGER NOT NULL,
        layer        INTEGER NOT NULL,
        neighbour_id INTEGER NOT NULL
      )
      SQL

        @db.exec "CREATE INDEX IF NOT EXISTS #{INDEX_EDGES} ON #{TABLE_GRAPH_EDGES} (node_id, layer)"

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
      @m = meta["m"]
      @ef_construction = meta["ef_construction"]
      @index = new_index

      load_from_db(meta)
      setup_node_provider
    end

    private def load_from_db(meta : Hash(String, Int32)) : Nil
      @db.query("SELECT vector FROM #{TABLE_ENTRIES} WHERE deleted = 0 ORDER BY id") do |results|
        results.each do
          blob = results.read(Bytes)
          @entry_embeddings << EntryVector.new(unpack_vector(blob))
        end
      end
      return if @entry_embeddings.empty?

      if meta["graph_saved"] == 1
        restore_graph_from_db(meta)
      else
        rebuild_index
      end
    end

    private def restore_graph_from_db(meta : Hash(String, Int32)) : Nil
      node_count = @entry_embeddings.size
      neighbours = Array(Array(Array(Int32))).new(node_count) { [] of Array(Int32) }

      @db.query("SELECT id, layer_count FROM #{TABLE_GRAPH_NODES} ORDER BY id") do |results|
        results.each do
          node_id = results.read(Int32)
          layer_count = results.read(Int32)
          next unless node_id < node_count
          neighbours[node_id] = Array(Array(Int32)).new(layer_count) { [] of Int32 }
        end
      end

      @db.query(
        "SELECT node_id, layer, neighbour_id FROM #{TABLE_GRAPH_EDGES} ORDER BY node_id, layer"
      ) do |results|
        results.each do
          node_id = results.read(Int32)
          layer = results.read(Int32)
          neighbour_id = results.read(Int32)
          next unless node_id < node_count
          neighbours[node_id][layer] << neighbour_id
        end
      end

      @index = HNSW::Index.restore(
        dims: @embedder.dimensions,
        m: @m,
        ef_construction: @ef_construction,
        entry_point: meta["entry_point"],
        max_layer: meta["max_layer"],
        node_count: node_count
      ) do |id|
        {@entry_embeddings[id].vector, neighbours[id]}
      end
    end

    private def rebuild_index : Nil
      @index = new_index
      @entry_embeddings.each_with_index do |entry, id|
        @index.add(id: id, vector: entry.vector)
      end
    end

    private def get_entry(id : Int32) : CachedEntry(M)
      @entry_cache.get(id) { retrieve_entry(id) }
    end

    private def retrieve_entry(id : Int32) : CachedEntry(M)
      entry = nil
      @db.query(
        "SELECT id, text, meta, payload_id FROM #{TABLE_ENTRIES} WHERE id = ? AND deleted = 0",
        id
      ) do |results|
        results.each do
          entry_id = results.read(Int32)
          text = results.read(String)
          meta_json = results.read(String?)
          payload_id = results.read(Int64?)
          meta = meta_json.try { |j| M.from_json(j) }
          entry = CachedEntry.new(text, @entry_embeddings[entry_id].vector, meta, payload_id)
        end
      end
      entry || raise Error.new("Unexpected error fetching entry (id = #{id}) from DB.")
    end

    def purge_expired_from_cache
      if purge_delay = @cache_purge_period
        if (now = Time.instant) - @cache_last_purged > purge_delay
          @cache_last_purged = now
          @entry_cache.purge_expired
        end
      end
    end

    private def new_index
      HNSW::Index.new(dims: @embedder.dimensions, m: @m, ef_construction: @ef_construction)
    end

    private def read_meta : Hash(String, Int32)
      meta = {} of String => Int32
      @db.query("SELECT key, value FROM #{TABLE_META}") do |results|
        results.each { meta[results.read(String)] = results.read(Int32) }
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
