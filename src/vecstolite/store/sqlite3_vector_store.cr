require "sqlite3"

require "../vector_store"
require "../indexer/*"
require "../sucre/cache"

module Vecstolite
  # A SQLite-backed vector store with an in-memory HNSW index.
  #
  # Design:
  #   - Entries (text + vector) are written to SQLite on every `add`.
  #   - The HNSW graph lives in memory for fast queries.
  #   - Graph topology is flushed to SQLite via `save_graph` (or `close`),
  #     so a full rebuild from disk is possible at any time.
  #   - On `open`, entries are read from SQLite and the HNSW index is
  #     reconstructed — either from stored graph edges (fast) or by
  #     re-inserting all vectors (fallback if no graph is saved yet).
  #   - When opening and reading an existing DB, search results are retrieved
  #     from the DB unless already in the entry cache
  #
  # Usage:
  # ```
  # store = Vecstolite::SQLiteVectorStore.create("store.db")
  # store.add("The sky is blue.")
  # store.add("Crystal is fast.")
  # results = store.search("colour of the sky", k: 3)
  # store.close # flushes graph to SQLite
  # ```
  #
  class SQLiteVectorStore
    include IndexedVectorStore

    class Error < Exception; end

    SCHEMA_VERSION = 1

    # Open an existing SQLite-backed vector store at `path`. Optionally, open it as `readonly`, and set
    # a TTL for cache entries, and set a cache purge duration period
    def self.open(path : String,
                  embedder : VectorEmbedder,
                  readonly = false,
                  cache_ttl : Time::Span? = nil,
                  cache_purge_period : Time::Span? = nil,) : SQLiteVectorStore
      raise Error.new("Database '#{path}' does not exist.") unless File.exists?(path)

      db = DB.open("sqlite3://#{path}")
      store = new(embedder, db, path,
        readonly: readonly,
        cache_ttl: cache_ttl,
        cache_purge_period: cache_purge_period)
      store.bootstrap
      store
    end

    # Create a SQLite-backed vector store at `path`; specify `m` and `ef_construction` to
    # control the number of max neighbours per node per layer and
    # beam width when inserting a new node distibution respectively.
    def self.create(path : String,
                    embedder : VectorEmbedder,
                    m = DEFAULT_M,
                    ef_construction = DEFAULT_EF_CONSTRUCTION,
                    cache_ttl : Time::Span? = nil,
                    cache_purge_period : Time::Span? = nil) : SQLiteVectorStore
      raise Error.new("Database '#{path}' already exists.") if File.exists?(path)

      db = DB.open("sqlite3://#{path}")
      store = new(embedder, db, path, m, ef_construction,
        readonly: false,
        cache_ttl: cache_ttl,
        cache_purge_period: cache_purge_period)
      store.bootstrap
      store
    end

    # Flush and close the backing database.
    def close : Nil
      return if @closed

      save_graph unless @readonly

      @db.close
      @closed = true
    end

    def clear : Nil
      raise Error.new("Clearing SQLiteVectorStore not yet supported.")
    end

    private TABLE_GRAPH_NODES = "vecsto_graph_nodes"
    private TABLE_GRAPH_EDGES = "vecsto_graph_edges"
    private TABLE_META        = "vecsto_meta"
    private TABLE_ENTRIES     = "vecsto_entries"
    private INDEX_EDGES       = "vecsto_idx_edges"

    # Flush the in-memory HNSW graph topology to SQLite.
    # Call this explicitly if you want a durable checkpoint without closing.
    def save_graph : Nil
      idx = @index
      @db.transaction do
        @db.exec "DELETE FROM #{TABLE_GRAPH_NODES}"
        @db.exec "DELETE FROM #{TABLE_GRAPH_EDGES}"

        idx.@nodes.each_with_index do |node, node_id|
          @db.exec "INSERT INTO #{TABLE_GRAPH_NODES} VALUES (?, ?)",
            node_id, node.neighbours.size

          node.neighbours.each_with_index do |layer_neighbours, layer|
            layer_neighbours.each do |nb_id|
              @db.exec "INSERT INTO #{TABLE_GRAPH_EDGES} VALUES (?, ?, ?)",
                node_id, layer, nb_id
            end
          end
        end

        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('entry_point', ?, NULL)", idx.@entry_point
        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('max_layer',   ?, NULL)", idx.@max_layer
        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('graph_saved', ?, NULL)", 1
      end
    end

    # Returns stats `NamedTuple` with following entried:
    # - No. of `embeddings` (total)
    # - No. of `indexed_nodes` (should match `embeddings`)
    # - No. of `cached` entries (may be less when opening a DB)
    def stats : NamedTuple
      {cached: @entry_cache.size, embeddings: @entry_embeddings.size, indexed_nodes: @index.size}
    end

    # -------------------------------------------------------------------------
    # Mutations
    # -------------------------------------------------------------------------

    # Add and index `text`, with optional `extra` data (not embedded or indexed)
    def add(text : String, extra : String? = nil) : Nil
      raise Error.new("Store is closed") if @closed
      raise Error.new("Store is readonly") if @readonly

      purge_expired_from_cache # For now; ideally this happens on a schedule

      vector = @embedder.embed(text)
      id = @entry_embeddings.size

      @db.transaction do
        @db.exec "INSERT INTO #{TABLE_ENTRIES} (id, text, vector, extra) VALUES (?, ?, ?, ?)",
          id, text, pack_vector(vector), extra
      end

      @entry_cache.put(id, Entry.new(text, vector, extra))
      @entry_embeddings << EntryVector.new(vector)

      @index.add(id: id, vector: vector)
    end

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def search(query : String, k : Int32 = DEFAULT_K, ef_search : Int32 = DEFAULT_EF_SEARCH) : Array(SearchResult)
      raise Error.new("Store is closed") if @closed

      return [] of SearchResult if @entry_embeddings.empty?

      purge_expired_from_cache # For now; ideally this happens on a schedule

      query_vec = @embedder.embed(query)
      @index.search(query_vec, k: k, ef: ef_search).map do |result|
        entry = get_entry(result.id)
        SearchResult.new(entry.text, result.score, entry.extra)
      end
    end

    def size : Int32
      @entry_cache.size
    end

    private def get_entry(id) : Entry
      @entry_cache.get(id) do
        retrieve_entry(id)
      end
    end

    # Retrieve an Entry from the DB and put it into the cache
    private def retrieve_entry(id : Int32) : Entry
      entry = nil
      @db.transaction do
        @db.query("SELECT id, text, extra FROM #{TABLE_ENTRIES} WHERE id = ? AND deleted = 0 ORDER BY id", id) do |results|
          results.each do
            entry_id = results.read(Int32)
            text = results.read(String)
            extra = results.read(String)
            entry = Entry.new(text, @entry_embeddings[entry_id].vector, extra)
          end
        end
      end
      entry || raise Error.new("Unexpected error fetching entry (id = #{id}) from DB.")
    end

    # Purge expired cached entries, but only if its past the `cache_purge_period` since
    # last purge.
    def purge_expired_from_cache
      if purge_delay = @cache_purge_period
        if (now = Time.instant) - @cache_last_purged > purge_delay
          @cache_last_purged = now
          puts "purging..."
          @entry_cache.purge_expired
        end
      end
    end

    # -------------------------------------------------------------------------

    # A minimal representation of an item in a vector store: the embedding vector only; the index is the ID.
    record EntryVector, vector : Embedding

    @readonly : Bool
    @db : DB::Database
    @path : String
    @entry_embeddings : Array(EntryVector)
    @embedder : VectorEmbedder
    @index : HNSW::Index
    @closed : Bool
    @m : Int32
    @ef_construction : Int32

    @entry_cache : Cache(Int32, Entry)
    @cache_last_purged = Time.instant
    @cache_purge_period : Time::Span?

    private def initialize(@embedder : VectorEmbedder, @db : DB::Database, @path : String,
                           @m : Int32, @ef_construction : Int32, @readonly,
                           cache_ttl : Time::Span? = nil,
                           @cache_purge_period : Time::Span? = nil)
      @entry_cache = Cache(Int32, Entry).new(cache_ttl)
      @entry_embeddings = [] of EntryVector
      @index = new_index
      @closed = false
    end

    private def initialize(@embedder : VectorEmbedder, @db : DB::Database,
                           @path : String, @readonly,
                           cache_ttl : Time::Span? = nil,
                           @cache_purge_period : Time::Span? = nil)
      @entry_cache = Cache(Int32, Entry).new(cache_ttl)
      @entry_embeddings = [] of EntryVector
      @m = DEFAULT_M
      @ef_construction = DEFAULT_EF_CONSTRUCTION
      @index = new_index
      @closed = false
    end

    # Create schema if needed, then load existing data.
    protected def bootstrap : Nil
      unless @readonly
        # WAL journaling and FULL sync are enabled to ensure durability,
        # especially since we write the index at the end.
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
      CREATE TABLE IF NOT EXISTS #{TABLE_ENTRIES} (
        id        INTEGER PRIMARY KEY,
        text      TEXT    NOT NULL,
        vector    BLOB    NOT NULL,
        extra     TEXT    DEFAULT NULL,
        deleted   INTEGER NOT NULL DEFAULT 0
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

        # Write schema version if this is a fresh database.
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('schema_version', ?, NULL)", SCHEMA_VERSION
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('m',              ?, NULL)", @m
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('ef_construction',?, NULL)", @ef_construction
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('graph_saved',    ?, NULL)", 0
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('entry_point',    ?, NULL)", -1
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('max_layer',      ?, NULL)", -1
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('dimensions',      ?, NULL)", @embedder.dimensions
        @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('embedder',      1, ?)", @embedder.model_name
      end

      # Read back params (may differ from constructor args if db already existed).
      meta = read_meta
      @m = meta["m"]
      @ef_construction = meta["ef_construction"]
      @index = new_index

      load_from_db(meta)
    end

    # Restore entries and HNSW index from the database.
    private def load_from_db(meta : Hash(String, Int32)) : Nil
      # Load non-deleted entry vectors in ID order.
      @db.query("SELECT vector FROM #{TABLE_ENTRIES} WHERE deleted = 0 ORDER BY id") do |results|
        results.each do
          blob = results.read(Bytes)
          vector = unpack_vector(blob)
          @entry_embeddings << EntryVector.new(vector)
        end
      end
      return if @entry_embeddings.empty?

      graph_saved = meta["graph_saved"] == 1

      if graph_saved
        restore_graph_from_db(meta)
      else
        rebuild_index
      end
    end

    # Fast path: wire the saved graph topology directly without re-inserting.
    private def restore_graph_from_db(meta : Hash(String, Int32)) : Nil
      ix_nodes = [] of HNSW::HNSWNode

      # Create node shells.
      @entry_embeddings.each do |entry|
        node = HNSW::HNSWNode.new(entry.vector, 0, @m)
        ix_nodes << node
      end

      # Restore neighbour lists.
      @db.query("SELECT id, layer_count FROM #{TABLE_GRAPH_NODES} ORDER BY id") do |results|
        results.each do
          node_id = results.read(Int32)
          layer_count = results.read(Int32)
          next unless node_id < ix_nodes.size
          ix_nodes[node_id].neighbours =
            Array(Array(Int32)).new(layer_count) { [] of Int32 }
        end
      end

      @db.query(
        "SELECT node_id, layer, neighbour_id FROM #{TABLE_GRAPH_EDGES} ORDER BY node_id, layer"
      ) do |results|
        results.each do
          node_id = results.read(Int32)
          layer = results.read(Int32)
          neighbour_id = results.read(Int32)
          next unless node_id < ix_nodes.size
          ix_nodes[node_id].neighbours[layer] << neighbour_id
        end
      end

      # HACK alert - reaches into HNSW::Index to reset.
      # ONE DAY do something better
      @index.reset_with(ix_nodes, meta["entry_point"], meta["max_layer"])
    end

    # Slow path: rebuild index by re-inserting all vectors.
    # Used when no graph has been saved yet, or after a deletion.
    private def rebuild_index : Nil
      @index = new_index
      @entry_embeddings.each_with_index do |entry, id|
        @index.add(id: id, vector: entry.vector)
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
