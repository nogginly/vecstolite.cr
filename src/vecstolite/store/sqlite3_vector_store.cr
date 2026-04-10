require "sqlite3"

require "../vector_store"
require "../indexer/*"

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
#
# Usage:
#   store = Vecstolite::SQLiteVectorStore.open("store.db")
#   store.add("The sky is blue.")
#   store.add("Crystal is fast.")
#   results = store.search("colour of the sky", k: 3)
#   store.close   # flushes graph to SQLite
#
module Vecstolite
  class SQLiteVectorStore
    include VectorStore

    class Error < Exception; end

    SCHEMA_VERSION = 1

    # -------------------------------------------------------------------------
    # Open / close
    # -------------------------------------------------------------------------

    # Open (or create) a store at *path*.
    # Pass *m* and *ef_construction* only when creating a new store;
    # they are ignored if the database already exists.
    def self.open(
      path : String,
      embedder : VectorEmbedder,
      m : Int32 = 16,
      ef_construction : Int32 = 200,
    ) : SQLiteVectorStore
      db = DB.open("sqlite3://#{path}")
      store = new(embedder, db, path, m, ef_construction)
      store.bootstrap
      store
    end

    def close : Nil
      return if @closed
      save_graph
      @db.close
      @closed = true
    end

    def clear : Nil
      raise Error.new("Clearing SQLiteVectoreStore not yet supported.")
    end

    TABLE_GRAPH_NODES = "vecsto_graph_nodes"
    TABLE_GRAPH_EDGES = "vecsto_graph_edges"
    TABLE_META        = "vecsto_meta"
    TABLE_ENTRIES     = "vecsto_entries"
    INDEX_EDGES       = "vecsto_idx_edges"

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

        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('entry_point', ?)", idx.@entry_point
        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('max_layer',   ?)", idx.@max_layer
        @db.exec "INSERT OR REPLACE INTO #{TABLE_META} VALUES ('graph_saved', ?)", 1
      end
    end

    # -------------------------------------------------------------------------
    # Mutations
    # -------------------------------------------------------------------------

    def add(text : String) : Nil
      raise Error.new("Store is closed") if @closed

      vector = @embedder.embed(text)
      id = @entries.size

      @db.transaction do
        @db.exec "INSERT INTO #{TABLE_ENTRIES} (id, text, vector) VALUES (?, ?, ?)",
          id, text, pack_vector(vector)
      end

      @entries << Entry.new(text, vector)
      @index.add(id: id, vector: vector)
    end

    def add_all(texts : Enumerable(String)) : Nil
      texts.each { |text| add(text) }
    end

    # Delete an entry by id.  Marks the row deleted in SQLite and rebuilds the
    # in-memory index from remaining entries (HNSW does not support deletion).
    def delete(id : Int32) : Nil
      raise Error.new("Store is closed") if @closed
      raise ArgumentError.new("id #{id} out of range") unless id < @entries.size

      @db.exec "UPDATE #{TABLE_ENTRIES} SET deleted = 1 WHERE id = ?", id
      rebuild_index
    end

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def search(query : String, k : Int32 = 5, ef : Int32 = 50) : Array(SearchResult)
      raise Error.new("Store is closed") if @closed
      return [] of SearchResult if @entries.empty?

      query_vec = @embedder.embed(query)
      @index.search(query_vec, k: k, ef: ef).map do |result|
        SearchResult.new(@entries[result.id].text, result.score)
      end
    end

    def size : Int32
      @entries.size
    end

    # -------------------------------------------------------------------------

    @db : DB::Database
    @path : String
    @entries : Array(Entry)
    @embedder : VectorEmbedder
    @index : HNSW::Index
    @closed : Bool
    @m : Int32
    @ef_construction : Int32

    private def initialize(
      @embedder : VectorEmbedder,
      @db : DB::Database,
      @path : String,
      @m : Int32,
      @ef_construction : Int32,
    )
      @entries = [] of Entry
      @index = new_index
      @closed = false
    end

    # Create schema if needed, then load existing data.
    protected def bootstrap : Nil
      # WAL journaling and FULL sync are enabled to ensure durability,
      # especially since we write the index at the end.
      @db.exec "PRAGMA journal_mode = WAL"
      @db.exec "PRAGMA synchronous  = FULL"
      @db.exec "PRAGMA foreign_keys = ON"

      @db.exec <<-SQL
      CREATE TABLE IF NOT EXISTS #{TABLE_META} (
        key   TEXT    PRIMARY KEY,
        value INTEGER NOT NULL
      )
      SQL

      @db.exec <<-SQL
      CREATE TABLE IF NOT EXISTS #{TABLE_ENTRIES} (
        id      INTEGER PRIMARY KEY,
        text    TEXT    NOT NULL,
        vector  BLOB    NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0
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
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('schema_version', ?)", SCHEMA_VERSION
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('m',              ?)", @m
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('ef_construction',?)", @ef_construction
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('graph_saved',    ?)", 0
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('entry_point',    ?)", -1
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('max_layer',      ?)", -1

      # Read back params (may differ from constructor args if db already existed).
      meta = read_meta
      @m = meta["m"]
      @ef_construction = meta["ef_construction"]
      @index = new_index

      load_from_db(meta)
    end

    # Restore entries and HNSW index from the database.
    private def load_from_db(meta : Hash(String, Int32)) : Nil
      # Load non-deleted entries in ID order.
      @db.query("SELECT id, text, vector FROM #{TABLE_ENTRIES} WHERE deleted = 0 ORDER BY id") do |results|
        results.each do
          _id = results.read(Int32)
          text = results.read(String)
          blob = results.read(Bytes)
          vector = unpack_vector(blob)
          @entries << Entry.new(text, vector)
        end
      end

      return if @entries.empty?

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
      @entries.each do |entry|
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
      @entries.each_with_index do |entry, id|
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
