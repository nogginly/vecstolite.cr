require "sqlite3"
require "../sizes"
require "../vector_embedder"

module Vecstolite
  # Owns the SQLite schema and every SQL statement in the shard. Knows about
  # rows — entries, vectors, graph nodes, payloads and metadata — and nothing
  # about graph traversal, embedding or caching.
  #
  # Two id spaces meet here and must not be confused:
  #
  # - `entry_id` is stable, assigned by SQLite, and survives compaction.
  # - `ord` is the graph position an index strategy assigns, contiguous over
  #   `0...node_count`, and rewritten wholesale by compaction.
  #
  # The mapping between them lives in `vecsto_nodes` and nowhere else.
  #
  # ```
  # repo = Vecstolite::Repository.open("notes.db", dimensions: 768)
  # repo.transaction do
  #   id = repo.insert_entry("The sky is blue.", vector)
  #   repo.insert_node(ord: 0, entry_id: id, neighbours: [[] of Int32])
  # end
  # repo.close
  # ```
  class Repository
    class Error < Exception; end

    SCHEMA_VERSION = 4

    TABLE_META     = "vecsto_meta"
    TABLE_PAYLOADS = "vecsto_payloads"
    TABLE_ENTRIES  = "vecsto_entries"
    TABLE_VECTORS  = "vecsto_vectors"
    TABLE_NODES    = "vecsto_nodes"

    DEFAULT_PAGE_CACHE_BYTES = 2_i64 * MB

    # How vectors are encoded in `vecsto_vectors`. Recorded in metadata so a
    # future encoding can be introduced without a schema migration.
    enum Encoding
      F32

      def self.from_meta(name : String) : Encoding
        parse?(name) || raise Error.new("Unknown vector encoding '#{name}'.")
      end

      def to_meta : String
        to_s.downcase
      end
    end

    # A row of `vecsto_entries`. `text`, `meta` and `payload_id` are `nil` on a
    # tombstoned row: they are released at delete time, leaving only the
    # vector as a routing waypoint until compaction.
    record EntryRow,
      id : Int64,
      key : String?,
      text : String?,
      meta : String?,
      payload_id : Int64?,
      deleted : Bool

    # A graph node with the vector it routes on, read in one join.
    record NodeRow,
      ord : Int32,
      entry_id : Int64,
      vector : Embedding,
      neighbours : Array(Array(Int32))

    getter dimensions : Int32
    getter encoding : Encoding
    getter? readonly : Bool
    getter? closed : Bool

    @db : DB::Connection

    # Opens *path* (or `":memory:"`), creating and initialising the schema when
    # absent. Raises `Error` if the file is missing and *create_if_missing* is
    # false, or if an existing database was written by an older schema.
    #
    # *dimensions* is checked against the stored value on an existing database.
    # *page_cache_bytes* sizes SQLite's own per-connection page cache, which is
    # separate from any node cache the caller maintains.
    def self.open(path : String,
                  dimensions : Int32,
                  readonly : Bool = false,
                  create_if_missing : Bool = true,
                  page_cache_bytes : Int64 = DEFAULT_PAGE_CACHE_BYTES,
                  encoding : Encoding = Encoding::F32) : self
      in_memory = path == ":memory:"
      exists = in_memory || File.exists?(path)

      unless exists
        raise Error.new("Database '#{path}' does not exist.") unless create_if_missing
        raise Error.new("Cannot create '#{path}' in readonly mode.") if readonly
      end

      new(path, dimensions, readonly, page_cache_bytes, encoding)
    end

    protected def initialize(path : String,
                             dimensions : Int32,
                             @readonly : Bool,
                             page_cache_bytes : Int64,
                             encoding : Encoding)
      # DB.connect rather than DB.open: SQLite is single-writer, so a pool buys
      # nothing, and every statement inside a transaction must share one
      # connection.
      @db = DB.connect(connection_uri(path))
      @closed = false
      @dimensions = dimensions
      @encoding = encoding
      begin
        configure(page_cache_bytes)
        create_schema unless @readonly
        verify_schema
      rescue ex
        @db.close
        @closed = true
        raise ex
      end
    end

    # `":memory:"` must be percent-encoded, or its colons parse as a URI port.
    private def connection_uri(path : String) : String
      path == ":memory:" ? "sqlite3://%3Amemory%3A" : "sqlite3://#{path}"
    end

    def close : Nil
      return if @closed
      @db.close
      @closed = true
    end

    # Runs *block* inside one SQLite transaction, rolled back if it raises.
    def transaction(&)
      @db.transaction do
        yield
      end
    end

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def meta_int(key : String) : Int64?
      @db.query_one? "SELECT value FROM #{TABLE_META} WHERE key = ?", key, as: Int64
    end

    def meta_text(key : String) : String?
      @db.query_one? "SELECT text FROM #{TABLE_META} WHERE key = ?", key, as: String?
    end

    def set_meta(key : String, value : Int, text : String? = nil) : Nil
      @db.exec "INSERT INTO #{TABLE_META} (key, value, text) VALUES (?, ?, ?) " \
               "ON CONFLICT(key) DO UPDATE SET value = excluded.value, text = excluded.text",
        key, value.to_i64, text
    end

    # Writes the graph's entry point, height and saved flag together. Callers
    # invoke this inside the same transaction as the rows those values
    # describe, so a crash cannot leave metadata describing a graph that was
    # never committed.
    def set_graph_meta(entry_point : Int32, max_layer : Int32, graph_saved : Bool) : Nil
      set_meta("entry_point", entry_point)
      set_meta("max_layer", max_layer)
      set_meta("graph_saved", graph_saved ? 1 : 0)
    end

    def graph_meta : {entry_point: Int32, max_layer: Int32, graph_saved: Bool}
      {
        entry_point: (meta_int("entry_point") || -1_i64).to_i32,
        max_layer:   (meta_int("max_layer") || -1_i64).to_i32,
        graph_saved: (meta_int("graph_saved") || 0_i64) == 1,
      }
    end

    def embedder_name : String?
      meta_text("embedder")
    end

    def embedder_name=(name : String) : Nil
      set_meta("embedder", 1, name)
    end

    # -------------------------------------------------------------------------
    # Entries and vectors
    # -------------------------------------------------------------------------

    # Inserts an entry and its vector, returning the new stable id. Raises
    # `Error` if *key* is already taken or *vector* has the wrong dimension.
    def insert_entry(text : String,
                     vector : Embedding,
                     meta : String? = nil,
                     payload_id : Int64? = nil,
                     key : String? = nil) : Int64
      check_dimensions(vector)
      check_key_available(key)
      result = @db.exec "INSERT INTO #{TABLE_ENTRIES} (key, text, meta, payload_id) VALUES (?, ?, ?, ?)",
        key, text, meta, payload_id
      id = result.last_insert_id
      @db.exec "INSERT INTO #{TABLE_VECTORS} (entry_id, vector) VALUES (?, ?)", id, pack_vector(vector)
      bump_live_count(1)
      id
    end

    def entry(id : Int64) : EntryRow?
      result_row = @db.query_one? "SELECT id, key, text, meta, payload_id, deleted FROM #{TABLE_ENTRIES} WHERE id = ?",
        id, as: {Int64, String?, String?, String?, Int64?, Int64}
      result_row.try { |row| to_entry_row(row) }
    end

    def entry_by_key(key : String) : EntryRow?
      result_row = @db.query_one? "SELECT id, key, text, meta, payload_id, deleted FROM #{TABLE_ENTRIES} WHERE key = ?",
        key, as: {Int64, String?, String?, String?, Int64?, Int64}
      result_row.try { |row| to_entry_row(row) }
    end

    # Resolves several graph positions to their entries in one query, for
    # search result assembly.
    def entries_by_ords(ords : Array(Int32)) : Hash(Int32, EntryRow)
      found = {} of Int32 => EntryRow
      return found if ords.empty?

      placeholders = Array.new(ords.size, "?").join(", ")
      @db.query(
        <<-SQL,
          SELECT n.ord, e.id, e.key, e.text, e.meta, e.payload_id, e.deleted
          FROM #{TABLE_NODES} n JOIN #{TABLE_ENTRIES} e ON e.id = n.entry_id
          WHERE n.ord IN (#{placeholders})
          SQL


        args: ords.map(&.as(DB::Any))
      ) do |result_set|
        result_set.each do
          ord = result_set.read(Int64).to_i32
          found[ord] = to_entry_row({
            result_set.read(Int64), result_set.read(String?), result_set.read(String?),
            result_set.read(String?), result_set.read(Int64?), result_set.read(Int64),
          })
        end
      end
      found
    end

    # Yields every live entry in stable id order, for rebuilds and compaction.
    def each_live_entry(& : EntryRow ->) : Nil
      @db.query(
        "SELECT id, key, text, meta, payload_id, deleted FROM #{TABLE_ENTRIES} " \
        "WHERE deleted = 0 ORDER BY id"
      ) do |result_set|
        result_set.each do
          yield to_entry_row({
            result_set.read(Int64), result_set.read(String?), result_set.read(String?),
            result_set.read(String?), result_set.read(Int64?), result_set.read(Int64),
          })
        end
      end
    end

    def vector(entry_id : Int64) : Embedding?
      blob = @db.query_one? "SELECT vector FROM #{TABLE_VECTORS} WHERE entry_id = ?", entry_id, as: Bytes
      blob.try { |b| unpack_vector(b) }
    end

    # Yields every live entry's id and vector, for exact scans and rebuilds.
    def each_live_vector(& : Int64, Embedding ->) : Nil
      @db.query(
        <<-SQL
          SELECT v.entry_id, v.vector
          FROM #{TABLE_VECTORS} v JOIN #{TABLE_ENTRIES} e ON e.id = v.entry_id
          WHERE e.deleted = 0
          ORDER BY v.entry_id
          SQL
      ) do |result_set|
        result_set.each do
          yield result_set.read(Int64), unpack_vector(result_set.read(Bytes))
        end
      end
    end

    # Marks an entry deleted and releases its text, metadata and payload link.
    # The vector and graph node remain until compaction, because the node is
    # still a routing waypoint for live neighbours. Returns false if the entry
    # was absent or already tombstoned.
    def tombstone(id : Int64) : Bool
      result = @db.exec "UPDATE #{TABLE_ENTRIES} SET deleted = 1, text = NULL, meta = NULL, " \
                        "payload_id = NULL WHERE id = ? AND deleted = 0", id
      return false if result.rows_affected == 0

      bump_live_count(-1)
      true
    end

    # Tombstones every live entry referencing *payload_id*, returning how many.
    def tombstone_by_payload(payload_id : Int64) : Int32
      result = @db.exec "UPDATE #{TABLE_ENTRIES} SET deleted = 1, text = NULL, meta = NULL, " \
                        "payload_id = NULL WHERE payload_id = ? AND deleted = 0", payload_id
      count = result.rows_affected.to_i32
      bump_live_count(-count) if count > 0
      count
    end

    # Removes tombstoned rows and their vectors, returning how many went.
    # Callers rebuild the graph in the same transaction: every `ord` is invalid
    # afterwards.
    def purge_tombstoned : Int32
      @db.exec "DELETE FROM #{TABLE_VECTORS} WHERE entry_id IN " \
               "(SELECT id FROM #{TABLE_ENTRIES} WHERE deleted = 1)"
      @db.exec("DELETE FROM #{TABLE_ENTRIES} WHERE deleted = 1").rows_affected.to_i32
    end

    def live_count : Int32
      (meta_int("live_count") || 0_i64).to_i32
    end

    def entry_count : Int32
      @db.scalar("SELECT COUNT(*) FROM #{TABLE_ENTRIES}").as(Int64).to_i32
    end

    # -------------------------------------------------------------------------
    # Graph nodes
    # -------------------------------------------------------------------------

    # Reads a node's vector and neighbours in one join. No `deleted` filter:
    # traversal must reach tombstoned nodes, which remain wired in as waypoints.
    # Result filtering happens above this layer.
    def node(ord : Int32) : NodeRow?
      @db.query(
        <<-SQL,
          SELECT n.entry_id, v.vector, n.neighbours
          FROM #{TABLE_NODES} n JOIN #{TABLE_VECTORS} v ON v.entry_id = n.entry_id
          WHERE n.ord = ?
          SQL
        ord
      ) do |result_set|
        result_set.each do
          return NodeRow.new(
            ord: ord,
            entry_id: result_set.read(Int64),
            vector: unpack_vector(result_set.read(Bytes)),
            neighbours: unpack_neighbours(result_set.read(Bytes))
          )
        end
      end
      nil
    end

    def insert_node(ord : Int32, entry_id : Int64, neighbours : Array(Array(Int32))) : Nil
      @db.exec "INSERT INTO #{TABLE_NODES} (ord, entry_id, neighbours) VALUES (?, ?, ?)",
        ord, entry_id, pack_neighbours(neighbours)
    end

    def update_neighbours(ord : Int32, neighbours : Array(Array(Int32))) : Nil
      @db.exec "UPDATE #{TABLE_NODES} SET neighbours = ? WHERE ord = ?",
        pack_neighbours(neighbours), ord
    end

    def each_node(& : NodeRow ->) : Nil
      @db.query(
        <<-SQL
          SELECT n.ord, n.entry_id, v.vector, n.neighbours
          FROM #{TABLE_NODES} n JOIN #{TABLE_VECTORS} v ON v.entry_id = n.entry_id
          ORDER BY n.ord
          SQL
      ) do |result_set|
        result_set.each do
          yield NodeRow.new(
            ord: result_set.read(Int64).to_i32,
            entry_id: result_set.read(Int64),
            vector: unpack_vector(result_set.read(Bytes)),
            neighbours: unpack_neighbours(result_set.read(Bytes))
          )
        end
      end
    end

    def node_count : Int32
      @db.scalar("SELECT COUNT(*) FROM #{TABLE_NODES}").as(Int64).to_i32
    end

    def entry_id_for(ord : Int32) : Int64?
      @db.query_one? "SELECT entry_id FROM #{TABLE_NODES} WHERE ord = ?", ord, as: Int64
    end

    def ord_for(entry_id : Int64) : Int32?
      @db.query_one?("SELECT ord FROM #{TABLE_NODES} WHERE entry_id = ?", entry_id, as: Int64).try(&.to_i32)
    end

    def clear_nodes : Nil
      @db.exec "DELETE FROM #{TABLE_NODES}"
    end

    # -------------------------------------------------------------------------
    # Payloads
    # -------------------------------------------------------------------------

    def insert_payload(content : String) : Int64
      @db.exec("INSERT INTO #{TABLE_PAYLOADS} (content) VALUES (?)", content).last_insert_id
    end

    def payload(id : Int64) : String?
      @db.query_one? "SELECT content FROM #{TABLE_PAYLOADS} WHERE id = ?", id, as: String
    end

    def update_payload(id : Int64, content : String) : Bool
      @db.exec("UPDATE #{TABLE_PAYLOADS} SET content = ? WHERE id = ?", content, id).rows_affected > 0
    end

    # Deletes the payload row itself. Raises `Error` while any entry still
    # references it: tombstone those first, in the same transaction.
    def delete_payload(id : Int64) : Bool
      referencing = @db.scalar(
        "SELECT COUNT(*) FROM #{TABLE_ENTRIES} WHERE payload_id = ?", id
      ).as(Int64)
      if referencing > 0
        raise Error.new("Payload #{id} is still referenced by #{referencing} entries.")
      end

      @db.exec("DELETE FROM #{TABLE_PAYLOADS} WHERE id = ?", id).rows_affected > 0
    end

    def payload_count : Int32
      @db.scalar("SELECT COUNT(*) FROM #{TABLE_PAYLOADS}").as(Int64).to_i32
    end

    # -------------------------------------------------------------------------
    private def configure(page_cache_bytes : Int64) : Nil
      # Negative cache_size is a KiB budget rather than a page count.
      @db.exec "PRAGMA cache_size = -#{(page_cache_bytes // KB).clamp(64_i64, Int32::MAX.to_i64)}"
      return if @readonly

      @db.exec "PRAGMA journal_mode = WAL"
      @db.exec "PRAGMA synchronous  = FULL"
      @db.exec "PRAGMA foreign_keys = ON"
    end

    private def create_schema : Nil
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
          id         INTEGER PRIMARY KEY AUTOINCREMENT,
          key        TEXT    DEFAULT NULL UNIQUE,
          text       TEXT    DEFAULT NULL,
          meta       TEXT    DEFAULT NULL,
          payload_id INTEGER DEFAULT NULL REFERENCES #{TABLE_PAYLOADS}(id),
          deleted    INTEGER NOT NULL DEFAULT 0
        )
        SQL

      @db.exec <<-SQL
        CREATE TABLE IF NOT EXISTS #{TABLE_VECTORS} (
          entry_id INTEGER PRIMARY KEY REFERENCES #{TABLE_ENTRIES}(id),
          vector   BLOB    NOT NULL
        )
        SQL

      @db.exec <<-SQL
        CREATE TABLE IF NOT EXISTS #{TABLE_NODES} (
          ord        INTEGER PRIMARY KEY,
          entry_id   INTEGER NOT NULL UNIQUE REFERENCES #{TABLE_ENTRIES}(id),
          neighbours BLOB    NOT NULL
        )
        SQL

      @db.exec "CREATE INDEX IF NOT EXISTS idx_entries_payload " \
               "ON #{TABLE_ENTRIES}(payload_id)"
      @db.exec "CREATE INDEX IF NOT EXISTS idx_entries_live " \
               "ON #{TABLE_ENTRIES}(id) WHERE deleted = 0"

      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('schema_version', ?, NULL)", SCHEMA_VERSION
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('dimensions',     ?, NULL)", @dimensions
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('graph_saved',    0, NULL)"
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('entry_point',   -1, NULL)"
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('max_layer',     -1, NULL)"
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('live_count',     0, NULL)"
      @db.exec "INSERT OR IGNORE INTO #{TABLE_META} VALUES ('encoding',       1, ?)", @encoding.to_meta
    end

    private def verify_schema : Nil
      version = (meta_int("schema_version") || 0_i64).to_i32
      if version != SCHEMA_VERSION
        raise Error.new(
          "Database schema version #{version} is not supported (current: " \
          "#{SCHEMA_VERSION}). Please recreate the database (re-run your " \
          "ingest pipeline)."
        )
      end

      stored_dims = (meta_int("dimensions") || 0_i64).to_i32
      if stored_dims != @dimensions
        raise Error.new(
          "Database was written with #{stored_dims}-dimension vectors but the " \
          "embedder produces #{@dimensions}. Opening it would read garbage."
        )
      end

      if stored = meta_text("encoding")
        @encoding = Encoding.from_meta(stored)
      end
    end

    private def to_entry_row(row : {Int64, String?, String?, String?, Int64?, Int64}) : EntryRow
      EntryRow.new(
        id: row[0], key: row[1], text: row[2],
        meta: row[3], payload_id: row[4], deleted: row[5] == 1
      )
    end

    private def bump_live_count(delta : Int32) : Nil
      @db.exec "UPDATE #{TABLE_META} SET value = value + ? WHERE key = 'live_count'", delta
    end

    private def check_key_available(key : String?) : Nil
      return if key.nil?
      return if entry_by_key(key).nil?
      raise Error.new("Entry key '#{key}' is already in use.")
    end

    private def check_dimensions(vector : Embedding) : Nil
      return if vector.size == @dimensions
      raise Error.new("Vector has #{vector.size} dimensions, expected #{@dimensions}.")
    end

    private def pack_vector(vector : Embedding) : Bytes
      case @encoding
      in Encoding::F32
        buf = Bytes.new(vector.size * 4)
        vector.each_with_index do |value, i|
          IO::ByteFormat::LittleEndian.encode(value, buf[i * 4, 4])
        end
        buf
      end
    end

    private def unpack_vector(blob : Bytes) : Embedding
      case @encoding
      in Encoding::F32
        count = blob.size // 4
        Embedding.new(count) do |i|
          IO::ByteFormat::LittleEndian.decode(Float32, blob[i * 4, 4])
        end
      end
    end

    # Neighbour wire format, little-endian Int32 throughout:
    #   [layer_count] [count₀ id₀ id₁ …] [count₁ id₀ id₁ …] …
    private def pack_neighbours(neighbours : Array(Array(Int32))) : Bytes
      total = 1 + neighbours.sum { |layer| 1 + layer.size }
      buf = Bytes.new(total * 4)
      off = 0
      IO::ByteFormat::LittleEndian.encode(neighbours.size.to_i32, buf[off, 4]); off += 4
      neighbours.each do |layer|
        IO::ByteFormat::LittleEndian.encode(layer.size.to_i32, buf[off, 4]); off += 4
        layer.each do |nbor|
          IO::ByteFormat::LittleEndian.encode(nbor, buf[off, 4]); off += 4
        end
      end
      buf
    end

    private def unpack_neighbours(blob : Bytes) : Array(Array(Int32))
      off = 0
      layer_count = IO::ByteFormat::LittleEndian.decode(Int32, blob[off, 4]); off += 4
      Array(Array(Int32)).new(layer_count) do
        count = IO::ByteFormat::LittleEndian.decode(Int32, blob[off, 4]); off += 4
        Array(Int32).new(count) do
          nb = IO::ByteFormat::LittleEndian.decode(Int32, blob[off, 4]); off += 4
          nb
        end
      end
    end
  end
end
