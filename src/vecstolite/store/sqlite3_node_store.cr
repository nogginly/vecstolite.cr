require "sqlite3"
require "../indexer/hnsw_node_store"

module Vecstolite
  # :nodoc:
  # Packs/unpacks Array(Array(Int32)) neighbour lists to/from a compact BLOB.
  #
  # Wire format (little-endian Int32 throughout):
  #   [layer_count] [count₀ id₀ id₁ …] [count₁ id₀ id₁ …] …
  module NodePacking
    def self.pack(neighbours : Array(Array(Int32))) : Bytes
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

    def self.unpack(blob : Bytes) : Array(Array(Int32))
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

  # :nodoc:
  # SQLite-backed NodeStore with an LRU memory budget.
  #
  # Nodes are always written through to the `vecsto_nodes` table on `append`
  # and `write_back`, so the DB is always consistent. The in-memory LRU cache
  # avoids redundant reads for hot nodes during traversal.
  #
  # The cache uses a doubly-linked list (O(1) promote/evict) backed by a Hash
  # (O(1) lookup). Memory is tracked per node and capped at `max_bytes`.
  class LRUNodeStore
    include HNSW::NodeStore

    # A single entry in the LRU doubly-linked list.
    private class Entry
      property id : Int32
      property node : HNSW::HNSWNode
      property bytes : Int64
      property prev : Entry? # towards tail (LRU end)
      property succ : Entry? # towards head (MRU end)

      def initialize(@id, @node, @bytes)
      end
    end

    @cache = {} of Int32 => Entry
    @head : Entry? = nil # most recently used
    @tail : Entry? = nil # least recently used
    @current_bytes = 0_i64
    @total = 0
    @hits = 0_i64
    @misses = 0_i64
    @evictions = 0_i64

    protected def initialize(@db : DB::Database,
                             @dims : Int32,
                             @table_entries : String,
                             @table_nodes : String,
                             @max_bytes : Int64)
    end

    def get(id : Int32) : HNSW::HNSWNode
      if entry = @cache[id]?
        move_to_head(entry)
        @hits += 1
        entry.node
      else
        @misses += 1
        node = fetch_from_db(id)
        insert_into_cache(id, node)
        node
      end
    end

    def append(node : HNSW::HNSWNode) : Nil
      id = @total
      @total += 1
      write_node_to_db(id, node)
      insert_into_cache(id, node)
    end

    def write_back(id : Int32, node : HNSW::HNSWNode) : Nil
      write_neighbours_to_db(id, node)
      # Update cached byte count if neighbour lists changed size.
      if entry = @cache[id]?
        new_bytes = node_bytes(node)
        @current_bytes += new_bytes - entry.bytes
        entry.bytes = new_bytes
      end
    end

    def size : Int32
      @total
    end

    def truncate_to(count : Int32) : Nil
      # Evict any cached nodes that are being rolled back.
      @cache.each_key.to_a.each do |id|
        next if id < count
        if entry = @cache.delete(id)
          remove_from_list(entry)
          @current_bytes -= entry.bytes
        end
      end
      @total = count
    end

    def restore_from_persisted(count : Int32, & : Int32 -> {Embedding, Array(Array(Int32))}) : Nil
      @total = count # nodes already in DB; start with empty cache
    end

    def each_node(& : Int32, HNSW::HNSWNode ->) : Nil
      @total.times { |id| yield id, get(id) }
    end

    def fully_persisted? : Bool
      true
    end

    def cache_mode : Symbol
      :lru
    end

    def cache_size : Int32
      @cache.size
    end

    def cache_hits : Int64
      @hits
    end

    def cache_misses : Int64
      @misses
    end

    def cache_evictions : Int64
      @evictions
    end

    # -------------------------------------------------------------------------
    private def insert_into_cache(id : Int32, node : HNSW::HNSWNode) : Nil
      bytes = node_bytes(node)
      # Evict from tail until we have room (or cache is empty).
      while @current_bytes + bytes > @max_bytes && @tail
        evict_tail
      end
      entry = Entry.new(id, node, bytes)
      @cache[id] = entry
      @current_bytes += bytes
      prepend_to_head(entry)
    end

    private def evict_tail : Nil
      tail = @tail
      return unless tail
      remove_from_list(tail)
      @cache.delete(tail.id)
      @current_bytes -= tail.bytes
      @evictions += 1
    end

    private def move_to_head(entry : Entry) : Nil
      return if entry.same?(@head)
      remove_from_list(entry)
      prepend_to_head(entry)
    end

    private def prepend_to_head(entry : Entry) : Nil
      entry.succ = @head
      entry.prev = nil
      if h = @head
        h.prev = entry
      end
      @head = entry
      @tail ||= entry
    end

    private def remove_from_list(entry : Entry) : Nil
      if p = entry.prev
        p.succ = entry.succ
      else
        @head = entry.succ
      end
      if s = entry.succ
        s.prev = entry.prev
      else
        @tail = entry.prev
      end
      entry.prev = nil
      entry.succ = nil
    end

    private def node_bytes(node : HNSW::HNSWNode) : Int64
      # Vector bytes + packed neighbour bytes + per-object overhead.
      (node.vector.bytesize + node.neighbours.sum { |layer| layer.size * 4 } + 128).to_i64
    end

    private def fetch_from_db(id : Int32) : HNSW::HNSWNode
      vector : Embedding? = nil
      neighbours : Array(Array(Int32))? = nil
      @db.query(
        "SELECT e.vector, n.neighbours
         FROM #{@table_entries} e JOIN #{@table_nodes} n ON e.id = n.id
         WHERE e.id = ? AND e.deleted = 0",
        id
      ) do |result_set|
        result_set.each do
          vector = unpack_vector(result_set.read(Bytes))
          neighbours = NodePacking.unpack(result_set.read(Bytes))
        end
      end
      v = vector || raise Exception.new("Node #{id} not found in database (LRUNodeStore).")
      n = neighbours || raise Exception.new("Node #{id} neighbours missing in database (LRUNodeStore).")
      node = HNSW::HNSWNode.new(v, 0, 0)
      node.neighbours = n
      node
    end

    private def write_node_to_db(id : Int32, node : HNSW::HNSWNode) : Nil
      @db.exec "INSERT INTO #{@table_nodes} (id, neighbours) VALUES (?, ?)",
        id, NodePacking.pack(node.neighbours)
    end

    private def write_neighbours_to_db(id : Int32, node : HNSW::HNSWNode) : Nil
      @db.exec "UPDATE #{@table_nodes} SET neighbours = ? WHERE id = ?",
        NodePacking.pack(node.neighbours), id
    end

    private def unpack_vector(blob : Bytes) : Embedding
      Embedding.new(@dims) do |i|
        IO::ByteFormat::LittleEndian.decode(Float32, blob[i * 4, 4])
      end
    end
  end

  # :nodoc:
  # SQLite-backed NodeStore with no in-memory cache.
  #
  # Every `get` reads from the database. Every `append` and `write_back`
  # writes to the database. Minimal memory footprint; slowest traversal.
  #
  # Recommended for: memory-constrained environments, infrequent one-off
  # searches against an already-built index.
  # Not recommended for: bulk ingest (use LRU mode instead).
  class DiskNodeStore
    include HNSW::NodeStore

    protected def initialize(@db : DB::Database,
                             @dims : Int32,
                             @table_entries : String,
                             @table_nodes : String)
      @total = 0
      @misses = 0_i64
    end

    def get(id : Int32) : HNSW::HNSWNode
      @misses += 1
      fetch_from_db(id)
    end

    def append(node : HNSW::HNSWNode) : Nil
      @db.exec "INSERT INTO #{@table_nodes} (id, neighbours) VALUES (?, ?)",
        @total, NodePacking.pack(node.neighbours)
      @total += 1
    end

    def write_back(id : Int32, node : HNSW::HNSWNode) : Nil
      @db.exec "UPDATE #{@table_nodes} SET neighbours = ? WHERE id = ?",
        NodePacking.pack(node.neighbours), id
    end

    def size : Int32
      @total
    end

    def truncate_to(count : Int32) : Nil
      @total = count
    end

    def restore_from_persisted(count : Int32, & : Int32 -> {Embedding, Array(Array(Int32))}) : Nil
      @total = count
    end

    def each_node(& : Int32, HNSW::HNSWNode ->) : Nil
      @total.times { |id| yield id, get(id) }
    end

    def fully_persisted? : Bool
      true
    end

    def cache_mode : Symbol
      :disk
    end

    def cache_size : Int32
      0
    end

    def cache_hits : Int64
      0_i64
    end

    def cache_misses : Int64
      @misses
    end

    def cache_evictions : Int64
      0_i64
    end

    # -------------------------------------------------------------------------
    private def fetch_from_db(id : Int32) : HNSW::HNSWNode
      vector : Embedding? = nil
      neighbours : Array(Array(Int32))? = nil
      @db.query(
        "SELECT e.vector, n.neighbours
         FROM #{@table_entries} e JOIN #{@table_nodes} n ON e.id = n.id
         WHERE e.id = ? AND e.deleted = 0",
        id
      ) do |result_set|
        result_set.each do
          vector = unpack_vector(result_set.read(Bytes))
          neighbours = NodePacking.unpack(result_set.read(Bytes))
        end
      end
      v = vector || raise Exception.new("Node #{id} not found in database (DiskNodeStore).")
      n = neighbours || raise Exception.new("Node #{id} neighbours missing in database (DiskNodeStore).")
      node = HNSW::HNSWNode.new(v, 0, 0)
      node.neighbours = n
      node
    end

    private def unpack_vector(blob : Bytes) : Embedding
      Embedding.new(@dims) do |i|
        IO::ByteFormat::LittleEndian.decode(Float32, blob[i * 4, 4])
      end
    end
  end
end
