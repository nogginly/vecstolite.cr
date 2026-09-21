require "../../spec_helper"

Spectator.describe Vecstolite::Repository do
  alias Repo = Vecstolite::Repository

  let(db_file_name) { File.tempname("vecsto_repo", ".db") }
  let(dims) { 4 }

  after_each { File.delete?(db_file_name) }

  private def vec(*values : Float32) : Vecstolite::Embedding
    floats = values.to_a
    Vecstolite::Embedding.new(floats.size) { |i| floats[i] }
  end

  # ---------------------------------------------------------------------------
  # lifecycle
  # ---------------------------------------------------------------------------

  describe ".open" do
    it "creates the database file" do
      Repo.open(db_file_name, dimensions: dims).close
      expect(File.exists?(db_file_name)).to be true
    end

    it "works in memory" do
      repo = Repo.open(":memory:", dimensions: dims)
      expect(repo.entry_count).to eq 0
      repo.close
    end

    it "fails when the file is missing and creation is not allowed" do
      expect { Repo.open(db_file_name, dimensions: dims, create_if_missing: false) }.to raise_error(Repo::Error)
    end

    it "reopens an existing database" do
      repo = Repo.open(db_file_name, dimensions: dims)
      id = repo.insert_entry("hello", vec(1_f32, 0_f32, 0_f32, 0_f32))
      repo.close

      repo = Repo.open(db_file_name, dimensions: dims)
      expect(repo.entry(id).try(&.text)).to eq "hello"
      repo.close
    end

    it "refuses a dimension mismatch" do
      Repo.open(db_file_name, dimensions: dims).close
      expect { Repo.open(db_file_name, dimensions: dims + 1) }.to raise_error(Repo::Error, /dimension/)
    end

    it "creates a database with the configured page size" do
      repo = Repo.open(db_file_name, dimensions: dims)
      expect(repo.page_size).to eq Repo::PAGE_SIZE
      repo.close
    end

    it "records the vector encoding" do
      repo = Repo.open(db_file_name, dimensions: dims)
      expect(repo.encoding).to eq Repo::Encoding::F32
      expect(repo.meta_text("encoding")).to eq "f32"
      repo.close
    end
  end

  # ---------------------------------------------------------------------------
  # entries and vectors
  # ---------------------------------------------------------------------------

  describe "entries" do
    it "round-trips an entry and its vector" do
      repo = Repo.open(":memory:", dimensions: dims)
      v = vec(0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32)
      id = repo.insert_entry("sky", v, meta: %({"code":"en"}), key: "doc-1")

      row = repo.entry(id).not_nil!
      expect(row.text).to eq "sky"
      expect(row.meta).to eq %({"code":"en"})
      expect(row.key).to eq "doc-1"
      expect(row.deleted).to be false
      expect(repo.vector(id).not_nil!.to_a).to eq v.to_a
      repo.close
    end

    it "issues ids that do not restart after deletion" do
      repo = Repo.open(":memory:", dimensions: dims)
      first = repo.insert_entry("a", vec(1_f32, 0_f32, 0_f32, 0_f32))
      repo.tombstone(first)
      repo.purge_tombstoned
      second = repo.insert_entry("b", vec(0_f32, 1_f32, 0_f32, 0_f32))

      expect(second).to be > first
      repo.close
    end

    it "finds an entry by key" do
      repo = Repo.open(":memory:", dimensions: dims)
      id = repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32), key: "doc-1")
      expect(repo.entry_by_key("doc-1").try(&.id)).to eq id
      expect(repo.entry_by_key("nope")).to be_nil
      repo.close
    end

    it "rejects a duplicate key" do
      repo = Repo.open(":memory:", dimensions: dims)
      repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32), key: "doc-1")
      expect {
        repo.insert_entry("ciel", vec(0_f32, 1_f32, 0_f32, 0_f32), key: "doc-1")
      }.to raise_error(Repo::Error, /already in use/)
      repo.close
    end

    it "rejects a wrong-dimension vector" do
      repo = Repo.open(":memory:", dimensions: dims)
      expect { repo.insert_entry("sky", vec(1_f32, 0_f32)) }.to raise_error(Repo::Error, /dimensions/)
      repo.close
    end

    it "tracks the live count across insert, tombstone and purge" do
      repo = Repo.open(":memory:", dimensions: dims)
      a = repo.insert_entry("a", vec(1_f32, 0_f32, 0_f32, 0_f32))
      repo.insert_entry("b", vec(0_f32, 1_f32, 0_f32, 0_f32))
      expect(repo.live_count).to eq 2

      expect(repo.tombstone(a)).to be true
      expect(repo.live_count).to eq 1
      expect(repo.entry_count).to eq 2

      expect(repo.tombstone(a)).to be false
      expect(repo.live_count).to eq 1

      expect(repo.purge_tombstoned).to eq 1
      expect(repo.entry_count).to eq 1
      repo.close
    end

    it "releases text and metadata on tombstone but keeps the vector" do
      repo = Repo.open(":memory:", dimensions: dims)
      id = repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32), meta: %({"code":"en"}))
      repo.tombstone(id)

      row = repo.entry(id).not_nil!
      expect(row.deleted).to be true
      expect(row.text).to be_nil
      expect(row.meta).to be_nil
      expect(repo.vector(id)).not_to be_nil
      repo.close
    end

    it "iterates live entries and vectors only" do
      repo = Repo.open(":memory:", dimensions: dims)
      a = repo.insert_entry("a", vec(1_f32, 0_f32, 0_f32, 0_f32))
      repo.insert_entry("b", vec(0_f32, 1_f32, 0_f32, 0_f32))
      repo.tombstone(a)

      texts = [] of String?
      repo.each_live_entry { |row| texts << row.text }
      expect(texts).to eq ["b"]

      ids = [] of Int64
      repo.each_live_vector { |id, _| ids << id }
      expect(ids.size).to eq 1
      repo.close
    end
  end

  # ---------------------------------------------------------------------------
  # graph nodes
  # ---------------------------------------------------------------------------

  describe "nodes" do
    it "reads a node's vector and neighbours in one call" do
      repo = Repo.open(":memory:", dimensions: dims)
      v = vec(0_f32, 0_f32, 1_f32, 0_f32)
      id = repo.insert_entry("sky", v)
      repo.insert_node(ord: 0, entry_id: id, neighbours: [[1, 2], [3]])

      node = repo.node(0).not_nil!
      expect(node.entry_id).to eq id
      expect(node.vector.to_a).to eq v.to_a
      expect(node.neighbours).to eq [[1, 2], [3]]
      expect(repo.node(7)).to be_nil
      repo.close
    end

    it "reaches a tombstoned node, since it is still a waypoint" do
      repo = Repo.open(":memory:", dimensions: dims)
      id = repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32))
      repo.insert_node(ord: 0, entry_id: id, neighbours: [[] of Int32])
      repo.tombstone(id)

      expect(repo.node(0)).not_to be_nil
      repo.close
    end

    it "maps between ord and entry id" do
      repo = Repo.open(":memory:", dimensions: dims)
      id = repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32))
      repo.insert_node(ord: 5, entry_id: id, neighbours: [[] of Int32])

      expect(repo.entry_id_for(5)).to eq id
      expect(repo.ord_for(id)).to eq 5
      expect(repo.ord_for(id + 999)).to be_nil
      repo.close
    end

    it "updates neighbours without touching the vector" do
      repo = Repo.open(":memory:", dimensions: dims)
      v = vec(1_f32, 0_f32, 0_f32, 0_f32)
      id = repo.insert_entry("sky", v)
      repo.insert_node(ord: 0, entry_id: id, neighbours: [[] of Int32])
      repo.update_neighbours(0, [[4, 5, 6]])

      node = repo.node(0).not_nil!
      expect(node.neighbours).to eq [[4, 5, 6]]
      expect(node.vector.to_a).to eq v.to_a
      repo.close
    end

    it "resolves several ords to entries in one query" do
      repo = Repo.open(":memory:", dimensions: dims)
      a = repo.insert_entry("a", vec(1_f32, 0_f32, 0_f32, 0_f32))
      b = repo.insert_entry("b", vec(0_f32, 1_f32, 0_f32, 0_f32))
      repo.insert_node(ord: 0, entry_id: a, neighbours: [[] of Int32])
      repo.insert_node(ord: 1, entry_id: b, neighbours: [[] of Int32])

      found = repo.entries_by_ords([1, 0])
      expect(found.size).to eq 2
      expect(found[0].text).to eq "a"
      expect(found[1].text).to eq "b"
      expect(repo.entries_by_ords([] of Int32)).to be_empty
      repo.close
    end

    it "purges a tombstoned entry's node with it" do
      repo = Repo.open(":memory:", dimensions: dims)
      id = repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32))
      repo.insert_node(ord: 0, entry_id: id, neighbours: [[] of Int32])
      repo.tombstone(id)

      expect(repo.purge_tombstoned).to eq 1
      expect(repo.node_count).to eq 0
      expect(repo.entry_count).to eq 0
      repo.close
    end

    it "clears the graph without losing entries" do
      repo = Repo.open(":memory:", dimensions: dims)
      id = repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32))
      repo.insert_node(ord: 0, entry_id: id, neighbours: [[] of Int32])
      repo.clear_nodes

      expect(repo.node_count).to eq 0
      expect(repo.entry_count).to eq 1
      repo.close
    end
  end

  # ---------------------------------------------------------------------------
  # payloads
  # ---------------------------------------------------------------------------

  describe "payloads" do
    it "round-trips, updates and deletes" do
      repo = Repo.open(":memory:", dimensions: dims)
      pid = repo.insert_payload(%({"en":"sky"}))
      expect(repo.payload(pid)).to eq %({"en":"sky"})

      expect(repo.update_payload(pid, %({"en":"the sky"}))).to be true
      expect(repo.payload(pid)).to eq %({"en":"the sky"})

      expect(repo.delete_payload(pid)).to be true
      expect(repo.payload(pid)).to be_nil
      expect(repo.delete_payload(pid)).to be false
      repo.close
    end

    it "refuses to delete a payload a live entry still references" do
      repo = Repo.open(db_file_name, dimensions: dims)
      pid = repo.insert_payload(%({"en":"sky"}))
      repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32), payload_id: pid)

      expect { repo.delete_payload(pid) }.to raise_error(Repo::Error, /still referenced/)
      repo.close
    end

    it "tombstones every entry of a payload, then allows its deletion" do
      repo = Repo.open(db_file_name, dimensions: dims)
      pid = repo.insert_payload(%({"en":"sky","fr":"ciel"}))
      repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32), payload_id: pid)
      repo.insert_entry("ciel", vec(0_f32, 1_f32, 0_f32, 0_f32), payload_id: pid)

      expect(repo.tombstone_by_payload(pid)).to eq 2
      expect(repo.live_count).to eq 0
      expect(repo.delete_payload(pid)).to be true
      repo.close
    end
  end

  # ---------------------------------------------------------------------------
  # metadata and transactions
  # ---------------------------------------------------------------------------

  describe "metadata" do
    it "round-trips integer and text values" do
      repo = Repo.open(":memory:", dimensions: dims)
      repo.set_meta("m", 16)
      repo.embedder_name = "lexical"

      expect(repo.meta_int("m")).to eq 16
      expect(repo.embedder_name).to eq "lexical"
      expect(repo.meta_int("nope")).to be_nil
      repo.close
    end

    it "overwrites an existing value" do
      repo = Repo.open(":memory:", dimensions: dims)
      repo.set_meta("m", 16)
      repo.set_meta("m", 8)
      expect(repo.meta_int("m")).to eq 8
      repo.close
    end

    it "writes graph metadata as a group" do
      repo = Repo.open(":memory:", dimensions: dims)
      repo.set_graph_meta(entry_point: 3, max_layer: 2, graph_saved: true)

      meta = repo.graph_meta
      expect(meta[:entry_point]).to eq 3
      expect(meta[:max_layer]).to eq 2
      expect(meta[:graph_saved]).to be true
      repo.close
    end

    it "defaults to an empty graph on a fresh database" do
      repo = Repo.open(":memory:", dimensions: dims)
      meta = repo.graph_meta
      expect(meta[:entry_point]).to eq(-1)
      expect(meta[:graph_saved]).to be false
      repo.close
    end
  end

  describe "#transaction" do
    it "commits the whole block" do
      repo = Repo.open(db_file_name, dimensions: dims)
      repo.transaction do
        id = repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32))
        repo.insert_node(ord: 0, entry_id: id, neighbours: [[] of Int32])
        repo.set_graph_meta(entry_point: 0, max_layer: 0, graph_saved: true)
      end

      expect(repo.entry_count).to eq 1
      expect(repo.node_count).to eq 1
      expect(repo.graph_meta[:graph_saved]).to be true
      repo.close
    end

    it "rolls back rows and metadata together when the block raises" do
      repo = Repo.open(db_file_name, dimensions: dims)
      expect {
        repo.transaction do
          id = repo.insert_entry("sky", vec(1_f32, 0_f32, 0_f32, 0_f32))
          repo.insert_node(ord: 0, entry_id: id, neighbours: [[] of Int32])
          repo.set_graph_meta(entry_point: 0, max_layer: 0, graph_saved: true)
          raise "boom"
        end
      }.to raise_error(/boom/)

      expect(repo.entry_count).to eq 0
      expect(repo.node_count).to eq 0
      expect(repo.live_count).to eq 0
      expect(repo.graph_meta[:entry_point]).to eq(-1)
      repo.close
    end
  end
end
