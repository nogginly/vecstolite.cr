require "../../spec_helper"

private record Lang, code : String do
  include JSON::Serializable
end

# Same vectors as LexicalEmbedder, but reporting a different model, so the
# store's embedder check has something to object to.
private class OtherEmbedder
  include Vecstolite::VectorEmbedder

  getter dimensions : Int32

  def initialize(@dimensions : Int32)
    @inner = Vecstolite::LexicalEmbedder.new(@dimensions)
  end

  def model_name : String
    "other/lexical"
  end

  def embed(text : String) : Vecstolite::Embedding
    @inner.embed(text)
  end
end

private record Pair, en : String, fr : String do
  include JSON::Serializable
end

Spectator.describe Vecstolite::Store do
  alias Store = Vecstolite::Store(Lang, Pair)
  alias Index = Vecstolite::Index
  alias CacheMode = Vecstolite::CacheMode

  let(embedder) { Vecstolite::LexicalEmbedder.new(dimensions: 64) }
  let(db_file_name) { File.tempname("vecsto_store", ".db") }

  after_each { File.delete?(db_file_name) }

  SENTENCES = [
    "The sky is blue.",
    "Le ciel est bleu.",
    "The ocean is deep and blue.",
    "The sun is bright and warm.",
    "The grass is green.",
  ]

  private def seeded(store) : Array(Int64)
    SENTENCES.map_with_index do |text, i|
      store.add(text, meta: Lang.new(i.even? ? "en" : "fr"))
    end
  end

  # ---------------------------------------------------------------------------
  # lifecycle
  # ---------------------------------------------------------------------------

  describe ".open" do
    it "creates a database and reopens it" do
      store = Store.open(db_file_name, embedder)
      ids = seeded(store)
      store.close

      store = Store.open(db_file_name, embedder)
      expect(store.size).to eq SENTENCES.size
      expect(store.get(ids[0]).try(&.text)).to eq SENTENCES[0]
      store.close
    end

    it "works in memory" do
      store = Store.open(":memory:", embedder)
      seeded(store)
      expect(store.size).to eq SENTENCES.size
      store.close
    end

    it "closes after the block, even when it raises" do
      store = nil.as(Store?)
      expect {
        Store.open(db_file_name, embedder) do |opened|
          store = opened
          opened.add("The sky is blue.")
          raise "boom"
        end
      }.to raise_error(/boom/)

      expect(store.not_nil!.closed?).to be true
    end

    it "refuses a database written by another embedder" do
      Store.open(db_file_name, embedder) { |store| store.add("The sky is blue.") }

      expect { Store.open(db_file_name, OtherEmbedder.new(64)) }
        .to raise_error(Store::Error, /embedder/)
    end

    it "opens anyway when verification is waived" do
      Store.open(db_file_name, embedder) { |store| store.add("The sky is blue.") }

      store = Store.open(db_file_name, OtherEmbedder.new(64), verify_embedder: false)
      expect(store.size).to eq 1
      store.close
    end

    it "rejects writes to a readonly store" do
      Store.open(db_file_name, embedder) { |store| seeded(store) }

      store = Store.open(db_file_name, embedder, readonly: true)
      expect(store.size).to eq SENTENCES.size
      expect { store.add("The moon is bright.") }.to raise_error(Store::Error, /readonly/)
      store.close
    end

    it "refuses use after close" do
      store = Store.open(":memory:", embedder)
      store.close
      expect(store.closed?).to be true
      expect { store.size }.to raise_error(Store::Error, /closed/)
    end
  end

  # ---------------------------------------------------------------------------
  # entries
  # ---------------------------------------------------------------------------

  describe "#add" do
    it "returns stable ids and round-trips metadata" do
      store = Store.open(":memory:", embedder)
      id = store.add("The sky is blue.", meta: Lang.new("en"), key: "sky-en")

      entry = store.get(id).not_nil!
      expect(entry.text).to eq "The sky is blue."
      expect(entry.meta.not_nil!.code).to eq "en"
      expect(entry.key).to eq "sky-en"
      expect(store.get_by_key("sky-en").try(&.id)).to eq id
      store.close
    end

    it "accepts a precomputed vector" do
      store = Store.open(":memory:", embedder)
      vector = embedder.embed("The sky is blue.")
      id = store.add("The sky is blue.", vector: vector)

      expect(store.get(id).try(&.text)).to eq "The sky is blue."
      expect(store.size).to eq 1
      store.close
    end

    it "rejects a duplicate key" do
      store = Store.open(":memory:", embedder)
      store.add("The sky is blue.", key: "sky")
      expect { store.add("Le ciel est bleu.", key: "sky") }.to raise_error(/already in use/)
      store.close
    end

    it "leaves the store usable after a failed add" do
      store = Store.open(":memory:", embedder)
      seeded(store)
      expect { store.add("Le ciel est bleu.", key: nil, vector: Vecstolite::Embedding.new(3) { 0.5_f32 }) }
        .to raise_error

      expect(store.size).to eq SENTENCES.size
      expect(store.search("blue sky", k: 2).size).to eq 2
      store.close
    end
  end

  describe "#bulk" do
    it "adds everything in one transaction" do
      store = Store.open(db_file_name, embedder)
      store.bulk do |batch|
        SENTENCES.each_with_index { |text, i| batch.add(text, meta: Lang.new(i.even? ? "en" : "fr")) }
      end

      expect(store.size).to eq SENTENCES.size
      store.close

      store = Store.open(db_file_name, embedder)
      expect(store.search("blue sky", k: 3).size).to eq 3
      store.close
    end

    it "rolls back the whole batch on failure" do
      store = Store.open(db_file_name, embedder)
      store.add("The sky is blue.", key: "sky")

      expect {
        store.bulk do |batch|
          batch.add("The ocean is deep and blue.")
          batch.add("Le ciel est bleu.", key: "sky")
        end
      }.to raise_error

      expect(store.size).to eq 1
      expect(store.search("deep ocean", k: 1).first.text).to eq "The sky is blue."
      store.close
    end

    it "does nothing for an empty batch" do
      store = Store.open(":memory:", embedder)
      store.bulk { |batch| }
      expect(store.size).to eq 0
      store.close
    end
  end

  # ---------------------------------------------------------------------------
  # payloads
  # ---------------------------------------------------------------------------

  describe "payloads" do
    it "shares one payload across entries" do
      store = Store.open(":memory:", embedder)
      pid = store.add_payload(Pair.new("The sky is blue.", "Le ciel est bleu."))
      store.add("The sky is blue.", payload_id: pid)
      store.add("Le ciel est bleu.", payload_id: pid)

      results = store.search("blue sky", k: 2)
      expect(results.size).to eq 2
      expect(results.all? { |r| r.payload_id == pid }).to be true
      expect(results.first.payload.not_nil!.fr).to eq "Le ciel est bleu."
      store.close
    end

    it "updates payload content without touching entries" do
      store = Store.open(":memory:", embedder)
      pid = store.add_payload(Pair.new("The sky is blue.", "Le ciel est bleu."))
      id = store.add("The sky is blue.", payload_id: pid)

      expect(store.update_payload(pid, Pair.new("The sky is blue.", "Le ciel est azur."))).to be true
      expect(store.get_payload(pid).not_nil!.fr).to eq "Le ciel est azur."
      expect(store.get(id).not_nil!.text).to eq "The sky is blue."
      store.close
    end
  end

  # ---------------------------------------------------------------------------
  # search and index strategies
  # ---------------------------------------------------------------------------

  describe "#search" do
    it "ranks the closest entry first" do
      store = Store.open(":memory:", embedder)
      seeded(store)

      result = store.search("green grass", k: 1).first
      expect(result.text).to eq "The grass is green."
      expect(result.score).to be > 0.0_f32
      store.close
    end

    it "searches with a vector directly" do
      store = Store.open(":memory:", embedder)
      seeded(store)

      results = store.search_vector(embedder.embed("bright sun"), k: 1)
      expect(results.first.text).to eq "The sun is bright and warm."
      store.close
    end

    it "returns nothing from an empty store" do
      store = Store.open(":memory:", embedder)
      expect(store.search("anything", k: 5)).to be_empty
      store.close
    end
  end

  describe "index strategies" do
    it "runs without a graph when flat" do
      store = Store.open(db_file_name, embedder, index: Index.flat)
      seeded(store)

      expect(store.search("green grass", k: 1).first.text).to eq "The grass is green."
      expect(store.stats[:index]).to eq :flat
      expect(store.stats[:indexed_nodes]).to eq 0
      store.close
    end

    it "rebuilds when reopened with a different strategy" do
      Store.open(db_file_name, embedder, index: Index.flat) { |store| seeded(store) }

      store = Store.open(db_file_name, embedder, index: Index.hnsw(seed: 42))
      expect(store.stats[:indexed_nodes]).to eq SENTENCES.size
      expect(store.search("green grass", k: 1).first.text).to eq "The grass is green."
      store.close
    end

    it "discards a stale graph when switching back to flat" do
      Store.open(db_file_name, embedder, index: Index.hnsw(seed: 42)) { |store| seeded(store) }

      store = Store.open(db_file_name, embedder, index: Index.flat)
      expect(store.stats[:indexed_nodes]).to eq 0
      expect(store.search("green grass", k: 1).first.text).to eq "The grass is green."
      store.close
    end
  end

  # ---------------------------------------------------------------------------
  # cache modes
  # ---------------------------------------------------------------------------

  describe "cache modes" do
    {% for mode in [:memory, :lru, :disk] %}
      it "persists and searches with the {{ mode.id }} cache" do
        cache =
          case {{ mode }}
          when :memory then CacheMode.memory
          when :lru    then CacheMode.lru(64_i64 * Vecstolite::KB)
          else              CacheMode.disk
          end

        Store.open(db_file_name, embedder, cache: cache) { |store| seeded(store) }

        store = Store.open(db_file_name, embedder, cache: cache)
        expect(store.size).to eq SENTENCES.size
        expect(store.search("green grass", k: 1).first.text).to eq "The grass is green."
        store.close
      end
    {% end %}

    it "rebuilds the graph when a memory-cached store was not closed cleanly" do
      store = Store.open(db_file_name, embedder, cache: CacheMode.memory)
      seeded(store)
      # Entries are committed as they are added; graph nodes are not, which is
      # what an unclean exit leaves behind.
      expect(store.stats[:indexed_nodes]).to eq 0

      reopened = Store.open(db_file_name, embedder, cache: CacheMode.memory)
      expect(reopened.size).to eq SENTENCES.size
      expect(reopened.search("green grass", k: 1).first.text).to eq "The grass is green."
      reopened.close
    end
  end

  describe "#stats" do
    it "reports entries, index and embedder" do
      store = Store.open(":memory:", embedder)
      seeded(store)

      stats = store.stats
      expect(stats[:entries]).to eq SENTENCES.size
      expect(stats[:tombstones]).to eq 0
      expect(stats[:index]).to eq :hnsw
      expect(stats[:dimensions]).to eq 64
      expect(stats[:embedder]).to eq embedder.model_name
      store.close
    end
  end
end
