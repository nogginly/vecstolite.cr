require "../../spec_helper"

private record Lang, code : String do
  include JSON::Serializable
end

private record Pair, en : String, fr : String do
  include JSON::Serializable
end

Spectator.describe "Vecstolite::Store deletion" do
  alias Store = Vecstolite::Store(Lang, Pair)
  alias Index = Vecstolite::Index
  alias CacheMode = Vecstolite::CacheMode

  let(embedder) { Vecstolite::LexicalEmbedder.new(dimensions: 64) }
  let(db_file_name) { File.tempname("vecsto_delete", ".db") }

  after_each { File.delete?(db_file_name) }

  CORPUS = [
    "The sky is blue.",
    "The ocean is deep and blue.",
    "The sun is bright and warm.",
    "The grass is green.",
    "The snow is cold and white.",
    "The river runs dark and deep.",
  ]

  private def seeded(store) : Array(Int64)
    CORPUS.map_with_index { |text, i| store.add(text, meta: Lang.new(i.even? ? "en" : "fr")) }
  end

  # ---------------------------------------------------------------------------
  # deleting entries
  # ---------------------------------------------------------------------------

  describe "#delete" do
    it "removes an entry from results and counts" do
      store = Store.open(":memory:", embedder)
      ids = seeded(store)

      expect(store.delete(ids[3])).to be true
      expect(store.size).to eq CORPUS.size - 1
      expect(store.tombstones).to eq 1
      expect(store.search("green grass", k: 3).map(&.text)).not_to contain CORPUS[3]
      store.close
    end

    it "releases text and metadata immediately" do
      store = Store.open(":memory:", embedder)
      ids = seeded(store)
      store.delete(ids[0])

      entry = store.get(ids[0]).not_nil!
      expect(entry.text).to be_nil
      expect(entry.meta).to be_nil
      store.close
    end

    it "reports false for an unknown or already-deleted entry" do
      store = Store.open(":memory:", embedder)
      ids = seeded(store)

      expect(store.delete(ids[0])).to be true
      expect(store.delete(ids[0])).to be false
      expect(store.delete(9_999_i64)).to be false
      store.close
    end

    it "deletes by key" do
      store = Store.open(":memory:", embedder)
      store.add("The sky is blue.", key: "sky")

      expect(store.delete_by_key("sky")).to be true
      expect(store.delete_by_key("sky")).to be false
      expect(store.size).to eq 0
      store.close
    end

    it "keeps deletions across close and reopen" do
      store = Store.open(db_file_name, embedder)
      ids = seeded(store)
      store.delete(ids[2])
      store.close

      store = Store.open(db_file_name, embedder)
      expect(store.size).to eq CORPUS.size - 1
      expect(store.search("bright sun", k: 3).map(&.text)).not_to contain CORPUS[2]
      store.close
    end

    it "still routes searches through a deleted node" do
      # The tombstoned node stays wired into the graph, so its live
      # neighbours must remain reachable.
      store = Store.open(":memory:", embedder)
      ids = seeded(store)
      store.delete(ids[1])

      found = store.search("deep dark river", k: 2).map(&.text)
      expect(found).not_to contain CORPUS[1]
      expect(found).to contain CORPUS[5]
      store.close
    end
  end

  describe "#delete_payload" do
    it "removes the payload and every entry using it" do
      store = Store.open(":memory:", embedder)
      pid = store.add_payload(Pair.new("The sky is blue.", "Le ciel est bleu."))
      store.add("The sky is blue.", payload_id: pid)
      store.add("Le ciel est bleu.", payload_id: pid)
      store.add("The grass is green.")

      expect(store.delete_payload(pid)).to eq 2
      expect(store.size).to eq 1
      expect(store.get_payload(pid)).to be_nil
      store.close
    end

    it "reports nothing deleted for an unknown payload" do
      store = Store.open(":memory:", embedder)
      seeded(store)
      expect(store.delete_payload(9_999_i64)).to eq 0
      expect(store.size).to eq CORPUS.size
      store.close
    end
  end

  # ---------------------------------------------------------------------------
  # compaction
  # ---------------------------------------------------------------------------

  describe "#compact!" do
    it "reclaims tombstones and keeps the store searchable" do
      store = Store.open(":memory:", embedder)
      ids = seeded(store)
      store.delete(ids[0])
      store.delete(ids[3])

      store.compact!
      expect(store.tombstones).to eq 0
      expect(store.size).to eq CORPUS.size - 2
      expect(store.total).to eq CORPUS.size - 2
      expect(store.search("deep ocean", k: 1).first.text).to eq CORPUS[1]
      store.close
    end

    it "leaves surviving ids and keys untouched" do
      store = Store.open(":memory:", embedder)
      store.add("The sky is blue.", key: "sky")
      doomed = store.add("The grass is green.")
      survivor = store.add("The sun is bright and warm.", key: "sun")

      store.delete(doomed)
      store.compact!

      expect(store.get(survivor).not_nil!.text).to eq "The sun is bright and warm."
      expect(store.get_by_key("sun").try(&.id)).to eq survivor
      expect(store.get(doomed)).to be_nil
      store.close
    end

    it "does nothing when there is nothing to reclaim" do
      store = Store.open(":memory:", embedder)
      seeded(store)
      before = store.stats[:indexed_nodes]

      store.compact!
      expect(store.stats[:indexed_nodes]).to eq before
      expect(store.size).to eq CORPUS.size
      store.close
    end

    it "survives close and reopen" do
      store = Store.open(db_file_name, embedder)
      ids = seeded(store)
      store.delete(ids[4])
      store.compact!
      store.close

      store = Store.open(db_file_name, embedder)
      expect(store.size).to eq CORPUS.size - 1
      expect(store.tombstones).to eq 0
      expect(store.search("cold snow", k: 2).map(&.text)).not_to contain CORPUS[4]
      store.close
    end

    it "handles repeated delete and compact cycles" do
      store = Store.open(db_file_name, embedder)
      ids = seeded(store)

      3.times do |round|
        store.delete(ids[round])
        store.compact!
        expect(store.tombstones).to eq 0
        expect(store.size).to eq CORPUS.size - (round + 1)
      end

      expect(store.search("dark river", k: 1).first.text).to eq CORPUS[5]
      store.close
    end
  end

  # ---------------------------------------------------------------------------
  # across cache modes and strategies
  # ---------------------------------------------------------------------------

  describe "cache modes" do
    {% for mode in [:memory, :lru, :disk] %}
      it "deletes and compacts with the {{ mode.id }} cache" do
        cache =
          case {{ mode }}
          when :memory then CacheMode.memory
          when :lru    then CacheMode.lru(64_i64 * Vecstolite::KB)
          else              CacheMode.disk
          end

        store = Store.open(db_file_name, embedder, cache: cache)
        ids = seeded(store)
        store.delete(ids[1])
        store.compact!
        store.close

        store = Store.open(db_file_name, embedder, cache: cache)
        expect(store.size).to eq CORPUS.size - 1
        expect(store.search("green grass", k: 1).first.text).to eq CORPUS[3]
        store.close
      end
    {% end %}
  end

  describe "flat index" do
    it "needs no graph rebuild to reclaim space" do
      store = Store.open(db_file_name, embedder, index: Index.flat)
      ids = seeded(store)
      store.delete(ids[0])

      expect(store.search("blue sky", k: 2).map(&.text)).not_to contain CORPUS[0]
      store.compact!
      expect(store.tombstones).to eq 0
      expect(store.stats[:indexed_nodes]).to eq 0
      expect(store.size).to eq CORPUS.size - 1
      store.close
    end
  end

  describe "heavily tombstoned store" do
    it "returns what is left rather than escalating to a full scan" do
      store = Store.open(":memory:", embedder)
      ids = seeded(store)
      ids[0..4].each { |id| store.delete(id) }

      results = store.search("dark river", k: 5)
      expect(results.size).to eq 1
      expect(results.first.text).to eq CORPUS[5]
      store.close
    end
  end
end
