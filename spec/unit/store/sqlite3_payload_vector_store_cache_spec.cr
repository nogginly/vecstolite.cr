require "../../spec_helper"

private record Lang, code : String do
  include JSON::Serializable
end

private record Translation, en : String, fr : String do
  include JSON::Serializable
end

# Shared corpus used across all cache-mode tests.
private SENTENCES = [
  "The sky is blue.",
  "Le ciel est bleu.",
  "The ocean is deep and blue.",
  "Die Sonne scheint hell.",
  "The sun is bright and warm.",
  "Es regnet heute.",
  "The grass is green.",
  "L'herbe est verte.",
]

# Build a store (default LRU cache), add SENTENCES, close. Returns db path.
private def build_store(path : String, embedder : Vecstolite::VectorEmbedder) : String
  alias_store = Vecstolite::SQLitePayloadVectorStore(Lang, Translation)
  store = alias_store.create(path, embedder)
  SENTENCES.each_with_index do |s, i|
    store.add(s, meta: Lang.new(i.even? ? "en" : "fr"))
  end
  store.close
  path
end

Spectator.describe Vecstolite::SQLitePayloadVectorStore do
  let(db_file_name) { File.tempname("vecsto_cache", ".db") }
  let(embedder) { Vecstolite::LexicalEmbedder.new }

  alias Store = Vecstolite::SQLitePayloadVectorStore(Lang, Translation)

  after_each { File.delete?(db_file_name) }

  # -------------------------------------------------------------------------
  # cache modes produce identical search results
  # -------------------------------------------------------------------------

  describe "cache modes" do
    before_each { build_store(db_file_name, embedder) }

    it "LRU mode (default) returns results" do
      store = Store.open(db_file_name, embedder)
      results = store.search("blue sky", k: 3)
      store.close
      expect(results.size).to eq(3)
    end

    it "disk mode returns identical results to LRU mode" do
      lru_store = Store.open(db_file_name, embedder)
      lru_results = lru_store.search("blue sky", k: 3).map(&.text)
      lru_store.close

      disk_store = Store.open(db_file_name, embedder, cache_max_bytes: nil)
      disk_results = disk_store.search("blue sky", k: 3).map(&.text)
      disk_store.close

      expect(disk_results).to eq(lru_results)
    end

    it "memory mode (load_all_in_memory!) returns identical results to LRU mode" do
      lru_store = Store.open(db_file_name, embedder)
      lru_results = lru_store.search("blue sky", k: 3).map(&.text)
      lru_store.close

      mem_store = Store.open(db_file_name, embedder)
      mem_store.load_all_in_memory!
      mem_results = mem_store.search("blue sky", k: 3).map(&.text)
      mem_store.close

      expect(mem_results).to eq(lru_results)
    end
  end

  # -------------------------------------------------------------------------
  # stats
  # -------------------------------------------------------------------------

  describe "#stats" do
    it "reports :lru mode for default open" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder)
      expect(store.stats[:cache_mode]).to eq(:lru)
      store.close
    end

    it "reports :disk mode when cache_max_bytes is nil" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder, cache_max_bytes: nil)
      expect(store.stats[:cache_mode]).to eq(:disk)
      store.close
    end

    it "reports :memory mode after load_all_in_memory!" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder)
      store.load_all_in_memory!
      expect(store.stats[:cache_mode]).to eq(:memory)
      store.close
    end

    it "accumulates cache misses on disk mode" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder, cache_max_bytes: nil)
      store.search("blue sky", k: 3)
      expect(store.stats[:cache_misses]).to be > 0
      store.close
    end

    it "accumulates cache hits on repeated LRU searches" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder)
      store.search("blue sky", k: 3) # primes cache
      store.search("blue sky", k: 3) # should hit cache
      expect(store.stats[:cache_hits]).to be > 0
      store.close
    end

    it "reports zero hits and misses for memory mode" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder)
      store.load_all_in_memory!
      store.search("blue sky", k: 3)
      expect(store.stats[:cache_hits]).to eq(0)
      expect(store.stats[:cache_misses]).to eq(0)
      store.close
    end

    it "respects a tiny cache budget (evictions occur)" do
      build_store(db_file_name, embedder)
      # Budget of 1 byte forces eviction after every insert.
      store = Store.open(db_file_name, embedder, cache_max_bytes: 1_i64)
      store.search("blue sky", k: 3)
      expect(store.stats[:cache_evictions]).to be > 0
      store.close
    end
  end

  # -------------------------------------------------------------------------
  # bulk_add
  # -------------------------------------------------------------------------

  describe "#bulk_add" do
    it "adds all entries within one call" do
      store = Store.create(db_file_name, embedder)
      store.bulk_add do |add|
        SENTENCES.each_with_index do |s, i|
          add.call(s, meta: Lang.new(i.even? ? "en" : "fr"))
        end
      end
      expect(store.size).to eq(SENTENCES.size)
      store.close
    end

    it "returns the same search results as individual adds" do
      # Reference: individual adds
      ref_store = Store.create(db_file_name, embedder)
      SENTENCES.each_with_index do |s, i|
        ref_store.add(s, meta: Lang.new(i.even? ? "en" : "fr"))
      end
      ref_results = ref_store.search("blue sky", k: 3).map(&.text)
      ref_store.close
      File.delete?(db_file_name)

      # Bulk add
      bulk_store = Store.create(db_file_name, embedder)
      bulk_store.bulk_add do |add|
        SENTENCES.each_with_index do |s, i|
          add.call(s, meta: Lang.new(i.even? ? "en" : "fr"))
        end
      end
      bulk_results = bulk_store.search("blue sky", k: 3).map(&.text)
      bulk_store.close

      expect(bulk_results).to eq(ref_results)
    end

    it "rolls back all entries on error, leaving the store unchanged" do
      store = Store.create(db_file_name, embedder)
      store.add("Before bulk.", meta: Lang.new("en"))
      size_before = store.size

      expect do
        store.bulk_add do |add|
          add.call("Entry one.", meta: Lang.new("en"))
          add.call("Entry two.", meta: Lang.new("fr"))
          raise "simulated failure"
        end
      end.to raise_error(Exception, "simulated failure")

      expect(store.size).to eq(size_before)
      store.close
    end

    it "persists correctly across close and reopen" do
      store = Store.create(db_file_name, embedder)
      store.bulk_add do |add|
        SENTENCES.each_with_index do |s, i|
          add.call(s, meta: Lang.new(i.even? ? "en" : "fr"))
        end
      end
      store.close

      store2 = Store.open(db_file_name, embedder)
      expect(store2.size).to eq(SENTENCES.size)
      results = store2.search("blue sky", k: 3)
      expect(results.size).to eq(3)
      store2.close
    end

    it "works with payloads" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
      store.bulk_add do |add|
        add.call("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
        add.call("Le ciel est bleu.", meta: Lang.new("fr"), payload_id: pid)
      end
      results = store.search("sky colour", k: 2)
      results.each { |r| expect(r.payload).not_to be_nil }
      store.close
    end
  end

  # -------------------------------------------------------------------------
  # load_all_in_memory!
  # -------------------------------------------------------------------------

  describe "#load_all_in_memory!" do
    it "is a no-op when already in memory mode" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder)
      store.load_all_in_memory!
      store.load_all_in_memory! # second call must not raise or corrupt state
      expect(store.stats[:cache_mode]).to eq(:memory)
      store.close
    end

    it "allows add after switching to memory mode" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder)
      store.load_all_in_memory!
      store.add("A new sentence.", meta: Lang.new("en"))
      expect(store.size).to eq(SENTENCES.size + 1)
      store.close
    end

    it "persists new entries added after load_all_in_memory! across close/reopen" do
      build_store(db_file_name, embedder)
      store = Store.open(db_file_name, embedder)
      store.load_all_in_memory!
      store.add("A new sentence after eager load.", meta: Lang.new("en"))
      store.close

      store2 = Store.open(db_file_name, embedder)
      expect(store2.size).to eq(SENTENCES.size + 1)
      store2.close
    end
  end

  # -------------------------------------------------------------------------
  # persistence round-trips across all modes
  # -------------------------------------------------------------------------

  describe "persistence round-trips" do
    {% for mode in [:lru, :disk, :memory] %}
      context "{{ mode.id }} mode" do
        it "restores correct size and search results after close/reopen" do
          # Create and populate
          store = Store.create(db_file_name, embedder)
          SENTENCES.each_with_index do |s, i|
            store.add(s, meta: Lang.new(i.even? ? "en" : "fr"))
          end
          store.close

          # Reopen in the target mode
          {% if mode == :lru %}
            store2 = Store.open(db_file_name, embedder)
          {% elsif mode == :disk %}
            store2 = Store.open(db_file_name, embedder, cache_max_bytes: nil)
          {% else %}
            store2 = Store.open(db_file_name, embedder)
            store2.load_all_in_memory!
          {% end %}

          expect(store2.size).to eq(SENTENCES.size)
          results = store2.search("blue sky", k: 3)
          expect(results.size).to eq(3)
          store2.close
        end
      end
    {% end %}
  end

  # -------------------------------------------------------------------------
  # Vecstolite::MB / KB / GB constants
  # -------------------------------------------------------------------------

  describe "Vecstolite size constants" do
    it "KB is 1024" do
      expect(Vecstolite::KB).to eq(1_024_i64)
    end

    it "MB is 1024 KB" do
      expect(Vecstolite::MB).to eq(1_024_i64 * Vecstolite::KB)
    end

    it "GB is 1024 MB" do
      expect(Vecstolite::GB).to eq(1_024_i64 * Vecstolite::MB)
    end

    it "can be used as cache_max_bytes" do
      store = Store.create(db_file_name, embedder, cache_max_bytes: 256 * Vecstolite::MB)
      expect(store.stats[:cache_mode]).to eq(:lru)
      store.close
    end
  end
end
