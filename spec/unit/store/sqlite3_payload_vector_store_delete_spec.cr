require "../../spec_helper"

private record Lang, code : String do
  include JSON::Serializable
end

private record Translation, en : String, fr : String do
  include JSON::Serializable
end

Spectator.describe Vecstolite::SQLitePayloadVectorStore do
  let(db_file_name) { File.tempname("vecsto_delete", ".db") }
  let(embedder) { Vecstolite::LexicalEmbedder.new }

  alias Store = Vecstolite::SQLitePayloadVectorStore(Lang, Translation)

  after_each { File.delete?(db_file_name) }

  # -------------------------------------------------------------------------
  # #delete_payload
  # -------------------------------------------------------------------------

  describe "#delete_payload" do
    it "removes the payload itself" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
      store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)

      store.delete_payload(pid)

      expect(store.get_payload(pid)).to be_nil
      store.close
    end

    it "removes every entry referencing the payload from search results" do
      store = Store.create(db_file_name, embedder, hnsw_seed: 1)
      pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
      store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
      store.add("Le ciel est bleu.", meta: Lang.new("fr"), payload_id: pid)
      store.add("The grass is green.", meta: Lang.new("en"))

      store.delete_payload(pid)

      results = store.search("sky blue grass", k: 3)
      texts = results.map(&.text)
      expect(texts).not_to contain("The sky is blue.")
      expect(texts).not_to contain("Le ciel est bleu.")
      expect(texts).to contain("The grass is green.")
      store.close
    end

    it "does not shrink size — physical removal is deferred to compact!" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      store.add("Hello.", meta: Lang.new("en"), payload_id: pid)
      store.add("Goodbye.", meta: Lang.new("en"))
      size_before = store.size

      store.delete_payload(pid)

      expect(store.size).to eq(size_before)
      store.close
    end

    it "still satisfies k by oversampling around tombstones" do
      store = Store.create(db_file_name, embedder, hnsw_seed: 1)
      pids = Array(Int64).new
      5.times do |i|
        pid = store.add_payload(Translation.new(en: "item #{i}", fr: "article #{i}"))
        pids << pid
        store.add("Item number #{i}.", meta: Lang.new("en"), payload_id: pid)
      end

      # Delete all but one of the "item" payloads.
      pids.first(4).each { |pid| store.delete_payload(pid) }

      results = store.search("item number", k: 1)
      expect(results.size).to eq(1)
      expect(results.first.text).to eq("Item number 4.")
      store.close
    end

    it "is a no-op for an unknown payload id" do
      store = Store.create(db_file_name, embedder)
      store.add("Hello.", meta: Lang.new("en"))
      expect { store.delete_payload(999_i64) }.not_to raise_error
      store.close
    end

    it "leaves unrelated payloads and entries intact" do
      store = Store.create(db_file_name, embedder)
      pid1 = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      pid2 = store.add_payload(Translation.new(en: "Goodbye", fr: "Au revoir"))
      store.add("Hello.", meta: Lang.new("en"), payload_id: pid1)
      store.add("Goodbye.", meta: Lang.new("en"), payload_id: pid2)

      store.delete_payload(pid1)

      expect(store.get_payload(pid2)).not_to be_nil
      results = store.search("Goodbye", k: 1)
      expect(results.first.text).to eq("Goodbye.")
      store.close
    end

    it "evicts cached entries so a previously-searched result disappears" do
      store = Store.create(db_file_name, embedder, hnsw_seed: 1)
      pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
      store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
      store.add("The grass is green.", meta: Lang.new("en"))

      # Prime the entry cache for the soon-to-be-deleted entry.
      store.search("sky", k: 1)

      store.delete_payload(pid)

      results = store.search("sky grass", k: 2)
      expect(results.map(&.text)).not_to contain("The sky is blue.")
      store.close
    end

    it "raises on a closed store" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      store.close
      expect { store.delete_payload(pid) }.to raise_error
    end

    it "raises on a readonly store" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      store.add("Hello.", meta: Lang.new("en"), payload_id: pid)
      store.close

      ro_store = Store.open(db_file_name, embedder, readonly: true)
      expect { ro_store.delete_payload(pid) }.to raise_error
      ro_store.close
    end

    {% for mode in [:lru, :disk, :memory] %}
      context "{{ mode.id }} mode, after close and reopen" do
        it "tombstoned entries stay out of search results" do
          store = Store.create(db_file_name, embedder, hnsw_seed: 1)
          pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
          store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
          store.add("The grass is green.", meta: Lang.new("en"))
          store.delete_payload(pid)
          store.close

          {% if mode == :lru %}
            reopened = Store.open(db_file_name, embedder)
          {% elsif mode == :disk %}
            reopened = Store.open(db_file_name, embedder, cache_max_bytes: nil)
          {% else %}
            reopened = Store.open(db_file_name, embedder)
            reopened.load_all_in_memory!
          {% end %}

          results = reopened.search("sky grass", k: 2)
          texts = results.map(&.text)
          expect(texts).not_to contain("The sky is blue.")
          expect(texts).to contain("The grass is green.")
          reopened.close
        end
      end
    {% end %}
  end

  # -------------------------------------------------------------------------
  # #compact!
  # -------------------------------------------------------------------------

  describe "#compact!" do
    it "reduces size to the live entry count" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      store.add("Hello.", meta: Lang.new("en"), payload_id: pid)
      store.add("Goodbye.", meta: Lang.new("en"))
      store.delete_payload(pid)
      expect(store.size).to eq(2) # tombstone still counted

      store.compact!

      expect(store.size).to eq(1)
      store.close
    end

    it "search still returns correct live results after compact" do
      store = Store.create(db_file_name, embedder, hnsw_seed: 1)
      pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
      store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
      store.add("The grass is green.", meta: Lang.new("en"))
      store.delete_payload(pid)

      store.compact!

      results = store.search("sky grass", k: 2)
      expect(results.size).to eq(1)
      expect(results.first.text).to eq("The grass is green.")
      store.close
    end

    it "new entries can be added after compact and are searchable" do
      store = Store.create(db_file_name, embedder, hnsw_seed: 1)
      pid = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      store.add("Hello.", meta: Lang.new("en"), payload_id: pid)
      store.delete_payload(pid)
      store.compact!

      store.add("Brand new entry.", meta: Lang.new("en"))

      expect(store.size).to eq(1)
      results = store.search("Brand new entry", k: 1)
      expect(results.first.text).to eq("Brand new entry.")
      store.close
    end

    it "persists correctly across close and reopen" do
      store = Store.create(db_file_name, embedder, hnsw_seed: 1)
      pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
      store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
      store.add("The grass is green.", meta: Lang.new("en"))
      store.delete_payload(pid)
      store.compact!
      store.close

      reopened = Store.open(db_file_name, embedder)
      expect(reopened.size).to eq(1)
      results = reopened.search("grass", k: 1)
      expect(results.first.text).to eq("The grass is green.")
      reopened.close
    end

    it "preserves payload associations for surviving entries" do
      store = Store.create(db_file_name, embedder, hnsw_seed: 1)
      pid_keep = store.add_payload(Translation.new(en: "Keep me", fr: "Garde-moi"))
      pid_drop = store.add_payload(Translation.new(en: "Drop me", fr: "Laisse-moi"))
      store.add("Keep me.", meta: Lang.new("en"), payload_id: pid_keep)
      store.add("Drop me.", meta: Lang.new("en"), payload_id: pid_drop)
      store.delete_payload(pid_drop)

      store.compact!

      results = store.search("Keep me", k: 1)
      expect(results.first.payload.not_nil!.en).to eq("Keep me")
      store.close
    end

    it "preserves the current memory mode (load_all_in_memory! survives compact)" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      store.add("Hello.", meta: Lang.new("en"), payload_id: pid)
      store.add("Goodbye.", meta: Lang.new("en"))
      store.load_all_in_memory!
      store.delete_payload(pid)

      store.compact!

      expect(store.stats[:cache_mode]).to eq(:memory)
      store.close
    end

    it "is safe to call with no tombstones present" do
      store = Store.create(db_file_name, embedder)
      store.add("Hello.", meta: Lang.new("en"))
      expect { store.compact! }.not_to raise_error
      expect(store.size).to eq(1)
      store.close
    end

    it "handles repeated delete/compact cycles" do
      store = Store.create(db_file_name, embedder, hnsw_seed: 1)
      3.times do |round|
        pid_drop = store.add_payload(Translation.new(en: "drop #{round}", fr: "laisse #{round}"))
        store.add("Drop round #{round}.", meta: Lang.new("en"), payload_id: pid_drop)
        pid_keep = store.add_payload(Translation.new(en: "keep #{round}", fr: "garde #{round}"))
        store.add("Keep round #{round}.", meta: Lang.new("en"), payload_id: pid_keep)

        store.delete_payload(pid_drop)
        store.compact!
      end

      expect(store.size).to eq(3)
      results = store.search("Keep round", k: 3)
      results.each { |r| expect(r.text).to start_with("Keep round") }
      store.close
    end

    it "raises on a closed store" do
      store = Store.create(db_file_name, embedder)
      store.close
      expect { store.compact! }.to raise_error
    end

    it "raises on a readonly store" do
      store = Store.create(db_file_name, embedder)
      store.add("Hello.", meta: Lang.new("en"))
      store.close

      ro_store = Store.open(db_file_name, embedder, readonly: true)
      expect { ro_store.compact! }.to raise_error
      ro_store.close
    end

    it "ending with zero live entries leaves an empty, working store" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      store.add("Hello.", meta: Lang.new("en"), payload_id: pid)
      store.delete_payload(pid)

      store.compact!

      expect(store.size).to eq(0)
      expect(store.search("anything", k: 5).size).to eq(0)

      store.add("Fresh start.", meta: Lang.new("en"))
      expect(store.size).to eq(1)
      store.close
    end

    {% for mode in [:lru, :disk, :memory] %}
      context "{{ mode.id }} mode" do
        it "rebuilds the graph correctly after compact, even after reopen" do
          store = Store.create(db_file_name, embedder, hnsw_seed: 1)
          pid = store.add_payload(Translation.new(en: "The sky is blue", fr: "Le ciel est bleu"))
          store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
          store.add("The grass is green.", meta: Lang.new("en"))
          store.add("The ocean is deep.", meta: Lang.new("en"))
          store.delete_payload(pid)
          store.compact!
          store.close

          {% if mode == :lru %}
            reopened = Store.open(db_file_name, embedder)
          {% elsif mode == :disk %}
            reopened = Store.open(db_file_name, embedder, cache_max_bytes: nil)
          {% else %}
            reopened = Store.open(db_file_name, embedder)
            reopened.load_all_in_memory!
          {% end %}

          expect(reopened.size).to eq(2)
          results = reopened.search("grass ocean", k: 2)
          texts = results.map(&.text)
          expect(texts).to contain("The grass is green.")
          expect(texts).to contain("The ocean is deep.")
          expect(texts).not_to contain("The sky is blue.")
          reopened.close
        end
      end
    {% end %}
  end
end
