require "../../spec_helper"

# Types used as M and P throughout these tests.
private record Lang, code : String do
  include JSON::Serializable
end

private record Translation, en : String, fr : String do
  include JSON::Serializable
end

Spectator.describe Vecstolite::SQLitePayloadVectorStore do
  let(db_file_name) { File.tempname("vecsto_payload", ".db") }
  let(embedder) { Vecstolite::LexicalEmbedder.new }

  # Convenience alias so tests don't repeat the type params.
  alias Store = Vecstolite::SQLitePayloadVectorStore(Lang, Translation)

  after_each { File.delete?(db_file_name) }

  # -------------------------------------------------------------------------
  # create / open lifecycle
  # -------------------------------------------------------------------------

  describe "#create" do
    it "creates a database file" do
      Store.create(db_file_name, embedder).close
      expect(File.exists?(db_file_name)).to be true
    end

    it "fails if the file already exists" do
      File.write(db_file_name, "boo")
      expect { Store.create(db_file_name, embedder) }.to raise_error
    end
  end

  describe "#open" do
    it "fails before create" do
      expect { Store.open(db_file_name, embedder) }.to raise_error
    end

    it "succeeds after create and close" do
      Store.create(db_file_name, embedder).close
      expect { Store.open(db_file_name, embedder) }.not_to raise_error
    end
  end

  # -------------------------------------------------------------------------
  # size
  # -------------------------------------------------------------------------

  describe "#size" do
    it "reflects entries added in the same session" do
      store = Store.create(db_file_name, embedder)
      store.add("The sky is blue.", meta: Lang.new("en"))
      store.add("Le ciel est bleu.", meta: Lang.new("fr"))
      expect(store.size).to eq(2)
    end

    context "after close and reopen" do
      before_each do
        s = Store.create(db_file_name, embedder)
        s.add("The sky is blue.", meta: Lang.new("en"))
        s.add("Le ciel est bleu.", meta: Lang.new("fr"))
        s.close
      end

      it "reflects stored entries even before any search (no cache hits)" do
        store = Store.open(db_file_name, embedder)
        expect(store.size).to eq(2)
      end
    end
  end

  # -------------------------------------------------------------------------
  # add_payload / get_payload
  # -------------------------------------------------------------------------

  describe "#add_payload" do
    it "returns a positive id" do
      store = Store.create(db_file_name, embedder)
      pid = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      expect(pid).to be > 0
    end

    it "returns distinct ids for distinct payloads" do
      store = Store.create(db_file_name, embedder)
      pid1 = store.add_payload(Translation.new(en: "Hello", fr: "Bonjour"))
      pid2 = store.add_payload(Translation.new(en: "Goodbye", fr: "Au revoir"))
      expect(pid1).not_to eq(pid2)
    end
  end

  describe "#get_payload" do
    it "retrieves a stored payload by id" do
      store = Store.create(db_file_name, embedder)
      translation = Translation.new(en: "The sky is blue", fr: "Le ciel est bleu")
      pid = store.add_payload(translation)
      result = store.get_payload(pid)
      expect(result).not_to be_nil
      expect(result.not_nil!.en).to eq(translation.en)
      expect(result.not_nil!.fr).to eq(translation.fr)
    end

    it "returns nil for an unknown id" do
      store = Store.create(db_file_name, embedder)
      expect(store.get_payload(999_i64)).to be_nil
    end
  end

  # -------------------------------------------------------------------------
  # search — meta and payload round-trips
  # -------------------------------------------------------------------------

  describe "#search" do
    it "returns meta for the matched embedding" do
      store = Store.create(db_file_name, embedder)
      store.add("The sky is blue.", meta: Lang.new("en"))
      results = store.search("sky colour", k: 1)
      expect(results.size).to eq(1)
      expect(results.first.meta.try(&.code)).to eq("en")
    end

    it "returns nil payload when none was linked" do
      store = Store.create(db_file_name, embedder)
      store.add("The sky is blue.", meta: Lang.new("en"))
      results = store.search("sky colour", k: 1)
      expect(results.first.payload).to be_nil
    end

    it "resolves the shared payload from a search result" do
      store = Store.create(db_file_name, embedder)
      translation = Translation.new(en: "The sky is blue", fr: "Le ciel est bleu")
      pid = store.add_payload(translation)
      store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
      store.add("Le ciel est bleu.", meta: Lang.new("fr"), payload_id: pid)

      results = store.search("sky colour", k: 1)
      payload = results.first.payload
      expect(payload).not_to be_nil
      expect(payload.not_nil!.en).to eq(translation.en)
      expect(payload.not_nil!.fr).to eq(translation.fr)
    end

    it "multiple embeddings share one payload (many-to-one)" do
      store = Store.create(db_file_name, embedder)
      translation = Translation.new(en: "The sky is blue", fr: "Le ciel est bleu")
      pid = store.add_payload(translation)
      store.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
      store.add("Le ciel est bleu.", meta: Lang.new("fr"), payload_id: pid)

      results = store.search("sky colour", k: 2)
      expect(results.size).to eq(2)
      results.each do |r|
        expect(r.payload.not_nil!.en).to eq(translation.en)
      end
    end

    context "after close and reopen" do
      before_each do
        s = Store.create(db_file_name, embedder)
        translation = Translation.new(en: "The sky is blue", fr: "Le ciel est bleu")
        pid = s.add_payload(translation)
        s.add("The sky is blue.", meta: Lang.new("en"), payload_id: pid)
        s.add("Le ciel est bleu.", meta: Lang.new("fr"), payload_id: pid)
        s.close
      end

      it "returns correct meta and payload after restore" do
        store = Store.open(db_file_name, embedder)
        results = store.search("sky is blue", k: 2)
        expect(results.size).to eq(2)
        # Both results should resolve the same payload regardless of ranking order.
        results.each do |r|
          expect(r.payload.not_nil!.en).to eq("The sky is blue")
        end
        # Both languages should be represented.
        langs = results.map { |r| r.meta.try(&.code) }
        expect(langs).to contain("en")
        expect(langs).to contain("fr")
      end
    end
  end
end
