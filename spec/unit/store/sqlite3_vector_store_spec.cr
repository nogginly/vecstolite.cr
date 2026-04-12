require "../../spec_helper"

Spectator.describe Vecstolite::SQLiteVectorStore do
  let(db_file_name) { File.tempname("vecsto", ".db") }
  let(embedder) { Vecstolite::LexicalEmbedder.new }

  after_each do
    # Cleanup temp files after every test
    File.delete?(db_file_name)
  end

  describe "Type is" do
    let(create_store) { described_class.create(db_file_name, embedder) }

    it "VectorStore" do
      expect(create_store.is_a? Vecstolite::VectorStore).to be true
    end
  end

  describe "#create" do
    let(create_store) { described_class.create(db_file_name, embedder) }

    it "succeeds" do
      create_store
      expect(File.exists?(db_file_name)).to be true
    end

    context "when file exists" do
      before_each do
        File.write(db_file_name, "Boo")
      end

      it "fails" do
        expect { create_store }.to raise_error
      end
    end
  end

  describe "#open" do
    context "before create" do
      let(open_store) { described_class.open(db_file_name, embedder) }

      it "fails" do
        expect { open_store }.to raise_error
      end
    end

    context "after create" do
      let(create_store) { described_class.create(db_file_name, embedder) }
      let(open_store) { described_class.open(db_file_name, embedder) }

      before_each do
        create_store.close
      end

      it "succeed" do
        expect { open_store }.not_to raise_error
      end
    end
  end
end
