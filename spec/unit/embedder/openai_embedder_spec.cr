require "../../spec_helper"

Spectator.describe Vecstolite::OpenAIEmbedder do
  let(dimensions) { 768 }
  let(api_key) { "ollama" }
  let(embedder) { described_class.new(dimensions, api_key) }

  describe "Type is" do
    it "VectorEmbedder" do
      expect(embedder.is_a? Vecstolite::VectorEmbedder).to be true
    end
  end

  describe "Ollama" do
    let(base_url) { "http://localhost:11434" }
    let(model) { "nomic-embed-text-v2-moe" }
    let(embedder) { described_class.new(dimensions, api_key, model, base_url) }

    describe "#model_name" do
      it "ends with '@localhost'" do
        expect(embedder.model_name).to end_with("@localhost")
      end
      it "starts with model" do
        expect(embedder.model_name).to start_with(model)
      end
    end

    describe "#embed" do
      context "get an embedding" do
        before_each do
          base_name = __DIR__
        end
        it "succeeds" do
          Wiretap.intercept("embed") do
            embedding = embedder.embed("The weather in the Ozarks is colder today than yesterday.")
            expect(embedding).to be_a(Slice(Float32))
            expect(embedding.size).to eq(dimensions)
          end
        end
      end
    end
  end

  describe "OpenAI" do
    let(model) { "text-embedding-3-small" }
    let(embedder) { described_class.new(dimensions, api_key, model) }

    describe "#model_name" do
      it "ends with '@api.openai.com'" do
        expect(embedder.model_name).to end_with("@api.openai.com")
      end
      it "starts with model" do
        expect(embedder.model_name).to start_with(model)
      end
    end
  end
end
