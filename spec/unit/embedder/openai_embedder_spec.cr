require "../../spec_helper"

Spectator.describe Vecstolite::OpenAIEmbedder do
  let(dimensions) { 1024 }
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
      it "starts with 'localhost/'" do
        expect(embedder.model_name).to start_with("localhost/")
      end
      it "ends with model" do
        expect(embedder.model_name).to end_with(model)
      end
    end
  end

  describe "OpenAI" do
    let(model) { "text-embedding-3-small" }
    let(embedder) { described_class.new(dimensions, api_key, model) }

    describe "#model_name" do
      it "starts with 'api.openai.com/'" do
        expect(embedder.model_name).to start_with("api.openai.com/")
      end
      it "ends with model" do
        expect(embedder.model_name).to end_with(model)
      end
    end
  end
end
