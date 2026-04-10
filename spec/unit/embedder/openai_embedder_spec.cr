require "../../spec_helper"

Spectator.describe Vecstolite::OpenAIEmbedder do
  let(dimensions) { 1024 }
  let(api_key) { "ollama" }
  let(base_url) { "http://localhost:11434" }
  let(embedder) { described_class.new dimensions, api_key, base_url }

  describe "Type is" do
    it "VectorEmbedder" do
      expect(embedder.is_a? Vecstolite::VectorEmbedder).to be true
    end
  end
end
