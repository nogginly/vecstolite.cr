require "../../spec_helper"

# Returns whatever it is given, so the shared helpers can be tested without a
# model behind them.
private class RawEmbedder
  include Vecstolite::VectorEmbedder

  getter dimensions : Int32

  def initialize(@dimensions : Int32, @value : Float32 = 3.0_f32)
  end

  def model_name : String
    "test/raw"
  end

  def embed(text : String) : Vecstolite::Embedding
    Vecstolite::Embedding.new(@dimensions, @value * text.size)
  end
end

Spectator.describe Vecstolite::VectorEmbedder do
  private def magnitude(vector : Vecstolite::Embedding) : Float32
    Math.sqrt(vector.sum { |v| v * v }).to_f32
  end

  describe "#l2_normalize!" do
    it "scales a vector to unit length in place" do
      embedder = RawEmbedder.new(4)
      vector = embedder.embed("abc")
      expect(magnitude(vector)).to be > 1.0_f32

      returned = embedder.l2_normalize!(vector)
      expect(magnitude(vector)).to be_close(1.0_f32, 1e-6)
      expect(returned.to_a).to eq vector.to_a
    end

    it "leaves a zero vector alone" do
      embedder = RawEmbedder.new(4, 0.0_f32)
      vector = embedder.l2_normalize!(Vecstolite::Embedding.new(4, 0.0_f32))
      expect(vector.all? { |v| v == 0.0_f32 }).to be true
    end
  end

  describe "#embed_all" do
    it "defaults to embedding one at a time, in order" do
      embedder = RawEmbedder.new(3)
      results = embedder.embed_all(["a", "bb", "ccc"])

      expect(results.size).to eq 3
      expect(results[0][0]).to be < results[1][0]
      expect(results[1][0]).to be < results[2][0]
    end

    it "returns nothing for no texts" do
      expect(RawEmbedder.new(3).embed_all([] of String)).to be_empty
    end
  end

  describe "implementations" do
    it "returns unit vectors from LexicalEmbedder" do
      embedder = Vecstolite::LexicalEmbedder.new(64)
      expect(magnitude(embedder.embed("The sky is blue."))).to be_close(1.0_f32, 1e-5)
    end

    it "returns a zero vector when nothing tokenises" do
      embedder = Vecstolite::LexicalEmbedder.new(64)
      expect(magnitude(embedder.embed("12345 !!"))).to eq 0.0_f32
    end

    it "reports the dimensions it actually produces" do
      embedder = Vecstolite::LexicalEmbedder.new(64)
      expect(embedder.embed("The sky is blue.").size).to eq embedder.dimensions
    end
  end
end
