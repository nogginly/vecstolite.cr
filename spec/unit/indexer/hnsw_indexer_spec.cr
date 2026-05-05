require "../../spec_helper"

Spectator.describe Vecstolite::HNSW::Index do
  # Build a minimal normalised vector of given dims with a 1.0 at position `hot`.
  def one_hot(hot : Int32, dims : Int32) : Vecstolite::Embedding
    v = Vecstolite::Embedding.new(dims, 0.0_f32)
    v[hot] = 1.0_f32
    v
  end

  let(dims) { 8 }
  let(m) { 4 }
  let(ef_construction) { 20 }

  # Build a small index with `count` one-hot vectors and a fixed seed for reproducibility.
  def build_index(count : Int32, dims : Int32, m : Int32, ef_construction : Int32) : Vecstolite::HNSW::Index
    index = Vecstolite::HNSW::Index.new(dims: dims, m: m, ef_construction: ef_construction, seed: 42)
    count.times do |i|
      index.add(id: i, vector: one_hot(i % dims, dims))
    end
    index
  end

  describe "#export" do
    it "yields every node id sequentially" do
      index = build_index(4, dims, m, ef_construction)
      seen = [] of Int32
      index.export { |id, _| seen << id }
      expect(seen).to eq([0, 1, 2, 3])
    end

    it "returns entry_point and max_layer in metadata" do
      index = build_index(4, dims, m, ef_construction)
      meta = index.export { }
      expect(meta[:entry_point]).to be >= 0
      expect(meta[:max_layer]).to be >= 0
    end

    it "yields neighbour lists as Array(Array(Int32))" do
      index = build_index(4, dims, m, ef_construction)
      index.export do |_id, neighbours|
        expect(neighbours).to be_a(Array(Array(Int32)))
      end
    end
  end

  describe ".restore" do
    it "produces an index with the same size as the original" do
      original = build_index(4, dims, m, ef_construction)

      vectors = [] of Vecstolite::Embedding
      neighbours = [] of Array(Array(Int32))
      meta = original.export do |id, nb|
        vectors << one_hot(id % dims, dims)
        neighbours << nb
      end

      restored = Vecstolite::HNSW::Index.restore(
        dims: dims, m: m, ef_construction: ef_construction,
        entry_point: meta[:entry_point], max_layer: meta[:max_layer],
        node_count: 4
      ) do |id|
        {vectors[id], neighbours[id]}
      end

      expect(restored.size).to eq(original.size)
    end

    it "produces search results matching the original index" do
      original = build_index(4, dims, m, ef_construction)
      query = one_hot(0, dims)
      original_results = original.search(query, k: 2)

      vectors = [] of Vecstolite::Embedding
      neighbours = [] of Array(Array(Int32))
      meta = original.export do |id, nb|
        vectors << one_hot(id % dims, dims)
        neighbours << nb
      end

      restored = Vecstolite::HNSW::Index.restore(
        dims: dims, m: m, ef_construction: ef_construction,
        entry_point: meta[:entry_point], max_layer: meta[:max_layer],
        node_count: 4
      ) do |id|
        {vectors[id], neighbours[id]}
      end

      restored_results = restored.search(query, k: 2)

      expect(restored_results.map(&.id)).to eq(original_results.map(&.id))
      expect(restored_results.map(&.score)).to eq(original_results.map(&.score))
    end

    it "export → restore round-trip is symmetric" do
      original = build_index(6, dims, m, ef_construction)

      vectors = [] of Vecstolite::Embedding
      neighbours = [] of Array(Array(Int32))
      meta = original.export do |id, nb|
        vectors << one_hot(id % dims, dims)
        neighbours << nb
      end

      restored = Vecstolite::HNSW::Index.restore(
        dims: dims, m: m, ef_construction: ef_construction,
        entry_point: meta[:entry_point], max_layer: meta[:max_layer],
        node_count: 6
      ) do |id|
        {vectors[id], neighbours[id]}
      end

      # Re-export the restored index and compare metadata.
      restored_meta = restored.export { }
      expect(restored_meta[:entry_point]).to eq(meta[:entry_point])
      expect(restored_meta[:max_layer]).to eq(meta[:max_layer])
    end
  end
end
