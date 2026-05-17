require "../../spec_helper"

Spectator.describe Vecstolite::HNSW::NodeCache do
  # Helper: Create a simple node with given dims and layer count
  def make_node(dims : Int32, num_layers : Int32) : Vecstolite::HNSW::HNSWNode
    vector = Vecstolite::Embedding.new(dims, 0.0_f32)
    node = Vecstolite::HNSW::HNSWNode.new(vector, num_layers - 1, 16)
    node
  end

  let(dims) { 128 }
  let(num_layers) { 3 }

  describe "#put and #get?" do
    let(cache) { described_class.new(10_000) } # 10 KB budget

    it "stores and retrieves a node" do
      node = make_node(dims, num_layers)
      cache.put(0, node)

      retrieved = cache.get?(0)
      expect(retrieved).not_to be_nil
      expect(retrieved.not_nil!.vector.size).to eq(dims)
    end

    it "returns nil for uncached nodes" do
      expect(cache.get?(999)).to be_nil
    end

    it "updates access time on each get" do
      node = make_node(dims, num_layers)
      cache.put(0, node)

      # Access it
      cache.get?(0)
      expect(cache.cached?(0)).to be_true
    end
  end

  describe "#invalidate" do
    let(cache) { described_class.new(10_000) }

    it "removes a cached node" do
      node = make_node(dims, num_layers)
      cache.put(0, node)
      expect(cache.cached?(0)).to be_true

      cache.invalidate(0)
      expect(cache.cached?(0)).to be_false
    end

    it "decreases bytes_used" do
      node = make_node(dims, num_layers)
      cache.put(0, node)
      before = cache.bytes_used

      cache.invalidate(0)
      expect(cache.bytes_used).to be < before
    end
  end

  describe "#size" do
    let(cache) { described_class.new(100_000) }

    it "returns number of cached nodes" do
      5.times do |i|
        cache.put(i, make_node(dims, num_layers))
      end

      expect(cache.size).to eq(5)
    end

    it "decreases when nodes are invalidated" do
      cache.put(0, make_node(dims, num_layers))
      expect(cache.size).to eq(1)

      cache.invalidate(0)
      expect(cache.size).to eq(0)
    end
  end

  describe "#bytes_used" do
    let(cache) { described_class.new(100_000) }

    it "tracks cumulative size" do
      node = make_node(dims, num_layers)
      cache.put(0, node)
      first_size = cache.bytes_used

      cache.put(1, make_node(dims, num_layers))
      expect(cache.bytes_used).to be > first_size
    end

    it "decreases on invalidation" do
      cache.put(0, make_node(dims, num_layers))
      before = cache.bytes_used

      cache.invalidate(0)
      expect(cache.bytes_used).to be < before
    end
  end

  describe "LRU eviction" do
    let(small_cache) { described_class.new(1_000) } # Very tight budget

    it "evicts least-recently-used node when over budget" do
      # Add node 0
      small_cache.put(0, make_node(64, 2))
      expect(small_cache.cached?(0)).to be_true

      # Add node 1, forcing eviction (cache is small)
      small_cache.put(1, make_node(64, 2))

      # Access node 0 to make it recent, then add node 2
      small_cache.get?(0)
      small_cache.put(2, make_node(64, 2))

      # Node 1 should be evicted (least recent)
      expect(small_cache.cached?(1)).to be_false
      expect(small_cache.cached?(0)).to be_true
      expect(small_cache.cached?(2)).to be_true
    end

    it "does not evict pinned nodes" do
      cache = described_class.new(1_000, pinned_ids: [0])
      small_cache.pin(0)

      small_cache.put(0, make_node(64, 2))
      small_cache.put(1, make_node(64, 2))
      small_cache.put(2, make_node(64, 2))

      # Node 0 is pinned, so it should never be evicted
      expect(small_cache.cached?(0)).to be_true
    end
  end

  describe "#pin and #unpin" do
    let(cache) { described_class.new(10_000) }

    it "pins a node so it's not evicted" do
      cache.pin(0)
      cache.put(0, make_node(dims, num_layers))

      # Fill cache to trigger eviction
      100.times do |i|
        cache.put(i + 1, make_node(32, 1))
      end

      # Node 0 should still be there (pinned)
      expect(cache.cached?(0)).to be_true
    end

    it "unpins a node" do
      cache.pin(0)
      expect { cache.unpin(0) }.not_to raise_error
    end
  end

  describe "#clear_unpinned" do
    let(cache) { described_class.new(100_000) }

    it "clears all unpinned nodes" do
      cache.put(0, make_node(dims, num_layers))
      cache.put(1, make_node(dims, num_layers))
      cache.put(2, make_node(dims, num_layers))
      expect(cache.size).to eq(3)

      cache.clear_unpinned
      expect(cache.size).to eq(0)
    end

    it "preserves pinned nodes" do
      cache.pin(0)
      cache.put(0, make_node(dims, num_layers))
      cache.put(1, make_node(dims, num_layers))
      cache.put(2, make_node(dims, num_layers))

      cache.clear_unpinned
      expect(cache.cached?(0)).to be_true
      expect(cache.cached?(1)).to be_false
      expect(cache.cached?(2)).to be_false
    end
  end

  describe "#cached?" do
    let(cache) { described_class.new(10_000) }

    it "returns true for cached nodes" do
      cache.put(0, make_node(dims, num_layers))
      expect(cache.cached?(0)).to be_true
    end

    it "returns false for non-cached nodes" do
      expect(cache.cached?(999)).to be_false
    end
  end
end
