require "../../spec_helper"

Spectator.describe Vecstolite::HNSW::CachedNodeProvider do
  # Helper: Create a simple test node
  def make_node(id : Int32, dims : Int32 = 64, layers : Int32 = 2) : Vecstolite::HNSW::HNSWNode
    vector = Vecstolite::Embedding.new(dims, id.to_f32)
    Vecstolite::HNSW::HNSWNode.new(vector, layers - 1, 16)
  end

  let(dims) { 64 }
  let(cache_budget) { 10_000 }
  let(cache) { Vecstolite::HNSW::NodeCache.new(cache_budget) }

  describe "#get with mock loader" do
    it "loads a node via the loader on first access" do
      load_count = 0
      loader = ->(id : Int32) do
        load_count += 1
        make_node(id, dims)
      end

      provider = described_class.new(cache, &loader)
      node = provider.get(0)

      expect(node).not_to be_nil
      expect(load_count).to eq(1)
    end

    it "caches the loaded node" do
      loader = ->(id : Int32) { make_node(id, dims) }
      provider = described_class.new(cache, &loader)

      provider.get(0)
      expect(cache.cached?(0)).to be_true
    end

    it "returns cached node on subsequent access without calling loader" do
      load_count = 0
      loader = ->(id : Int32) do
        load_count += 1
        make_node(id, dims)
      end

      provider = described_class.new(cache, &loader)
      node1 = provider.get(0)
      node2 = provider.get(0)

      expect(load_count).to eq(1) # Loader called only once
      expect(node1.vector[0]).to eq(node2.vector[0])
    end

    it "loads different nodes on different IDs" do
      loaded_ids = [] of Int32
      loader = ->(id : Int32) do
        loaded_ids << id
        make_node(id, dims)
      end

      provider = described_class.new(cache, &loader)
      provider.get(0)
      provider.get(1)
      provider.get(2)

      expect(loaded_ids).to eq([0, 1, 2])
    end
  end

  describe "cache hit vs miss" do
    it "hits cache on repeated access" do
      load_count = 0
      loader = ->(id : Int32) do
        load_count += 1
        make_node(id, dims)
      end

      provider = described_class.new(cache, &loader)

      # First access: miss
      provider.get(0)
      expect(load_count).to eq(1)

      # Subsequent accesses: hits
      provider.get(0)
      provider.get(0)
      expect(load_count).to eq(1) # Still 1, no new loads
    end

    it "updates LRU on cache hit" do
      small_cache = Vecstolite::HNSW::NodeCache.new(1_000) # Conservative budget
      loader = ->(id : Int32) { make_node(id, dims) }
      provider = described_class.new(small_cache, &loader)

      # Load node 0
      provider.get(0)
      expect(small_cache.cached?(0)).to be_true

      # Load node 1
      provider.get(1)
      expect(small_cache.cached?(1)).to be_true

      # Load node 2 — should evict the oldest (node 0, which has never been re-accessed)
      provider.get(2)

      # Node 0 should be evicted (oldest)
      expect(small_cache.cached?(0)).to be_false
      # Nodes 1 and 2 should remain
      expect(small_cache.cached?(1)).to be_true
      expect(small_cache.cached?(2)).to be_true
    end
  end

  describe "loader exception handling" do
    it "propagates loader exceptions" do
      loader = ->(id : Int32) do
        raise Exception.new("Load error for id #{id}")
        make_node(id, dims) # Unreachable, but satisfies type checker
      end

      provider = described_class.new(cache, &loader)
      expect { provider.get(0) }.to raise_error(Exception)
    end
  end

  describe "with eviction" do
    it "calls loader for evicted nodes on re-access" do
      small_cache = Vecstolite::HNSW::NodeCache.new(1_200) # Fits exactly 2 nodes
      load_calls = [] of Int32

      loader = ->(id : Int32) do
        load_calls << id
        make_node(id, dims)
      end

      provider = described_class.new(small_cache, &loader)

      # Load nodes 0 and 1 (both fit: 2 × 476 = 952 < 1200)
      provider.get(0)
      provider.get(1)

      # Load node 2 to force eviction of node 0 (oldest)
      provider.get(2)

      # Node 0 should have been evicted
      expect(small_cache.cached?(0)).to be_false

      # Accessing it again should reload via loader
      provider.get(0)
      expect(load_calls.count(0)).to eq(2) # Loaded twice (once + reload)
    end
  end

  describe "pinned nodes" do
    it "never evicts pinned nodes" do
      small_cache = Vecstolite::HNSW::NodeCache.new(1_500)
      small_cache.pin(0) # Pin entry point

      loader = ->(id : Int32) { make_node(id, dims) }
      provider = described_class.new(small_cache, &loader)

      provider.get(0)
      # Fill cache with additional nodes
      5.times { |i| provider.get(i + 1) }

      # Pinned node 0 should still be cached
      expect(small_cache.cached?(0)).to be_true
    end
  end

  describe "concurrent node access" do
    it "handles accessing many nodes without duplication" do
      load_calls = {} of Int32 => Int32 # id -> count
      loader = ->(id : Int32) do
        load_calls[id] = (load_calls[id]? || 0) + 1
        make_node(id, dims)
      end

      provider = described_class.new(cache, &loader)

      # Access nodes in various orders
      [0, 1, 2, 1, 0, 3, 2, 1, 0].each { |id| provider.get(id) }

      # Each node loaded exactly once
      expect(load_calls.values.all? { |count| count == 1 }).to be_true
    end
  end
end
