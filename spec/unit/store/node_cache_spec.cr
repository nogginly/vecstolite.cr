require "../../spec_helper"

Spectator.describe Vecstolite::NodeCache do
  alias Repo = Vecstolite::Repository
  alias Cache = Vecstolite::NodeCache

  let(dims) { 4 }

  private def vec(value : Float32) : Vecstolite::Embedding
    Vecstolite::Embedding.new(4) { value }
  end

  private def node_for(value : Float32, neighbours : Array(Array(Int32))) : Vecstolite::Index::Node
    node = Vecstolite::Index::Node.new(vec(value), 0)
    node.neighbours = neighbours
    node
  end

  # Seeds *repo* with *count* entries and returns their ids.
  private def seed_entries(repo, count) : Array(Int64)
    Array(Int64).new(count) do |i|
      repo.insert_entry("text #{i}", vec(i.to_f32))
    end
  end

  # Every strategy must behave identically for these; only durability and
  # memory use differ.
  {% for mode in [:memory, :lru, :disk] %}
    describe "{{ mode.id }} strategy" do
      it "appends, reads back and reports size" do
        repo = Repo.open(":memory:", dimensions: dims)
        ids = seed_entries(repo, 3)
        cache = build_cache(repo, {{ mode }})

        ids.each_with_index do |entry_id, i|
          ord = cache.append(node_for(i.to_f32, [[] of Int32]), entry_id)
          expect(ord).to eq i
        end

        expect(cache.size).to eq 3
        expect(cache.get(1).vector.to_a).to eq vec(1_f32).to_a
        expect(cache.entry_id(2)).to eq ids[2]
        repo.close
      end

      it "persists mutated neighbours" do
        repo = Repo.open(":memory:", dimensions: dims)
        ids = seed_entries(repo, 2)
        cache = build_cache(repo, {{ mode }})
        ids.each_with_index { |entry_id, i| cache.append(node_for(i.to_f32, [[] of Int32]), entry_id) }

        node = cache.get(0)
        node.neighbours[0] << 1
        cache.write_back(0, node)

        expect(cache.get(0).neighbours[0]).to eq [1]
        repo.close
      end

      it "iterates every node with its entry id" do
        repo = Repo.open(":memory:", dimensions: dims)
        ids = seed_entries(repo, 3)
        cache = build_cache(repo, {{ mode }})
        ids.each_with_index { |entry_id, i| cache.append(node_for(i.to_f32, [[] of Int32]), entry_id) }

        seen = [] of {Int32, Int64}
        cache.each_node { |ord, entry_id, _| seen << {ord, entry_id} }
        expect(seen).to eq [{0, ids[0]}, {1, ids[1]}, {2, ids[2]}]
        repo.close
      end
    end
  {% end %}

  # ---------------------------------------------------------------------------
  # strategy-specific behaviour
  # ---------------------------------------------------------------------------

  describe Cache::Memory do
    it "holds the graph until flushed" do
      repo = Repo.open(":memory:", dimensions: dims)
      ids = seed_entries(repo, 2)
      cache = Cache::Memory.new(repo)
      ids.each_with_index { |entry_id, i| cache.append(node_for(i.to_f32, [[] of Int32]), entry_id) }

      expect(cache.fully_persisted?).to be false
      expect(repo.node_count).to eq 0

      cache.flush
      expect(repo.node_count).to eq 2
      repo.close
    end

    it "reloads a persisted graph" do
      repo = Repo.open(":memory:", dimensions: dims)
      ids = seed_entries(repo, 2)
      writer = Cache::Memory.new(repo)
      ids.each_with_index { |entry_id, i| writer.append(node_for(i.to_f32, [[1 - i]]), entry_id) }
      writer.flush

      reader = Cache::Memory.new(repo)
      reader.load
      expect(reader.size).to eq 2
      expect(reader.get(0).neighbours).to eq [[1]]
      expect(reader.entry_id(1)).to eq ids[1]
      repo.close
    end
  end

  describe Cache::Disk do
    it "writes through on append and never caches" do
      repo = Repo.open(":memory:", dimensions: dims)
      ids = seed_entries(repo, 1)
      cache = Cache::Disk.new(repo)
      cache.append(node_for(0_f32, [[] of Int32]), ids[0])

      expect(cache.fully_persisted?).to be true
      expect(repo.node_count).to eq 1
      expect(cache.stats[:cached]).to eq 0
      repo.close
    end

    it "picks up a graph written before it existed" do
      repo = Repo.open(":memory:", dimensions: dims)
      ids = seed_entries(repo, 2)
      first = Cache::Disk.new(repo)
      ids.each_with_index { |entry_id, i| first.append(node_for(i.to_f32, [[] of Int32]), entry_id) }

      second = Cache::Disk.new(repo)
      expect(second.size).to eq 2
      repo.close
    end

    it "raises for an unknown node" do
      repo = Repo.open(":memory:", dimensions: dims)
      cache = Cache::Disk.new(repo)
      expect { cache.get(0) }.to raise_error(Repo::Error, /not found/)
      repo.close
    end
  end

  describe Cache::LRU do
    # Budget for roughly two nodes, so appends evict as they go.
    let(tiny_budget) { 2_i64 * (dims * 4 + 128) }

    it "evicts under a small budget but keeps reads correct" do
      repo = Repo.open(":memory:", dimensions: dims)
      ids = seed_entries(repo, 10)
      cache = Cache::LRU.new(repo, tiny_budget)
      ids.each_with_index { |entry_id, i| cache.append(node_for(i.to_f32, [[] of Int32]), entry_id) }

      expect(cache.stats[:evictions]).to be > 0
      expect(cache.get(0).vector.to_a).to eq vec(0_f32).to_a
      expect(cache.stats[:misses]).to be > 0
      repo.close
    end

    it "serves a cached node without a repository read" do
      repo = Repo.open(":memory:", dimensions: dims)
      ids = seed_entries(repo, 1)
      cache = Cache::LRU.new(repo, tiny_budget)
      cache.append(node_for(0_f32, [[] of Int32]), ids[0])

      cache.get(0)
      expect(cache.stats[:hits]).to eq 1
      repo.close
    end

    it "does not let a stale copy overwrite a written node" do
      # Regression: holding a node across an eviction, then reading the same
      # ord back, used to leave two objects for one node. Writing the older
      # one last stripped the neighbours of the newer.
      repo = Repo.open(":memory:", dimensions: dims)
      ids = seed_entries(repo, 4)
      cache = Cache::LRU.new(repo, tiny_budget)
      ids.each_with_index { |entry_id, i| cache.append(node_for(i.to_f32, [[] of Int32]), entry_id) }

      held = cache.get(0)
      held.neighbours[0] << 3
      cache.get(1) # evicts ord 0 under the tiny budget
      cache.get(2)
      cache.get(0) # reads a second, still-empty copy of ord 0 into the cache
      cache.write_back(0, held)

      later = cache.get(0)
      expect(later.neighbours[0]).to contain 3

      later.neighbours[0] << 99
      cache.write_back(0, later)
      expect(repo.node(0).not_nil!.neighbours[0]).to eq [3, 99]
      repo.close
    end
  end

  # ---------------------------------------------------------------------------
  private def build_cache(repo, mode : Symbol) : Vecstolite::NodeCache::Strategy
    case mode
    when :memory then Cache::Memory.new(repo)
    when :lru    then Cache::LRU.new(repo, 1_i64 * 1024 * 1024)
    else              Cache::Disk.new(repo)
    end
  end
end
