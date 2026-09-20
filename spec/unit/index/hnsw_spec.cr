require "../../spec_helper"

Spectator.describe Vecstolite::Index::HNSW do
  alias Repo = Vecstolite::Repository
  alias Cache = Vecstolite::NodeCache
  alias Flat = Vecstolite::Index::Flat
  alias Hnsw = Vecstolite::Index::HNSW

  let(dims) { 8 }

  # A deterministic unit vector, so a seeded run is reproducible.
  private def unit_vector(rng : Random, dims : Int32) : Vecstolite::Embedding
    values = Array(Float32).new(dims) { rng.rand(-1.0..1.0).to_f32 }
    norm = Math.sqrt(values.sum { |v| v * v }).to_f32
    Vecstolite::Embedding.new(dims) { |i| values[i] / norm }
  end

  # Seeds *count* entries and returns the repository with their ids.
  private def corpus(count : Int32, dims : Int32, seed : Int32 = 1) : {Repo, Array(Int64), Array(Vecstolite::Embedding)}
    repo = Repo.open(":memory:", dimensions: dims)
    rng = Random.new(seed)
    ids = [] of Int64
    vectors = [] of Vecstolite::Embedding
    count.times do |i|
      vector = unit_vector(rng, dims)
      vectors << vector
      ids << repo.insert_entry("entry #{i}", vector)
    end
    {repo, ids, vectors}
  end

  private def build(repo, ids, vectors, cache : Cache::Strategy, seed : Int32 = 42) : Hnsw
    index = Hnsw.new(cache, dims: vectors.first.size, m: 8, ef_construction: 64, seed: seed)
    ids.each_with_index { |entry_id, i| index.add(entry_id, vectors[i]) }
    index
  end

  describe "#add and #search" do
    it "finds an indexed vector exactly" do
      repo, ids, vectors = corpus(50, dims)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))

      hit = index.search(vectors[7], k: 1).first
      expect(hit.entry_id).to eq ids[7]
      expect(hit.score).to be_close(1.0_f32, 1e-4)
      repo.close
    end

    it "returns hits in descending score order" do
      repo, ids, vectors = corpus(50, dims)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))

      scores = index.search(vectors[3], k: 10).map(&.score)
      expect(scores.size).to eq 10
      expect(scores).to eq scores.sort.reverse
      repo.close
    end

    it "reports entry ids, never graph positions" do
      repo, ids, vectors = corpus(20, dims)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))

      found = index.search(vectors[0], k: 5).map(&.entry_id)
      expect(found.all? { |id| ids.includes?(id) }).to be true
      repo.close
    end

    it "honours an allowed set" do
      repo, ids, vectors = corpus(50, dims)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))

      allowed = Set{ids[10], ids[20], ids[30]}
      hits = index.search(vectors[10], k: 3, allowed: allowed)
      expect(hits.map(&.entry_id).to_set.subset_of?(allowed)).to be true
      repo.close
    end

    it "returns nothing before anything is added" do
      repo = Repo.open(":memory:", dimensions: dims)
      index = Hnsw.new(Cache::Memory.new(repo), dims: dims, seed: 42)

      expect(index.search(Vecstolite::Embedding.new(dims) { 0.5_f32 }, k: 5)).to be_empty
      expect(index.size).to eq 0
      repo.close
    end

    it "rejects a wrong-dimension vector" do
      repo = Repo.open(":memory:", dimensions: dims)
      index = Hnsw.new(Cache::Memory.new(repo), dims: dims, seed: 42)

      expect {
        index.add(1_i64, Vecstolite::Embedding.new(3) { 0.5_f32 })
      }.to raise_error(ArgumentError, /dims/)
      repo.close
    end
  end

  describe "graph state" do
    it "tracks an entry point and height" do
      repo, ids, vectors = corpus(30, dims)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))

      expect(index.entry_point).to be >= 0
      expect(index.max_layer).to be >= 0
      expect(index.size).to eq 30
      expect(index.kind).to eq :hnsw
      repo.close
    end

    it "clears back to an empty graph" do
      repo, ids, vectors = corpus(30, dims)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))
      index.flush
      expect(repo.node_count).to eq 30

      index.clear
      expect(index.size).to eq 0
      expect(index.entry_point).to eq(-1)
      expect(repo.node_count).to eq 0
      repo.close
    end

    it "restores graph metadata without rebuilding" do
      repo, ids, vectors = corpus(30, dims)
      writer = build(repo, ids, vectors, Cache::Memory.new(repo))
      writer.flush
      entry_point = writer.entry_point
      max_layer = writer.max_layer

      reader_cache = Cache::Memory.new(repo)
      reader_cache.load
      reader = Hnsw.new(reader_cache, dims: dims, m: 8, ef_construction: 64, seed: 42)
      reader.reset_with(entry_point, max_layer)

      expect(reader.search(vectors[5], k: 1).first.entry_id).to eq ids[5]
      repo.close
    end
  end

  describe "cache modes" do
    # The graph must not depend on where nodes are held.
    {% for mode in [:memory, :lru, :disk] %}
      it "builds an equivalent graph with the {{ mode.id }} cache" do
        repo, ids, vectors = corpus(80, dims)
        cache =
          case {{ mode }}
          when :memory then Cache::Memory.new(repo).as(Cache::Strategy)
          when :lru    then Cache::LRU.new(repo, 4_i64 * (dims * 4 + 128)).as(Cache::Strategy)
          else              Cache::Disk.new(repo).as(Cache::Strategy)
          end
        index = build(repo, ids, vectors, cache)

        expect(index.search(vectors[11], k: 1).first.entry_id).to eq ids[11]
        expect(index.size).to eq 80
        repo.close
      end
    {% end %}
  end

  describe "recall against Flat" do
    # Flat is exact, so it defines the right answer. This is the harness that
    # turns future tuning into a measurement rather than an argument.
    it "agrees with an exact scan on most queries" do
      repo, ids, vectors = corpus(300, dims, seed: 5)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))
      flat = Flat.new(repo)

      k = 10
      queries = 20
      matched = 0
      queries.times do |i|
        query = vectors[i * 7]
        exact = flat.search(query, k: k).map(&.entry_id).to_set
        approx = index.search(query, k: k, ef: 64).map(&.entry_id).to_set
        matched += (exact & approx).size
      end

      recall = matched / (queries * k).to_f
      expect(recall).to be >= 0.85
      repo.close
    end

    it "returns the true nearest neighbour for its own vectors" do
      repo, ids, vectors = corpus(300, dims, seed: 5)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))

      hits = 0
      20.times do |i|
        hits += 1 if index.search(vectors[i * 11], k: 1).first.entry_id == ids[i * 11]
      end
      expect(hits).to eq 20
      repo.close
    end
  end
end
