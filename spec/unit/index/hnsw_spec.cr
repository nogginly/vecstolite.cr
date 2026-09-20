require "../../spec_helper"

Spectator.describe Vecstolite::Index::HNSW do
  alias Repo = Vecstolite::Repository
  alias Cache = Vecstolite::NodeCache
  alias Flat = Vecstolite::Index::Flat
  alias Hnsw = Vecstolite::Index::HNSW

  let(dims) { 8 }

  # Recall floor for the HNSW-vs-Flat harness. Measured baseline is 0.976
  # (2,000 vectors, 32 dimensions, 50 clusters at spread 0.15, m=8,
  # ef_construction=64, ef=5, k=5, corpus seed 5, graph seed 42). Both seeds
  # are fixed, so this is close to deterministic; the floor sits just below
  # the baseline to catch a real regression rather than float noise.
  #
  # How it got here, since the two knobs turned out to be coupled:
  #   0.752  nearest-m neighbours, layer probability 1/e
  #   0.572  nearest-m neighbours, layer probability 1/m
  #   0.936  diversity neighbours, layer probability 1/e
  #   0.976  diversity neighbours, layer probability 1/m
  # The middle row is why the layer draw cannot be judged on its own: without
  # diverse neighbours, the surplus upper layers were carrying the routing.
  RECALL_FLOOR = 0.94

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

  describe "graph shape" do
    it "puts roughly one node in m above layer 0" do
      # The layer draw decides how tall the graph is. At p = 1/m about an
      # eighth of nodes reach layer 1 with m = 8; an earlier p = 1/e put
      # better than a third of them there, costing a beam search and back-edge
      # writes on every insert without improving routing.
      repo, ids, vectors = corpus(1_000, dims, seed: 3)
      cache = Cache::Memory.new(repo)
      build(repo, ids, vectors, cache)

      above_zero = 0
      cache.each_node { |_, _, node| above_zero += 1 if node.neighbours.size > 1 }
      fraction = above_zero / 1_000.0

      expect(fraction).to be < 0.25
      expect(fraction).to be > 0.02
      repo.close
    end
  end

  describe "neighbour selection" do
    it "keeps every node within its neighbour limit" do
      repo, ids, vectors = corpus(200, dims, seed: 9)
      cache = Cache::Memory.new(repo)
      build(repo, ids, vectors, cache)

      cache.each_node do |_, _, node|
        node.neighbours.each_with_index do |layer_neighbours, layer|
          limit = layer == 0 ? 16 : 8
          expect(layer_neighbours.size).to be <= limit
        end
      end
      repo.close
    end

    it "gives nodes edges that leave their own cluster" do
      # With nearest-m selection, a node in a dense cluster keeps only edges
      # pointing inward and a search arriving there cannot get out. Built from
      # two well-separated clusters, some node must bridge them.
      repo = Repo.open(":memory:", dimensions: dims)
      rng = Random.new(11)
      ids = [] of Int64
      vectors = [] of Vecstolite::Embedding
      100.times do |i|
        base = Array(Float32).new(dims) { |d| (d < dims // 2) == (i.even?) ? 1.0_f32 : 0.0_f32 }
        values = base.map { |b| (b + rng.rand(-0.15..0.15)).to_f32 }
        norm = Math.sqrt(values.sum { |v| v * v }).to_f32
        vector = Vecstolite::Embedding.new(dims) { |d| values[d] / norm }
        vectors << vector
        ids << repo.insert_entry("entry #{i}", vector)
      end

      cache = Cache::Memory.new(repo)
      index = Hnsw.new(cache, dims: dims, m: 8, ef_construction: 64, seed: 42)
      ids.each_with_index { |entry_id, i| index.add(entry_id, vectors[i]) }

      # Each node's cluster is decided by the parity of its ord, since entries
      # were inserted alternating between the two.
      crossings = 0
      cache.each_node do |ord, _, node|
        node.neighbours[0].each do |nb_ord|
          crossings += 1 if nb_ord.even? != ord.even?
        end
      end
      expect(crossings).to be > 0
      repo.close
    end
  end

  describe "recall against Flat" do
    # Flat is exact, so it defines the right answer. Three things make this
    # harness discriminating, and all three matter:
    #
    # 1. Held-out queries. A query that is itself in the graph is trivially
    #    found — the descent lands on it and its neighbours are its true
    #    nearest neighbours. Corpus members score 1.0 and prove nothing.
    # 2. Clustered vectors. Uniformly random points are the easy case for a
    #    navigable graph; real embeddings sit in clusters, which is where
    #    neighbour selection is tested.
    # 3. A beam far narrower than the corpus.
    let(recall_dims) { 32 }
    let(recall_corpus) { 2_000 }
    # The effective beam is max(ef, k), so narrowing the search means lowering
    # both. With diversity-based neighbour selection the index scores a flat
    # 1.0 at k=10, ef=16 — a saturated harness proves nothing, so this one is
    # deliberately harder.
    let(recall_k) { 5 }
    let(recall_ef) { 5 }
    let(recall_queries) { 25 }

    # A vector near *centre*, normalised. Small *spread* means tight clusters.
    private def near(rng : Random, centre : Array(Float32), spread : Float64) : Vecstolite::Embedding
      values = centre.map { |c| (c + rng.rand(-spread..spread)).to_f32 }
      norm = Math.sqrt(values.sum { |v| v * v }).to_f32
      Vecstolite::Embedding.new(values.size) { |i| values[i] / norm }
    end

    # Clustered corpus plus held-out queries drawn from the same clusters but
    # never inserted.
    private def clustered(count : Int32, dims : Int32, clusters : Int32, queries : Int32, seed : Int32)
      rng = Random.new(seed)
      centres = Array.new(clusters) { Array(Float32).new(dims) { rng.rand(-1.0..1.0).to_f32 } }

      repo = Repo.open(":memory:", dimensions: dims)
      ids = [] of Int64
      vectors = [] of Vecstolite::Embedding
      count.times do |i|
        vector = near(rng, centres[i % clusters], 0.15)
        vectors << vector
        ids << repo.insert_entry("entry #{i}", vector)
      end

      held_out = Array.new(queries) { |i| near(rng, centres[i % clusters], 0.15) }
      {repo, ids, vectors, held_out}
    end

    private def measure_recall(repo, ids, vectors, held_out, index) : Float64
      flat = Flat.new(repo)
      matched = 0
      held_out.each do |query|
        exact = flat.search(query, k: recall_k).map(&.entry_id).to_set
        approx = index.search(query, k: recall_k, ef: recall_ef).map(&.entry_id).to_set
        matched += (exact & approx).size
      end
      matched / (held_out.size * recall_k).to_f
    end

    it "agrees with an exact scan on most queries" do
      repo, ids, vectors, held_out =
        clustered(recall_corpus, recall_dims, clusters: 50, queries: recall_queries, seed: 5)
      index = build(repo, ids, vectors, Cache::Memory.new(repo))

      recall = measure_recall(repo, ids, vectors, held_out, index)
      expect(recall).to be >= RECALL_FLOOR

      # A harness that scores a perfect 1.0 has stopped measuring anything: it
      # cannot register a regression. This has already happened twice, once
      # with corpus members as queries and once with a beam wider than the
      # index needed. If this assertion fails, harden the configuration —
      # narrower beam, tighter clusters, larger corpus — and recalibrate
      # RECALL_FLOOR against the new figure. Do not delete it.
      expect(recall).to be < 1.0
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
