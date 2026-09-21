require "../../spec_helper"

private record Lang, code : String do
  include JSON::Serializable
end

private record Pair, en : String, fr : String do
  include JSON::Serializable
end

# Volume tests: the behaviours that only appear once a store is larger than
# its cache. The equivalence test below is what caught an LRU coherence bug
# that every small-corpus test passed straight through.
Spectator.describe "Vecstolite::Store at volume" do
  alias Store = Vecstolite::Store(Lang, Pair)
  alias CacheMode = Vecstolite::CacheMode

  let(embedder) { Vecstolite::LexicalEmbedder.new(dimensions: 64) }
  let(db_file_name) { File.tempname("vecsto_volume", ".db") }
  let(other_db_file_name) { File.tempname("vecsto_volume_other", ".db") }

  after_each do
    File.delete?(db_file_name)
    File.delete?(other_db_file_name)
  end

  # 500 deterministic sentences from a small set of templates.
  let(corpus) do
    subjects = ["The sky", "The ocean", "The grass", "The sun", "The moon",
                "The wind", "The rain", "The snow", "The forest", "The river"]
    verbs = ["is", "appears", "looks", "feels", "seems"]
    adjectives = ["blue", "green", "bright", "warm", "cold", "dark",
                  "deep", "clear", "vast", "calm"]
    Array(String).new(500) do |i|
      subject = subjects[i % subjects.size]
      verb = verbs[(i // subjects.size) % verbs.size]
      adjective = adjectives[(i // (subjects.size * verbs.size)) % adjectives.size]
      "#{subject} #{verb} #{adjective} (#{i})."
    end
  end

  # Budget for roughly ten nodes, so eviction runs constantly.
  let(tiny_budget) { 10_i64 * (64 * 4 + 128) }

  private def fill(store, corpus) : Nil
    store.bulk { |batch| corpus.each { |text| batch.add(text) } }
  end

  it "handles 500 entries under constant eviction" do
    store = Store.open(db_file_name, embedder,
      index: Vecstolite::Index.hnsw(seed: 7),
      cache: CacheMode.lru(tiny_budget))
    fill(store, corpus)

    expect(store.size).to eq 500
    results = store.search("blue sky", k: 10)
    expect(results.size).to eq 10
    results.each { |result| expect(corpus).to contain result.text }
    expect(store.stats[:cache_evictions]).to be > 0
    store.close
  end

  it "builds the same graph under eviction as with room to spare" do
    # Regression: a node evicted mid-insert could be read back with empty
    # neighbour lists, and a later write_back would persist that stale copy
    # over the real one, silently stripping edges from the graph.
    evicting = Store.open(db_file_name, embedder,
      index: Vecstolite::Index.hnsw(seed: 7),
      cache: CacheMode.lru(tiny_budget))
    fill(evicting, corpus)
    evicting_results = evicting.search("warm bright sun", k: 10).map(&.text)
    expect(evicting.stats[:cache_evictions]).to be > 0
    evicting.close

    roomy = Store.open(other_db_file_name, embedder,
      index: Vecstolite::Index.hnsw(seed: 7),
      cache: CacheMode.memory)
    fill(roomy, corpus)
    roomy_results = roomy.search("warm bright sun", k: 10).map(&.text)
    roomy.close

    expect(evicting_results).to eq roomy_results
  end

  it "mostly agrees with an exact scan on the same corpus" do
    # HNSW is approximate, and on this corpus many entries tie: sentences
    # differing only in "seems" and "feels" score identically against a query
    # sharing neither word. So the test compares scores, not which entry won —
    # an equally good answer that happens to be a different entry is correct.
    graph = Store.open(db_file_name, embedder,
      index: Vecstolite::Index.hnsw(seed: 7),
      cache: CacheMode.memory)
    fill(graph, corpus)
    graph_scores = graph.search("deep dark river", k: 5).map(&.score)
    graph.close

    exact = Store.open(other_db_file_name, embedder, index: Vecstolite::Index.flat)
    fill(exact, corpus)
    exact_scores = exact.search("deep dark river", k: 5).map(&.score)
    exact.close

    expect(graph_scores.first).to be_close(exact_scores.first, 1e-5)
    threshold = exact_scores.last - 1e-5_f32
    expect(graph_scores.count { |score| score >= threshold }).to be >= 4
  end

  it "deletes and compacts at volume" do
    store = Store.open(db_file_name, embedder,
      index: Vecstolite::Index.hnsw(seed: 7),
      cache: CacheMode.lru(tiny_budget))
    # Keyed by their own text, so specific entries can be deleted by name.
    store.bulk { |batch| corpus.each { |text| batch.add(text, key: text) } }

    doomed = corpus.each_slice(2).map(&.first).to_a
    doomed.each { |text| expect(store.delete_by_key(text)).to be true }

    expect(store.size).to eq 500 - doomed.size
    expect(store.tombstones).to eq doomed.size
    expect(store.search("blue sky", k: 5).map(&.text)).not_to contain doomed.first

    store.compact!
    expect(store.tombstones).to eq 0
    expect(store.size).to eq 500 - doomed.size
    expect(store.search("blue sky", k: 5).size).to eq 5
    store.close

    store = Store.open(db_file_name, embedder, cache: CacheMode.lru(tiny_budget))
    expect(store.size).to eq 500 - doomed.size
    expect(store.get_by_key(doomed.first)).to be_nil
    expect(store.search("deep dark river", k: 5).size).to eq 5
    store.close
  end
end
