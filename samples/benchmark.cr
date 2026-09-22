require "../src/vecstolite"

require "json"
require "benchmark"

# Benchmarks behind the sizing decisions in DESIGN.md §13.
#
#   crystal run samples/benchmark.cr --release -- [suite] [options]
#
# Suites: cache, footprint, ingest, restart, crossover, duplicates, compact, all
#
#   --sizes 1000,10000    corpus sizes to sweep
#   --dims 768            embedding width for the synthetic embedder
#   --model <dir>         use StaticEmbedder from a model directory instead
#
# Results print as markdown tables, for pasting into DEVELOPMENT.md with the
# machine and Crystal version recorded alongside.
module Benchmarks
  alias Store = Vecstolite::Store(String, String)
  alias CacheMode = Vecstolite::CacheMode

  DB_PATH = "tmp_benchmark.db"
  QUERIES = 50
  TOP_K   =  5

  # Graph construction draws node layers at random, so an unseeded run builds
  # a different graph every time and recall wanders by several points between
  # runs. Fixed here so results are comparable with each other.
  SEED = 42

  # Deterministic sentences, each with a unique tag, so runs are comparable
  # and no two entries embed identically.
  #
  # The tag is spelled in letters because LexicalEmbedder keeps letters only:
  # an earlier version used "(12345)", which the tokenizer dropped, so a
  # 100,000-entry corpus collapsed to 11,760 distinct vectors with up to ten
  # exact copies each — and measured duplicates rather than scale.
  def self.corpus(count : Int32) : Array(String)
    subjects = ["The sky", "The ocean", "The grass", "The sun", "The moon",
                "The wind", "The rain", "The snow", "The forest", "The river",
                "A mountain", "A valley", "A desert", "A meadow", "A glacier"]
    verbs = ["is", "appears", "looks", "feels", "seems", "remains", "turns"]
    adjectives = ["blue", "green", "bright", "warm", "cold", "dark", "deep",
                  "clear", "vast", "calm", "harsh", "still", "pale", "golden"]
    contexts = ["at dawn", "at dusk", "in winter", "in summer", "after rain",
                "before the storm", "under cloud", "in the distance"]

    Array(String).new(count) do |i|
      "#{subjects[i % subjects.size]} #{verbs[(i // subjects.size) % verbs.size]} " \
      "#{adjectives[(i // 7) % adjectives.size]} #{contexts[(i // 13) % contexts.size]} " \
      "ref #{letters(i)}."
    end
  end

  # *n* in base 26, as lowercase letters: 0 is "a", 26 is "ba".
  def self.letters(n : Int32) : String
    String.build do |io|
      digits = [] of Char
      loop do
        digits << ('a' + n % 26)
        n //= 26
        break if n == 0
      end
      digits.reverse_each { |digit| io << digit }
    end
  end

  # Queries near the corpus rather than in it: an exact member is trivially
  # found and measures nothing.
  def self.queries(texts : Array(String)) : Array(String)
    Array(String).new(QUERIES) do |i|
      words = texts[(i * 97) % texts.size].split(' ')
      words.first([words.size - 2, 3].max).join(' ')
    end
  end

  # The default index, seeded for comparable runs.
  def self.graph : Vecstolite::Index::Config
    Vecstolite::Index.hnsw(seed: SEED)
  end

  def self.reset_db : Nil
    Dir.glob("#{DB_PATH}*") { |file| File.delete?(file) }
  end

  def self.db_bytes : Int64
    Dir.glob("#{DB_PATH}*").sum { |file| File.size(file).to_i64 }
  end

  def self.mb(bytes : Int64) : String
    "%.1f MB" % (bytes / 1_048_576.0)
  end

  def self.ms(span : Time::Span) : String
    "%.2f ms" % span.total_milliseconds
  end

  def self.fill(store, texts : Array(String)) : Nil
    store.bulk { |batch| texts.each { |text| batch.add(text) } }
  end

  def self.search_all(store, probes : Array(String)) : Time::Span
    Time.measure { probes.each { |query| store.search(query, k: TOP_K) } }
  end

  # ---------------------------------------------------------------------------
  # 1. Cache sizing — what a given budget actually buys.
  # ---------------------------------------------------------------------------
  def self.cache(embedder, sizes : Array(Int32)) : Nil
    puts "\n## Cache sizing"
    puts
    puts "| corpus | budget | hit rate | #{QUERIES} queries | per query |"
    puts "|-------:|--------|---------:|-------------------:|----------:|"

    budgets = {
      "0.5 MB" => CacheMode.lru(512_i64 * Vecstolite::KB),
      "5 MB"   => CacheMode.lru(5_i64 * Vecstolite::MB),
      "50 MB"  => CacheMode.lru(50_i64 * Vecstolite::MB),
      "all"    => CacheMode.memory,
    }

    sizes.each do |size|
      texts = corpus(size)
      probes = queries(texts)

      budgets.each do |label, mode|
        reset_db
        Store.open(DB_PATH, embedder, index: graph, cache: mode) { |store| fill(store, texts) }

        store = Store.open(DB_PATH, embedder, index: graph, cache: mode)
        search_all(store, probes.first(5))
        elapsed = search_all(store, probes)
        stats = store.stats
        looked_up = stats[:cache_hits] + stats[:cache_misses]
        rate = looked_up > 0 ? "%.1f%%" % (100.0 * stats[:cache_hits] / looked_up) : "n/a"
        store.close

        puts "| #{size} | #{label} | #{rate} | #{ms(elapsed)} | #{ms(elapsed / QUERIES)} |"
      end
    end
    reset_db
  end

  # ---------------------------------------------------------------------------
  # 2. Footprint — how many instances fit on one machine.
  # ---------------------------------------------------------------------------
  # Reports what the store accounts for, not what the collector sees. Two
  # earlier versions measured the process heap and both failed: `heap_size`
  # is a high-water mark that never shrinks, and bytes in use went negative,
  # because a conservative collector frees the previous fill's garbage during
  # the measurement. The heap cannot isolate one store; the store can.
  def self.footprint(embedder, sizes : Array(Int32)) : Nil
    puts "\n## Per-instance footprint"
    puts
    puts "Node cache is what the store holds in memory for its graph, after"
    puts "#{QUERIES} queries. SQLite's own page cache is separate, and set with"
    puts "`page_cache_bytes:` (#{mb(Vecstolite::Repository::DEFAULT_PAGE_CACHE_BYTES)} by default)."
    puts
    puts "| corpus | cache | node cache | per entry | database |"
    puts "|-------:|-------|-----------:|----------:|---------:|"

    modes = {
      "lru 0.5 MB" => CacheMode.lru(512_i64 * Vecstolite::KB),
      "lru 5 MB"   => CacheMode.lru(5_i64 * Vecstolite::MB),
      "memory"     => CacheMode.memory,
      "disk"       => CacheMode.disk,
    }

    sizes.each do |size|
      texts = corpus(size)
      probes = queries(texts)

      modes.each do |label, mode|
        reset_db
        Store.open(DB_PATH, embedder, index: graph, cache: mode) { |store| fill(store, texts) }

        store = Store.open(DB_PATH, embedder, index: graph, cache: mode)
        search_all(store, probes)
        held = store.stats[:cache_bytes]
        store.close

        per_entry = "%.2f KB" % (held / size / 1024.0)
        puts "| #{size} | #{label} | #{mb(held)} | #{per_entry} | #{mb(db_bytes)} |"
      end
    end
    reset_db
  end

  # ---------------------------------------------------------------------------
  # 3. Ingest — the write path W2a and W2b live on.
  # ---------------------------------------------------------------------------
  def self.ingest(embedder, sizes : Array(Int32)) : Nil
    puts "\n## Ingest"
    puts
    puts "| corpus | config | style | total | per entry | database |"
    puts "|-------:|--------|-------|------:|----------:|---------:|"

    configs = {
      "hnsw m=16 memory" => {Vecstolite::Index.hnsw(m: 16, ef_construction: 200, seed: SEED), CacheMode.memory},
      "hnsw m=8 memory"  => {Vecstolite::Index.hnsw(m: 8, ef_construction: 200, seed: SEED), CacheMode.memory},
      "hnsw m=16 lru"    => {Vecstolite::Index.hnsw(m: 16, ef_construction: 200, seed: SEED),
                          CacheMode.lru(512_i64 * Vecstolite::KB)},
      "flat" => {Vecstolite::Index.flat, CacheMode.disk},
    }

    sizes.each do |size|
      texts = corpus(size)

      configs.each do |label, (index, cache)|
        {"bulk", "single"}.each do |style|
          reset_db
          store = Store.open(DB_PATH, embedder, index: index, cache: cache)
          elapsed = Time.measure do
            if style == "bulk"
              fill(store, texts)
            else
              texts.each { |text| store.add(text) }
            end
          end
          store.close

          puts "| #{size} | #{label} | #{style} | #{ms(elapsed)} | " \
               "#{ms(elapsed / size)} | #{mb(db_bytes)} |"
        end
      end
    end
    reset_db
  end

  # ---------------------------------------------------------------------------
  # 4. Restart — W2a restarts constantly.
  # ---------------------------------------------------------------------------
  def self.restart(embedder, sizes : Array(Int32)) : Nil
    puts "\n## Open and first query"
    puts
    puts "| corpus | cache | exit | open | first query |"
    puts "|-------:|-------|------|-----:|------------:|"

    modes = {
      "lru 0.5 MB" => CacheMode.lru(512_i64 * Vecstolite::KB),
      "memory"     => CacheMode.memory,
      "disk"       => CacheMode.disk,
    }

    sizes.each do |size|
      texts = corpus(size)
      probe = queries(texts).first

      modes.each do |label, mode|
        # Clean: the graph was flushed at close.
        reset_db
        Store.open(DB_PATH, embedder, index: graph, cache: mode) { |store| fill(store, texts) }
        measure_open(embedder, mode, probe) do |open_span, query_span|
          puts "| #{size} | #{label} | clean | #{ms(open_span)} | #{ms(query_span)} |"
        end

        # Unclean: entries were committed as they were added, but the store
        # was never closed, so a memory cache never wrote its nodes.
        reset_db
        abandoned = Store.open(DB_PATH, embedder, index: graph, cache: mode)
        fill(abandoned, texts)
        measure_open(embedder, mode, probe) do |open_span, query_span|
          puts "| #{size} | #{label} | unclean | #{ms(open_span)} | #{ms(query_span)} |"
        end
      end
    end
    reset_db
  end

  private def self.measure_open(embedder, mode, probe : String, &)
    store = uninitialized Store
    open_span = Time.measure { store = Store.open(DB_PATH, embedder, index: graph, cache: mode) }
    query_span = Time.measure { store.search(probe, k: TOP_K) }
    store.close
    yield open_span, query_span
  end

  # ---------------------------------------------------------------------------
  # 5. Crossover — where the graph starts paying for itself.
  # ---------------------------------------------------------------------------
  def self.crossover(embedder, sizes : Array(Int32)) : Nil
    puts "\n## Flat against HNSW"
    puts
    recall_note
    comparison_header

    sizes.each do |size|
      texts = corpus(size)
      compare(embedder, size.to_s, texts, queries(texts), TOP_K)
    end
    reset_db
  end

  # ---------------------------------------------------------------------------
  # 5b. Duplicates — how the graph copes with identical vectors.
  #
  # Real corpora repeat themselves: a translation memory holds the same segment
  # many times. Each distinct sentence here appears COPIES times with an
  # identical vector, and each query asks for exactly COPIES results, so the
  # perfect answer is every copy of the nearest sentence.
  # ---------------------------------------------------------------------------
  COPIES = 5

  def self.duplicates(embedder, sizes : Array(Int32)) : Nil
    puts "\n## Duplicate vectors"
    puts
    puts "Each distinct sentence appears #{COPIES} times with an identical vector,"
    puts "and every query asks for #{COPIES} results."
    puts
    comparison_header

    sizes.each do |size|
      distinct = corpus(size // COPIES)
      texts = distinct.flat_map { |text| Array.new(COPIES, text) }
      compare(embedder, "#{size} (#{distinct.size} × #{COPIES})", texts, queries(distinct), COPIES)
    end
    reset_db
  end

  private def self.recall_note : Nil
    puts "Two recall figures, because they disagree when scores tie. *ids* is"
    puts "the share of exact results the graph also returned. *scores* is the"
    puts "share of graph results scoring at least as well as the exact k-th —"
    puts "an equally good answer that happens to be a different entry counts"
    puts "here but not above. Sparse embeddings tie constantly, so *ids*"
    puts "understates quality on them."
    puts
  end

  private def self.comparison_header : Nil
    puts "| corpus | strategy | #{QUERIES} queries | per query | recall (ids) | recall (scores) | database |"
    puts "|-------:|----------|-------------------:|----------:|-------------:|----------------:|---------:|"
  end

  # Runs the same queries through Flat and HNSW and prints a row for each.
  private def self.compare(embedder, label : String, texts : Array(String),
                           probes : Array(String), k : Int32) : Nil
    reset_db
    exact = [] of Array({String, Float32})
    flat_span = Time::Span.zero
    Store.open(DB_PATH, embedder, index: Vecstolite::Index.flat) do |store|
      fill(store, texts)
      flat_span = Time.measure do
        probes.each { |query| exact << store.search(query, k: k).map { |hit| {hit.text, hit.score} } }
      end
    end
    puts "| #{label} | flat | #{ms(flat_span)} | #{ms(flat_span / probes.size)} | 1.000 | 1.000 | #{mb(db_bytes)} |"

    reset_db
    approx = [] of Array({String, Float32})
    graph_span = Time::Span.zero
    Store.open(DB_PATH, embedder, index: graph, cache: CacheMode.memory) do |store|
      fill(store, texts)
      graph_span = Time.measure do
        probes.each { |query| approx << store.search(query, k: k).map { |hit| {hit.text, hit.score} } }
      end
    end

    by_id = 0
    by_score = 0
    exact.zip(approx) do |want, got|
      # Duplicates share their text, so compare as multisets by counting.
      want_counts = want.map(&.[0]).tally
      got_counts = got.map(&.[0]).tally
      by_id += want_counts.sum { |text, count| Math.min(count, got_counts.fetch(text, 0)) }
      next if want.empty?

      # The exact k-th score is the bar any equally good answer clears.
      threshold = want.last[1] - 1e-5_f32
      by_score += got.count { |(_, score)| score >= threshold }
    end

    total = (probes.size * k).to_f
    puts "| #{label} | hnsw | #{ms(graph_span)} | #{ms(graph_span / probes.size)} | " \
         "#{"%.3f" % (by_id / total)} | #{"%.3f" % (by_score / total)} | #{mb(db_bytes)} |"
  end

  # ---------------------------------------------------------------------------
  # 6. Deletion and compaction — what tombstones cost, and when to compact.
  # ---------------------------------------------------------------------------
  def self.compact(embedder, sizes : Array(Int32)) : Nil
    puts "\n## Deletion and compaction"
    puts
    puts "| corpus | tombstones | search (#{QUERIES}) | compact! | before | after |"
    puts "|-------:|-----------:|-------------------:|---------:|-------:|------:|"

    sizes.each do |size|
      texts = corpus(size)
      probes = queries(texts)

      {0.1, 0.25, 0.5}.each do |ratio|
        reset_db
        store = Store.open(DB_PATH, embedder, index: graph, cache: CacheMode.memory)
        store.bulk { |batch| texts.each { |text| batch.add(text, key: text) } }

        texts.first((size * ratio).to_i).each { |text| store.delete_by_key(text) }

        search_span = search_all(store, probes)
        before = db_bytes
        compact_span = Time.measure { store.compact! }
        store.close

        puts "| #{size} | #{"%.0f%%" % (ratio * 100)} | #{ms(search_span)} | " \
             "#{ms(compact_span)} | #{mb(before)} | #{mb(db_bytes)} |"
      end
    end
    reset_db
  end

  def self.run : Nil
    args = ARGV.dup
    first = args.first?
    suite = first && !first.starts_with?("--") ? args.shift : "all"

    sizes = [1_000, 10_000]
    dims = 768
    model_dir = nil.as(String?)

    while option = args.shift?
      case option
      when "--sizes" then sizes = args.shift.split(',').map(&.to_i)
      when "--dims"  then dims = args.shift.to_i
      when "--model" then model_dir = args.shift
      else                abort("Unknown option: #{option}")
      end
    end

    embedder =
      if dir = model_dir
        Vecstolite::StaticEmbedder.load(dir)
      else
        Vecstolite::LexicalEmbedder.new(dims)
      end

    puts "# Vecstolite benchmarks"
    puts
    puts "- Crystal #{Crystal::VERSION}"
    puts "- embedder: #{embedder.model_name}, #{embedder.dimensions} dimensions"
    puts "- corpus sizes: #{sizes.join(", ")}"
    puts "- SQLite page size: #{Vecstolite::Repository::PAGE_SIZE} bytes"
    {% unless flag?(:release) %}
      puts
      puts "> Built without --release. Timings are not meaningful."
    {% end %}

    reset_db
    case suite
    when "cache"      then cache(embedder, sizes)
    when "footprint"  then footprint(embedder, sizes)
    when "ingest"     then ingest(embedder, sizes)
    when "restart"    then restart(embedder, sizes)
    when "crossover"  then crossover(embedder, sizes)
    when "duplicates" then duplicates(embedder, sizes)
    when "compact"    then compact(embedder, sizes)
    when "all"
      cache(embedder, sizes)
      footprint(embedder, sizes)
      ingest(embedder, sizes)
      restart(embedder, sizes)
      crossover(embedder, sizes)
      duplicates(embedder, sizes)
      compact(embedder, sizes)
    else abort("Unknown suite: #{suite}")
    end
  end
end

Benchmarks.run
