require "../src/vecstolite"

require "json"
require "benchmark"

alias BenchStore = Vecstolite::Store(String, String)

USAGE = "Usage: benchmark <path_to_static_model_dir> <path_to_sentence_file>"
model_dir = ARGV[0]? || abort(USAGE)
text_file = ARGV[1]? || abort(USAGE)

sentences = File.read_lines(text_file)

embedder = Vecstolite::StaticEmbedder.load(model_dir)

File.delete("tmp_bench_m16.db") if File.exists?("tmp_bench_m16.db")
File.delete("tmp_bench_m8.db") if File.exists?("tmp_bench_m8.db")

vector_stores = {
  "HNSW(M=16, EF=200)" => BenchStore.open("tmp_bench_m16.db", embedder,
    index: Vecstolite::Index.hnsw(m: 16, ef_construction: 200)),
  "HNSW(M=8, EF=200)" => BenchStore.open("tmp_bench_m8.db", embedder,
    index: Vecstolite::Index.hnsw(m: 8, ef_construction: 200)),
}

puts "### Add all (#{sentences.size} sentences)"

vector_stores.each do |name, store|
  result = Benchmark.measure {
    sentences.each do |text|
      store.add(text, meta: {hash: text.hash}.to_json)
    end
  }
  puts "#{result}\t#{name}"
end

sample = sentences.sample(sentences.size // 10)

puts "### Search (#{sample.size} sentences)"

vector_stores.each do |name, store|
  result = Benchmark.measure {
    sample.each do |line|
      store.search(line, k: 3)
    end
  }
  puts "#{result}\t#{name}"
end

vector_stores.each { |_, store| store.close }
