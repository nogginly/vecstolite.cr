require "../src/vecstolite"

require "json"
require "benchmark"

USAGE = "Usage: benchmark <path_to_static_model_dir> <path_to_sentence_file>"
model_dir = ARGV[0]? || abort(USAGE)
text_file = ARGV[1]? || abort(USAGE)

sentences = File.read_lines(text_file)

embedder = Vecstolite::StaticEmbedder.load(model_dir)

File.delete("demo_m16.db") if File.exists?("demo_m16.db")
File.delete("demo_m8.db") if File.exists?("demo_m8.db")

vector_stores = {
  "SQLite3(M=16, EF=200)" => Vecstolite::SQLiteVectorStore.open("tmp_bench_m16.db", embedder, m: 16, ef_construction: 200),
  "SQLite3(M=8, EF=200)"  => Vecstolite::SQLiteVectorStore.open("tmp_bench_m8.db", embedder, m: 8, ef_construction: 200),
  "Indexed(M=16, EF=200)" => Vecstolite::IndexedVectorStore.new(embedder, m: 16, ef_construction: 200),
  "Indexed(M=8, EF=200)"  => Vecstolite::IndexedVectorStore.new(embedder, m: 8, ef_construction: 200),
  "Indexed(M=16, EF=100)" => Vecstolite::IndexedVectorStore.new(embedder, m: 16, ef_construction: 100),
  "Indexed(M=8, EF=100)"  => Vecstolite::IndexedVectorStore.new(embedder, m: 8, ef_construction: 100),
  "Simple"                => Vecstolite::LinearVectorStore.new(embedder),
}

puts "### Add all (#{sentences.size} sentences)"

vector_stores.each do |name, store|
  result = Benchmark.measure {
    sentences.each do |text|
      store.add(text, extra: {hash: text.hash}.to_json)
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

vector_stores.each do |_, store|
  if store.is_a? Vecstolite::SQLiteVectorStore
    store.close
  end
end
