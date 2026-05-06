require "../src/vecstolite"

embedder = Vecstolite::OpenAIEmbedder.new(
  dimensions: 768,
  api_key: "ollama",
  model: "nomic-embed-text-v2-moe",
  base_url: "http://localhost:11434",
)

DBNAME = "test03.db"

open_existing = ARGV[0]? == "--open"
store = if open_existing
          Vecstolite::SQLiteVectorStore.open(DBNAME, embedder,
            cache_ttl: 3.seconds,
            cache_purge_period: 2.seconds)
        else
          Dir.glob("#{DBNAME}*") do |file|
            puts "Deleting: #{file} #{File.delete?(file) ? "✅" : "❌"}"
          end

          Vecstolite::SQLiteVectorStore.create(DBNAME, embedder,
            cache_ttl: 3.seconds,
            cache_purge_period: 2.seconds)
        end

begin
  unless open_existing
    [
      "The sky is blue during a clear day.",
      "Crystal is a statically typed language with Ruby-like syntax.",
      "Machine learning models learn from data.",
      "Compiled languages are generally faster than interpreted ones.",
      "Flowers bloom in spring when temperatures rise.",
    ].each do |text|
      store.add(text, meta: {hash: text.hash}.to_json)
    end

    puts "Added 5"
    puts store.stats
    puts "Sleep for 4s"
    sleep(4.seconds)
    [
      "Roses are red and violets are blue.",
      "The ocean reflects the colour of the sky.",
      "A transformer is a type of neural network architecture.",
    ].each do |text|
      store.add(text, meta: {hash: text.hash}.to_json)
    end
  end
  puts store.stats

  queries = [
    "What colour is the sky?",
    "Tell me about programming languages",
    "neural networks and deep learning",
  ]

  queries.each do |q|
    puts "Query: #{q}"
    puts "-" * 50
    store.search(q, k: 3).each_with_index do |r, i|
      puts "  #{i + 1}. [#{r.score.round(4)}] #{r.text}"
    end
    puts
  end

  puts store.stats
ensure
  store.close
end
