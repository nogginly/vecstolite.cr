require "../src/vecstolite"

embedder = Vecstolite::LexicalEmbedder.new
store = Vecstolite::IndexedVectorStore.new(embedder)

store.add_all([
  "The sky is blue during a clear day.",
  "Roses are red and violets are blue.",
  "Crystal is a statically typed language with Ruby-like syntax.",
  "Machine learning models learn from data.",
  "The ocean reflects the colour of the sky.",
  "Compiled languages are generally faster than interpreted ones.",
  "A transformer is a type of neural network architecture.",
  "Flowers bloom in spring when temperatures rise.",
])

puts "Stored #{store.size} entries.\n\n"

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
