require "../src/vecstolite"

# The shortest path through the API: open, add, search, delete, compact,
# reopen. Runs as-is, with no model to download.
#
#   crystal run samples/quickstart.cr
#
# LexicalEmbedder hashes words into a vector, so it matches shared words rather
# than meaning. Swap in a StaticEmbedder or OpenAIEmbedder for real semantic
# search; nothing else changes.

DB_PATH = "quickstart.db"
Dir.glob("#{DB_PATH}*") { |file| File.delete?(file) }

embedder = Vecstolite::LexicalEmbedder.new(dimensions: 256)

SENTENCES = [
  "The sky is blue on a clear day.",
  "Roses are red and violets are blue.",
  "Crystal is a statically typed language with Ruby-like syntax.",
  "A transformer is a type of neural network architecture.",
  "The ocean looks deep and blue from the cliffs.",
]

def show(label : String, results) : Nil
  puts label
  results.each { |result| puts "  [#{result.score.round(3)}] #{result.text}" }
  puts
end

# 1. Open a store, add some text, search it. The block form closes the store
#    on the way out, even if something inside raises.
sky_id = 0_i64
Vecstolite::Store(String, String).open(DB_PATH, embedder) do |store|
  ids = SENTENCES.map { |text| store.add(text) }
  sky_id = ids.first

  # A key of your own, for finding an entry again without keeping its id.
  store.add("Grass is green in spring.", key: "grass")

  show("Search 'blue sky':", store.search("blue sky", k: 3))
  puts "By key 'grass': #{store.get_by_key("grass").try(&.text)}"
  puts
end

# 2. Reopen. Everything persisted, including the search graph.
Vecstolite::Store(String, String).open(DB_PATH, embedder) do |store|
  puts "Reopened with #{store.size} entries."
  puts

  # 3. Delete one entry. It disappears from results immediately.
  store.delete(sky_id)
  show("After deleting the sky sentence:", store.search("blue sky", k: 3))

  # 4. Reclaim the space. Ids and keys are unaffected.
  puts "Pending deletions before compact!: #{store.tombstones}"
  store.compact!
  puts "Pending deletions after compact!:  #{store.tombstones}"
  puts "Still findable by key: #{store.get_by_key("grass").try(&.text)}"
end

Dir.glob("#{DB_PATH}*") { |file| File.delete?(file) }
