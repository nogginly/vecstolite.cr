require "../src/vecstolite"

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

# Per-embedding metadata: which language this search key is in.
record EmbeddingMeta, language : String do
  include JSON::Serializable
end

# Shared payload: the full translation set for a phrase.
record TranslationSet, en : String, fr : String, de : String do
  include JSON::Serializable
end

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DBNAME = "test04.db"

open_existing = ARGV[0]? == "--open"

embedder = Vecstolite::LexicalEmbedder.new(dimensions: 512)

store = if open_existing
          Vecstolite::SQLitePayloadVectorStore(EmbeddingMeta, TranslationSet)
            .open(DBNAME, embedder)
        else
          Dir.glob("#{DBNAME}*") { |f| File.delete?(f) }
          Vecstolite::SQLitePayloadVectorStore(EmbeddingMeta, TranslationSet)
            .create(DBNAME, embedder)
        end

# ---------------------------------------------------------------------------
# Populate (skipped when --open)
# ---------------------------------------------------------------------------

translations = [
  TranslationSet.new(
    en: "The sky is blue during a clear day.",
    fr: "Le ciel est bleu par temps clair.",
    de: "Der Himmel ist an einem klaren Tag blau."),
  TranslationSet.new(
    en: "Flowers bloom in spring when temperatures rise.",
    fr: "Les fleurs fleurissent au printemps quand les températures montent.",
    de: "Blumen blühen im Frühling, wenn die Temperaturen steigen."),
  TranslationSet.new(
    en: "The ocean reflects the colour of the sky.",
    fr: "L'océan reflète la couleur du ciel.",
    de: "Der Ozean spiegelt die Farbe des Himmels wider."),
  TranslationSet.new(
    en: "Crystal is a statically typed language with Ruby-like syntax.",
    fr: "Crystal est un langage statiquement typé avec une syntaxe similaire à Ruby.",
    de: "Crystal ist eine statisch typisierte Sprache mit Ruby-ähnlicher Syntax."),
  TranslationSet.new(
    en: "Machine learning models learn from data.",
    fr: "Les modèles d'apprentissage automatique apprennent à partir de données.",
    de: "Maschinenlernmodelle lernen aus Daten."),
]

begin
  unless open_existing
    translations.each do |t|
      pid = store.add_payload(t)
      # Each phrase is indexed in all three languages, sharing one payload.
      store.add(t.en, meta: EmbeddingMeta.new("en"), payload_id: pid)
      store.add(t.fr, meta: EmbeddingMeta.new("fr"), payload_id: pid)
      store.add(t.de, meta: EmbeddingMeta.new("de"), payload_id: pid)
    end

    puts "Stored #{translations.size} translation sets " \
         "(#{store.size} embeddings, #{translations.size} payloads)."
    puts
  end

  puts store.stats
  puts

  # -------------------------------------------------------------------------
  # Search
  # -------------------------------------------------------------------------

  queries = [
    "What colour is the sky?",
    "programming languages and syntax",
    "learning from data",
  ]

  queries.each do |q|
    puts "Query: #{q}"
    puts "-" * 60
    store.search(q, k: 3).each_with_index do |r, i|
      lang = r.meta.try(&.language) || "?"
      en = r.payload.try(&.en) || "(no payload)"
      puts "  #{i + 1}. [#{r.score.round(4)}] (#{lang}) #{r.text}"
      puts "       → EN: #{en}"
    end
    puts
  end
ensure
  store.close
  puts store.stats
  puts
end
