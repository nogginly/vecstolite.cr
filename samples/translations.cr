require "../src/vecstolite"

# Shared payloads: one translation set, searchable in every language it holds.
# Each language's text is its own entry, tagged with metadata, and all of them
# point at the same payload — so a French query returns the English and German
# alongside.
#
#   crystal run samples/translations.cr            # build, search, delete
#   crystal run samples/translations.cr -- --open  # reopen and search again

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

DBNAME = "translations.db"

open_existing = ARGV[0]? == "--open"

embedder = Vecstolite::LexicalEmbedder.new(dimensions: 512)

Dir.glob("#{DBNAME}*") { |file| File.delete?(file) } unless open_existing

store = Vecstolite::Store(EmbeddingMeta, TranslationSet).open(DBNAME, embedder)

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
    # One transaction for everything. A payload queued in the batch has no id
    # until the batch commits, so `add_payload` returns a placeholder that
    # `add` accepts in its place.
    store.bulk do |batch|
      translations.each do |translation|
        set = batch.add_payload(translation)
        # Each phrase is indexed in all three languages, sharing one payload.
        batch.add(translation.en, meta: EmbeddingMeta.new("en"), payload_id: set)
        batch.add(translation.fr, meta: EmbeddingMeta.new("fr"), payload_id: set)
        batch.add(translation.de, meta: EmbeddingMeta.new("de"), payload_id: set)
      end
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

  queries.each do |query|
    puts "Query: #{query}"
    puts "-" * 60
    store.search(query, k: 3).each_with_index do |result, i|
      lang = result.meta.try(&.language) || "?"
      en = result.payload.try(&.en) || "(no payload)"
      puts "  #{i + 1}. [#{result.score.round(4)}] (#{lang}) #{result.text}"
      puts "       → EN: #{en}"
    end
    puts
  end
  # -------------------------------------------------------------------------
  # Delete and compact
  # -------------------------------------------------------------------------

  unless open_existing
    first = store.search("What colour is the sky?", k: 1).first
    puts "Deleting payload #{first.payload_id} (#{first.text})"
    if pid = first.payload_id
      puts "  #{store.delete_payload(pid)} entries tombstoned, #{store.tombstones} awaiting compaction."
    end
    store.compact!
    puts "  after compact!: #{store.size} entries, #{store.tombstones} tombstones."
    puts
  end
ensure
  store.close
end
