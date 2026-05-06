# Vecstolite

A vector embedding-based storage _shard_ for Crystal with in-memory and SQLite3 back-ends.

## Installation

1. Add the dependency to your `shard.yml`:

```yml
dependencies:
  vecstolite:
    github: nogginly/vecstolite.cr
```

2. Run shards install

## Usage

### Embedder

#### OpenAI protocol

Here's an example of an OpenAI protocol embedder configured to use `ollama` to provider the embedding model:

```cr
require "vecstolite"

# Setup an embedder
embedder = Vecstolite::OpenAIEmbedder.new(
  dimensions: 768,
  base_url: "http://localhost:11434",
  api_key: "ollama",
  model: "nomic-embed-text-v2-moe",
)
```

#### Static local

Alternatively you can use a "static" embedder that doesn't need a GPU or a server to generate embeddings:

```cr
embedder = Vecstolite::StaticEmbedder.load(MODEL_PATH)
```

The `MODEL_PATH` should point to a directory that contains two files: `model.safetensors` and `tokenizer.json`.

Two static models have been tested, content files for which you can download from HuggingFace:

1. `static-retrieval-mrl-en-v1` [here in HF](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1)
2. `static-similarity-mrl-multilingual-v1` [here in HF](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1)

> NOTE that `vecstolite` only supports WordPiece tokenizers.

### Vector stores

#### In-memory indexed vector store

Here's a snippet to create an in-memory indexed vector store:

```cr
# Created an indexed in-memory vector store
store = Vecstolite::MemoryVectorStore.new(embedder)
```

#### SQLite3-backed Simple Vector Store

Here's a snippet to create a persistent indexed vector store backed by SQLite3:

```cr
# Created an indexed in-memory vector store
store = Vecstolite::SQLiteVectorStore.create("my_vector_store.db", embedder)
```
If one exists, use the `#open` method instead.

#### SQLite3-backed Payload Vector Store

The `SQLitePayloadVectorStore(M,P)` class is also backed by a SQLite DB, with the difference that you control the `meta` and `payload` types. As long as they are `JSON::Serializable`, the class takes care of the serialization and deserialization for you.

This supports a separate API for adding payloads separately, and then referring to payloads when adding searchable text. Look at `samples/test04.cr` for a translation-related example.

### Adding text

#### Just add text to search for later

```cr
# Add text like so
store.add "The sky is blue during a clear day."
store.add "Roses are red and violets are blue."
store.add "Crystal is a statically typed language with Ruby-like syntax."
store.add "A transformer is a type of neural network architecture."
```

#### Add text with metadata per entry

```cr
store.add("The sky is blue during a clear day.", meta: "en")
store.add("Le ciel est bleu par temps clair.", meta: "fr")
store.add("Der Himmel ist an einem klaren Tag blau", meta: "de")
```

#### Add text with typed meta-data and payload

```cr
store = Vecstolite::SQLitePayloadVectorStore(LangMeta, TranslationSet)
            .open(DBNAME, embedder)
set = TranslationSet.new(
    en: "The sky is blue during a clear day.",
    fr: "Le ciel est bleu par temps clair.",
    de: "Der Himmel ist an einem klaren Tag blau.")
pid = store.add_payload(set)
store.add(set.en, meta: LangMeta.new("en"), payload_id: pid)
store.add(set.fr, meta: LangMeta.new("fr"), payload_id: pid)
store.add(set.de, meta: LangMeta.new("de"), payload_id: pid)
```

### Searching

```cr
# Find top-k matches like so
store.search("What is the color of the sky?", k: 3)
```

Result is an array of matches with at least `text` and `score` properties (per `VectorSearchResult`) and others depending on the vector store itself.

## Development

See [DEVELOPMENT](./DEVELOPMENT.md) for how to build `vecstolite` and run the samples.

## Contributions, by invitation!

*With apologies*, at this time contributions are *by invitation only* and limited to people I know and see often.

These are early days for _Vecstolite_ and I am busy with family and work.

At this time I want to work on this at a manageable pace.
