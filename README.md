# Vecstolite

A vector store _shard_ for Crystal with in-memory and (coming soon) file-based storage support.

### Installation

1. Add the dependency to your `shard.yml`:

```yml
dependencies:
  vecstolite:
    github: nogginly/vecstolite.cr
```

2. Run shards install

### Usage

Using a local `ollama` or an Open AI-compatible cloud LLM provider with vector embedding, herte's a very simple example using `vecstolite`:

```cr
require "vecstolite"

# Setup an embedder
embedder = Vecstolite::OpenAIEmbedder.new(
  dimensions: 768,
  base_url: "http://localhost:11434",
  api_key: "ollama",
  model: "nomic-embed-text-v2-moe",
)

# Created an indexed in-memory vector store
store = Vecstolite::IndexedVectorStore.new(embedder)

# Add text like so
store.add "The sky is blue during a clear day."
store.add "Roses are red and violets are blue."
store.add "Crystal is a statically typed language with Ruby-like syntax."
store.add "A transformer is a type of neural network architecture."

# Find top-k matches like so
store.search("What is the color of the sky?", k: 3)

# Result is an array of matches with `text` and `score` properties
```

### Development

See [DEVELOPMENT](./DEVELOPMENT.md) for how to build `vecstolite` and run the samples.

## Contributions, by invitation!

*With apologies*, at this time contributions are *by invitation only* and limited to people I know and see often.

These are early days for _Vecstolite_ and I am busy with family and work.

At this time I want to work on this at a manageable pace.
