require "./vecstolite/embedder/*"
require "./vecstolite/store/*"
require "./vecstolite/index/*"

# **Vecstolite** is a vector store _shard_ for Crystal, backed by SQLite3 on
# disk or in memory.
#
# `Store` is the only store. How it searches and how much it keeps in memory
# are arguments, not separate classes:
#
# - `Index.hnsw` for approximate search over a navigable graph, `Index.flat`
#   for an exact scan.
# - `CacheMode.lru`, `CacheMode.memory` or `CacheMode.disk` for how graph
#   nodes are held.
#
# The following types of vector embedders are available:
#
# - `StaticEmbedder`
# - `OpenAIEmbedder`
# - `LexicalEmbedder` (for testing)
#
module Vecstolite
  # :nodoc:
  module Version
    VERSION    = {{ `shards version #{__DIR__}`.chomp.stringify }}
    PRERELEASE = VERSION.match(/^\d+\.\d+\.\d+$/).nil?
  end
end
