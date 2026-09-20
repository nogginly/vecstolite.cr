require "./vecstolite/embedder/*"
require "./vecstolite/store/*"
require "./vecstolite/index/*"

# **Vecstolite** is a vector store _shard_ for Crystal, backed by SQLite3 on disk
# or in memory, with an HNSW index and configurable node caching.
#
# The store is `SQLitePayloadVectorStore`.
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
