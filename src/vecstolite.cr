require "./vecstolite/embedder/*"
require "./vecstolite/store/*"

# **Vecstolite** is a vector store _shard_ for Crystal with in-memory and SQLite3-backed storage support.
#
# Different kinds of vector stores are available:
#
# - `LinearVectorStore`
# - `IndexedVectorStore`
# - `SQLiteVectorStore`
# - `SQLitePayloadVectorStore`
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
