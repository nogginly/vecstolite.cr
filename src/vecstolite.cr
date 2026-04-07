require "./vecstolite/embedder/*"
require "./vecstolite/store/*"

# Vector emmbedding-based in-memory and file (coming soon) storage.
module Vecstolite
  # Read this at compile time from shard.yml one day
  VERSION    = {{ `shards version #{__DIR__}`.chomp.stringify }}
  PRERELEASE = VERSION.match(/^\d+\.\d+\.\d+$/).nil?
end
