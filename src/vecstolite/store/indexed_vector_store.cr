require "../indexer/hnsw_indexer"

require "../vector_store"

module Vecstolite
  # Base definition for an HNSW-indexed vector store
  module IndexedVectorStore
    include VectorStore

    DEFAULT_M               =  16
    DEFAULT_EF_CONSTRUCTION = 200
    DEFAULT_EF_SEARCH       =  50

    # :inherit:
    # Adjust `ef` to trade speed (lower) for recall (higher).
    abstract def search(query : String, k : Int32 = DEFAULT_K, ef_search : Int32 = DEFAULT_EF_SEARCH) : Array(SearchResult)
  end
end
