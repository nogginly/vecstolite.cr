require "../indexer/hnsw_indexer"

require "../vector_store"

module Vecstolite
  # Base definition for an HNSW-indexed vector store
  module IndexedVectorStore(M)
    include VectorStore(M)

    DEFAULT_M               =  16
    DEFAULT_EF_CONSTRUCTION = 200
    DEFAULT_EF_SEARCH       =  50

    # :inherit:
    def search(query : String, k : Int32 = DEFAULT_K) : Array
      search(query, k, DEFAULT_EF_SEARCH)
    end

    # :inherit:
    # Adjust `ef` to trade speed (lower) for recall (higher).
    abstract def search(query : String, k : Int32, ef_search : Int32) : Array
  end
end
