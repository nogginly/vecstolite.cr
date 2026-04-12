require "../indexer/hnsw_indexer"

require "../vector_store"

module Vecstolite
  # An in-memory vector store with an HNSW index.
  #
  # Usage:
  # ```
  # store = Vecstolite::IndexedVectorStore.new(embedder)
  # store.add("The sky is blue.")
  # store.add("Crystal is fast.")
  # results = store.search("colour of the sky", k: 3)
  # ```
  class IndexedVectorStore
    include VectorStore

    DEFAULT_M               =  16
    DEFAULT_EF_CONSTRUCTION = 200
    DEFAULT_EF_SEARCH       =  50

    @entries : Array(Entry)
    @embedder : VectorEmbedder
    @index : HNSW::Index

    # Create an in-memory index vector store; specify `m` and `ef_construction` to
    # control the number of max neighbours per node per layer and
    # beam width when inserting a new node distibution respectively.
    def initialize(@embedder, m : Int32 = DEFAULT_M, ef_construction : Int32 = DEFAULT_EF_CONSTRUCTION)
      @entries = [] of Entry
      @index = HNSW::Index.new(dims: embedder.dimensions, m: m, ef_construction: ef_construction)
    end

    def add(text : String, extra : String? = nil) : Nil
      vector = @embedder.embed(text)
      id = @entries.size
      @entries << Entry.new(text, vector, extra)
      @index.add(id: id, vector: vector)
    end

    # :inherit:
    # Adjust `ef` to trade speed (lower) for recall (higher).
    def search(query : String, k : Int32 = DEFAULT_K, ef : Int32 = DEFAULT_EF_SEARCH) : Array(SearchResult)
      return [] of SearchResult if @entries.empty?

      query_vec = @embedder.embed(query)
      @index.search(query_vec, k: k, ef: ef).map do |result|
        entry = @entries[result.id]
        SearchResult.new(entry.text, result.score, entry.extra)
      end
    end

    def size : Int32
      @entries.size
    end

    def clear : Nil
      @entries.clear
      @index = HNSW::Index.new(dims: embedder.dimensions)
    end
  end
end
