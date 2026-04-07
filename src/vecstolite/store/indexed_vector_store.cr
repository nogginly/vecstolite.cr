require "../indexer/hnsw_indexer"

require "../vector_store"

module Vecstolite
  # An implementation of VectorStore that maintains both the entry array
  # and the HNSW index.
  class IndexedVectorStore
    include VectorStore

    @entries : Array(Entry)
    @embedder : VectorEmbedder
    @index : HNSW::Index

    def initialize(@embedder, m : Int32 = 16, ef_construction : Int32 = 200)
      @entries = [] of Entry
      @index = HNSW::Index.new(dims: embedder.dimensions, m: m, ef_construction: ef_construction)
    end

    def add(text : String) : Nil
      vector = @embedder.embed(text)
      id = @entries.size
      @entries << Entry.new(text, vector)
      @index.add(id: id, vector: vector)
    end

    # *ef* can be raised at query time to trade speed for recall.
    def search(query : String, k : Int32 = 5, ef : Int32 = 50) : Array(SearchResult)
      return [] of SearchResult if @entries.empty?

      query_vec = @embedder.embed(query)
      @index.search(query_vec, k: k, ef: ef).map do |r|
        SearchResult.new(@entries[r.id].text, r.score)
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
