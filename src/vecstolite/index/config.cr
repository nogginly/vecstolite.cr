require "./strategy"
require "./flat"
require "./hnsw"

module Vecstolite
  module Index
    # Describes an index strategy before a store exists to build it against.
    #
    # ```
    # Index.hnsw(m: 16, ef_construction: 200) # navigable graph
    # Index.flat                              # exact scan, no graph
    # ```
    abstract struct Config
      abstract def build(repo : Repository, cache : NodeCache::Strategy) : Strategy
      abstract def kind : Symbol

      # True when the strategy maintains a graph, and so needs a node cache
      # and graph metadata.
      abstract def graph? : Bool
    end

    struct HNSWConfig < Config
      getter m : Int32
      getter ef_construction : Int32
      getter seed : Int32?

      def initialize(@m : Int32 = HNSW::DEFAULT_M,
                     @ef_construction : Int32 = HNSW::DEFAULT_EF_CONSTRUCTION,
                     @seed : Int32? = nil)
      end

      def build(repo : Repository, cache : NodeCache::Strategy) : Strategy
        HNSW.new(cache,
          dims: repo.dimensions,
          m: @m,
          ef_construction: @ef_construction,
          seed: @seed)
      end

      def kind : Symbol
        :hnsw
      end

      def graph? : Bool
        true
      end
    end

    struct FlatConfig < Config
      def build(repo : Repository, cache : NodeCache::Strategy) : Strategy
        Flat.new(repo)
      end

      def kind : Symbol
        :flat
      end

      def graph? : Bool
        false
      end
    end

    def self.hnsw(m : Int32 = HNSW::DEFAULT_M,
                  ef_construction : Int32 = HNSW::DEFAULT_EF_CONSTRUCTION,
                  seed : Int32? = nil) : Config
      HNSWConfig.new(m, ef_construction, seed)
    end

    def self.flat : Config
      FlatConfig.new
    end
  end
end
