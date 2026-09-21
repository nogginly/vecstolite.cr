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

      # Fills anything left unspecified from what *repo* was built with, or
      # from defaults for a new store.
      def resolve(repo : Repository) : Config
        self
      end

      # True unless *repo* was built with parameters this configuration
      # contradicts in a way that invalidates the stored index.
      def compatible_with?(repo : Repository) : Bool
        true
      end

      # Writes this configuration's parameters into *repo*'s metadata.
      def record(repo : Repository) : Nil
      end
    end

    # Parameters left `nil` take whatever the store was built with, so
    # reopening with a bare `Index.hnsw` never silently changes a graph built
    # with non-default settings. A new store falls back to the defaults.
    struct HNSWConfig < Config
      getter m : Int32?
      getter ef_construction : Int32?
      getter seed : Int32?

      def initialize(@m : Int32? = nil,
                     @ef_construction : Int32? = nil,
                     @seed : Int32? = nil)
      end

      def resolve(repo : Repository) : Config
        HNSWConfig.new(
          @m || repo.meta_int("m").try(&.to_i32) || HNSW::DEFAULT_M,
          @ef_construction || repo.meta_int("ef_construction").try(&.to_i32) || HNSW::DEFAULT_EF_CONSTRUCTION,
          @seed)
      end

      # A different `m` changes how many edges every node may hold, so a graph
      # built with one cannot be extended with another: the store rebuilds.
      # `ef_construction` only shapes how carefully future inserts search, so
      # changing it is a tuning decision, not an incompatibility.
      def compatible_with?(repo : Repository) : Bool
        stored = repo.meta_int("m")
        stored.nil? || stored == @m
      end

      def record(repo : Repository) : Nil
        repo.set_meta("m", effective_m)
        repo.set_meta("ef_construction", effective_ef_construction)
      end

      def build(repo : Repository, cache : NodeCache::Strategy) : Strategy
        HNSW.new(cache,
          dims: repo.dimensions,
          m: effective_m,
          ef_construction: effective_ef_construction,
          seed: @seed)
      end

      def kind : Symbol
        :hnsw
      end

      def graph? : Bool
        true
      end

      private def effective_m : Int32
        @m || HNSW::DEFAULT_M
      end

      private def effective_ef_construction : Int32
        @ef_construction || HNSW::DEFAULT_EF_CONSTRUCTION
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

    # A navigable graph. Leave *m* and *ef_construction* unset to reuse what
    # an existing store was built with.
    def self.hnsw(m : Int32? = nil,
                  ef_construction : Int32? = nil,
                  seed : Int32? = nil) : Config
      HNSWConfig.new(m, ef_construction, seed)
    end

    def self.flat : Config
      FlatConfig.new
    end
  end
end
