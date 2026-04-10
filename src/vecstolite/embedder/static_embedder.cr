require "../vector_embedder"
require "../tokenizer/*"

require "./safe_tensors"

# Pure-Crystal embedder for sentence-transformers/static-retrieval-mrl-en-v1
# (and any compatible StaticEmbedding / Model2Vec model).
#
# Forward pass:
#   text
#     → WordPiece tokenize  (includes [CLS] and [SEP])
#     → token id lookup
#     → embedding row lookup per id  (zero-copy slice into SafeTensors buffer)
#     → mean pool across all tokens
#     → L2 normalise
#     → optionally truncate to fewer dimensions (Matryoshka)
#
# The [CLS] and [SEP] special tokens ARE included in the mean pool.
# This matches the sentence-transformers reference implementation, which uses
# EmbeddingBag over the full encoded sequence including special tokens.
#
# File layout expected on disk:
#   <model_dir>/
#     model.safetensors          ← embedding table [vocab_size × dim]
#     tokenizer.json             ← WordPiece tokenizer config
#
# The tensor name inside model.safetensors is "embeddings".
module Vecstolite
  class StaticEmbedder
    include VectorEmbedder

    class Error < Exception; end

    TENSOR_NAME  = "embedding.weight"
    DEFAULT_DIMS = 1024 # full dimensionality of this model

    getter full_dims : Int32 # native embedding dimension (e.g. 1024)
    getter vocab_size : Int32

    # Load from a model directory containing `model.safetensors` + `tokenizer.json` files.
    # Supports "WordPiece" tokenizers only.
    def self.load(model_dir : String, truncate_dims : Int32? = nil) : self
      st_path = File.join(model_dir, "model.safetensors")
      tok_path = File.join(model_dir, "tokenizer.json")

      raise Error.new("Not found: #{st_path}") unless File.exists?(st_path)
      raise Error.new("Not found: #{tok_path}") unless File.exists?(tok_path)

      st = SafeTensors::File.load(st_path)
      tok = WordPieceTokenizer.load(tok_path)

      new(st, tok, truncate_dims)
    end

    private def initialize(
      @st : SafeTensors::File,
      @tokenizer : WordPieceTokenizer,
      @truncate_dims : Int32? = nil,
    )
      info = @st.tensors[TENSOR_NAME]? ||
             raise Error.new("model.safetensors has no '#{TENSOR_NAME}' tensor. " \
                             "Available: #{@st.tensors.keys.join(", ")}")

      raise Error.new("Expected 2-D tensor, got #{info.shape.size}-D") \
        unless info.shape.size == 2
      raise Error.new("Expected F32 tensor, got #{info.dtype}") \
        unless info.dtype == SafeTensors::DType::F32

      @vocab_size = info.shape[0].to_i32
      @full_dims = info.shape[1].to_i32

      if td = @truncate_dims
        raise Error.new("truncate_dims (#{td}) must be > 0") if td <= 0
        raise Error.new("truncate_dims (#{td}) exceeds model dims (#{@full_dims})") \
          if td > @full_dims
      end
    end

    # The output dimensionality after any truncation.
    def output_dims : Int32
      @truncate_dims || @full_dims
    end

    # Returns the number of dimensions
    def dimensions : Int32
      @full_dims
    end

    # Embed a single string.  Returns a normalised Embedding.
    def embed(text : String) : Embedding
      ids = @tokenizer.encode(text)
      embed_ids(ids)
    end

    # Embed from pre-computed token ids (e.g. if you already tokenised).
    def embed_ids(ids : Array(Int32)) : Embedding
      raise Error.new("No token ids provided") if ids.empty?

      dims = @full_dims
      sum = Embedding.new(dims, 0.0_f32)

      ids.each do |id|
        # Clamp out-of-range ids to [UNK] rather than crashing — matches
        # the behaviour of EmbeddingBag with padding_idx.
        safe_id = id.clamp(0, @vocab_size - 1)
        row = @st.tensor_f32_row(TENSOR_NAME, safe_id)
        dims.times { |dim| sum[dim] += row[dim] }
      end

      # Mean pool
      n = ids.size.to_f32
      dims.times { |dim| sum[dim] /= n }

      # L2 normalise
      l2_normalize!(sum)

      # Matryoshka truncation
      if td = @truncate_dims
        sum = sum[0, td]
        l2_normalize!(sum) # re-normalise after truncation
      end

      sum
    end

    # Cosine similarity between two already-normalised embedding vectors.
    # Since both are L2-normalised, this is just a dot product.
    def cosine_similarity(a : Embedding, b : Embedding) : Float32
      raise Error.new("Vector size mismatch") unless a.size == b.size

      VECM.dot(a, b)
    end

    # ---------------------------------------------------------------------------
    private def l2_normalize!(v : Embedding) : Nil
      norm = Math.sqrt(VECM.dot(v, v))
      return if norm < 1e-12_f32 # avoid divide-by-zero on zero vectors

      inv = 1.0_f32 / norm
      v.size.times { |i| v[i] *= inv }
    end
  end
end
