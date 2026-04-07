require "json"

# Pure-Crystal WordPiece tokenizer compatible with bert-base-uncased and
# any model that uses the same tokenizer.json (including
# sentence-transformers/static-retrieval-mrl-en-v1).
#
# Pipeline (mirrors HuggingFace tokenizers exactly):
#
#   raw text
#     → Normalizer     : NFD → strip accents → lowercase
#     → PreTokenizer   : split on whitespace and punctuation
#     → WordPiece      : greedy longest-match with "##" continuation prefix
#     → token ids      : vocab hash lookup, unknown → [UNK]
#
# The tokenizer is loaded from a standard HuggingFace tokenizer.json file.

module Vecstolite
  class WordPieceTokenizer
    class Error < Exception; end

    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    PAD_TOKEN = "[PAD]"

    getter vocab : Hash(String, Int32) # token → id
    getter unk_id : Int32
    getter cls_id : Int32
    getter sep_id : Int32
    getter pad_id : Int32

    # Load from a tokenizer.json file on disk.
    def self.load(path : String) : self
      json = File.read(path)
      from_json(json)
    end

    # Parse a tokenizer.json string directly (useful for embedded configs).
    def self.from_json(json : String) : self
      root = JSON.parse(json)
      model = root["model"]? || raise Error.new("tokenizer.json missing 'model' key")

      type = model["type"]?.try(&.as_s?) || raise Error.new("tokenizer.json model missing 'type'")
      raise Error.new("Expected WordPiece tokenizer, got '#{type}'") unless type == "WordPiece"

      vocab_obj = model["vocab"]?.try(&.as_h?) || raise Error.new("tokenizer.json model missing 'vocab'")
      vocab = vocab_obj.transform_values(&.as_i)

      unk_id = vocab[UNK_TOKEN]? || raise Error.new("Vocab missing #{UNK_TOKEN}")
      cls_id = vocab[CLS_TOKEN]? || raise Error.new("Vocab missing #{CLS_TOKEN}")
      sep_id = vocab[SEP_TOKEN]? || raise Error.new("Vocab missing #{SEP_TOKEN}")
      pad_id = vocab[PAD_TOKEN]? || 0

      new(vocab, unk_id, cls_id, sep_id, pad_id)
    end

    private def initialize(
      @vocab : Hash(String, Int32),
      @unk_id : Int32,
      @cls_id : Int32,
      @sep_id : Int32,
      @pad_id : Int32,
    )
    end

    # Encode a string to token ids, with [CLS] and [SEP] added.
    def encode(text : String) : Array(Int32)
      ids = [] of Int32
      ids << @cls_id
      tokenize(text).each { |tok| ids << (@vocab[tok]? || @unk_id) }
      ids << @sep_id
      ids
    end

    # Tokenize without adding special tokens.  Returns token strings.
    def tokenize(text : String) : Array(String)
      tokens = [] of String
      pre_tokenize(normalize(text)).each do |word|
        tokens.concat(wordpiece(word))
      end
      tokens
    end

    # Vocab size
    def vocab_size : Int32
      @vocab.size
    end

    # Reverse lookup: id → token string
    def decode_token(id : Int32) : String
      @vocab.key_for?(id) || UNK_TOKEN
    end

    # ---------------------------------------------------------------------------
    # Step 1 — Normalisation
    #
    # Matches HuggingFace normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    #   1. Unicode NFD decomposition
    #   2. Remove characters in Unicode category Mn (non-spacing marks = accents)
    #   3. Lowercase
    # ---------------------------------------------------------------------------
    def normalize(text : String) : String
      # NFD decomposition — splits base characters from their combining marks
      nfd = text.unicode_normalize(:nfd)

      # Strip non-spacing combining marks (Unicode category Mn) and lowercase
      result = String::Builder.new(nfd.bytesize)
      nfd.each_char do |ch|
        next if combining_mark?(ch) # skip combining marks (accents)
        result << ch.downcase
      end
      result.to_s
    end

    # ---------------------------------------------------------------------------
    # Step 2 — Pre-tokenisation
    #
    # Matches HuggingFace pre_tokenizers.BertPreTokenizer (also called Whitespace):
    #   - Split on whitespace
    #   - Additionally split on punctuation characters: each punctuation char
    #     becomes its own token
    # ---------------------------------------------------------------------------
    def pre_tokenize(text : String) : Array(String)
      words = [] of String
      buf = String::Builder.new

      text.each_char do |ch|
        if ch.whitespace?
          unless buf.empty?
            words << buf.to_s
            buf = String::Builder.new
          end
        elsif punctuation?(ch)
          unless buf.empty?
            words << buf.to_s
            buf = String::Builder.new
          end
          words << ch.to_s
        else
          buf << ch
        end
      end

      words << buf.to_s unless buf.empty?
      words
    end

    # ---------------------------------------------------------------------------
    # Step 3 — WordPiece
    #
    # Greedy longest-match-first subword segmentation.
    # Continuation subtokens are prefixed with "##".
    # If a single character can't be found, the whole word becomes [UNK].
    # ---------------------------------------------------------------------------
    def wordpiece(word : String) : Array(String)
      return [UNK_TOKEN] if word.empty?

      tokens = [] of String
      chars = word.chars
      start = 0
      bad_word = false

      while start < chars.size
        finish = chars.size
        cur_tok = nil
        prefix = start > 0 ? "##" : ""

        while finish > start
          substr = prefix + chars[start, finish - start].join
          if @vocab.has_key?(substr)
            cur_tok = substr
            break
          end
          finish -= 1
        end

        if cur_tok.nil?
          bad_word = true
          break
        end

        tokens << cur_tok
        start = finish
      end

      bad_word ? [UNK_TOKEN] : tokens
    end

    # Returns true for Unicode non-spacing combining marks (category Mn).
    # These are the accent characters that NFD decomposition splits off from
    # their base letters, and which BERT's normalizer strips.
    #
    # Covers the canonical ranges:
    #   U+0300–U+036F  Combining Diacritical Marks (acute, grave, umlaut, etc.)
    #   U+1AB0–U+1AFF  Combining Diacritical Marks Extended
    #   U+1DC0–U+1DFF  Combining Diacritical Marks Supplement
    #   U+20D0–U+20FF  Combining Diacritical Marks for Symbols
    #   U+FE20–U+FE2F  Combining Half Marks
    private def combining_mark?(ch : Char) : Bool
      cp = ch.ord
      (cp >= 0x0300 && cp <= 0x036F) ||
        (cp >= 0x1AB0 && cp <= 0x1AFF) ||
        (cp >= 0x1DC0 && cp <= 0x1DFF) ||
        (cp >= 0x20D0 && cp <= 0x20FF) ||
        (cp >= 0xFE20 && cp <= 0xFE2F)
    end

    private def punctuation?(ch : Char) : Bool
      cp = ch.ord
      # ASCII punctuation ranges
      return true if (cp >= 33 && cp <= 47)   # !"#$%&'()*+,-./
      return true if (cp >= 58 && cp <= 64)   # :;<=>?@
      return true if (cp >= 91 && cp <= 96)   # [\]^_`
      return true if (cp >= 123 && cp <= 126) # {|}~
      # Unicode: anything in the P (punctuation) or S (symbol) general category.
      # Crystal's Char#mark? covers Mn/Mc/Me; we use the codepoint path above for
      # those.  For P and S we rely on the fact that non-ASCII punctuation and
      # symbols are not letters, digits, marks, or whitespace.
      !ch.ascii? && !ch.letter? && !ch.number? && !ch.whitespace? && !combining_mark?(ch)
    end
  end
end
