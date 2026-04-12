require "json"

# Pure-Crystal parser for the safetensors file format.
#
# Format (all integers little-endian):
#   [8 bytes: uint64 header_len]
#   [header_len bytes: UTF-8 JSON]
#   [remaining bytes: raw tensor data]
#
# The JSON header is an object mapping tensor name → TensorInfo:
#   {
#     "__metadata__": { "format": "pt" },   ← optional, always skipped
#     "embeddings": {
#       "dtype":        "F32",
#       "shape":        [30000, 1024],
#       "data_offsets": [0, 122880000]       ← byte range in the data block
#     }
#   }
#
# data_offsets are relative to the start of the data block (i.e. byte 8 +
# header_len), NOT to the start of the file.
module Vecstolite
  # :nodoc:
  module SafeTensors
    class Error < Exception; end

    private DTYPE_MAP = {
      "F64":  DType::F64,
      "F32":  DType::F32,
      "F16":  DType::F16,
      "BF16": DType::BF16,
      "I64":  DType::I64,
      "I32":  DType::I32,
      "I16":  DType::I16,
      "I8":   DType::I8,
      "U8":   DType::U8,
      "BOOL": DType::BOOL,
    }

    # Supported element types.  We only need F32 and F16 for embedding models,
    # but the full set is listed so parsing never silently ignores an unknown dtype.
    enum DType
      F64
      F32
      F16
      BF16
      I64
      I32
      I16
      I8
      U8
      BOOL

      def self.from_string(s : String) : DType
        DTYPE_MAP[s]? || raise Error.new("Unknown safetensors dtype: #{s}")
      end

      # Bytes per scalar element.
      def byte_width : Int32
        case self
        when F64, I64       then 8
        when F32, I32       then 4
        when F16, BF16, I16 then 2
        when I8, U8, BOOL   then 1
        else                     raise Error.new("Unknown byte width for #{self}")
        end
      end
    end

    # Metadata for a single tensor as recorded in the JSON header.
    struct TensorInfo
      getter dtype : DType
      getter shape : Array(Int64)
      getter data_offsets : Tuple(Int64, Int64) # [start, end) in data block

      def initialize(@dtype, @shape, @data_offsets)
      end

      def element_count : Int64
        @shape.empty? ? 1_i64 : @shape.reduce(1_i64) { |acc, value| acc * value }
      end

      def byte_size : Int64
        element_count * @dtype.byte_width
      end
    end

    # A loaded safetensors file.
    #
    # Tensor data is accessed as a view into the memory-resident Bytes buffer —
    # no extra copies.  The buffer is kept alive by this object.
    class File
      getter tensors : Hash(String, TensorInfo)

      # The raw data block.  Slices returned by #tensor_bytes are sub-slices of
      # this, so callers must keep the File alive as long as they use the slices.
      getter data : Bytes

      def initialize(@tensors : Hash(String, TensorInfo), @data : Bytes)
      end

      # Loads a safetensors file from disk.
      def self.load(path : String) : self
        bytes = ::File.open(path, "rb") { |file| Bytes.new(file.size).tap { |buf| file.read_fully(buf) } }
        parse(bytes)
      end

      # Parses an in-memory buffer (e.g. embedded via `read_file` macro).
      def self.parse(bytes : Bytes) : self
        raise Error.new("Buffer too short to contain header size") if bytes.size < 8

        header_len = IO::ByteFormat::LittleEndian.decode(UInt64, bytes[0, 8]).to_i64
        raise Error.new("Header length (#{header_len}) exceeds buffer") \
          if 8 + header_len > bytes.size
        raise Error.new("Header length exceeds 100 MB safety limit") \
          if header_len > 100 * 1024 * 1024

        header_bytes = bytes[8, header_len]
        header_json = String.new(header_bytes)
        data_block = bytes[8 + header_len, bytes.size - 8 - header_len]

        tensors = parse_header(header_json, data_block.size.to_i64)

        new(tensors, data_block)
      end

      # Returns a zero-copy sub-slice of the data block for the named tensor.
      def tensor_bytes(name : String) : Bytes
        info = @tensors[name]? || raise Error.new("Tensor '#{name}' not found. Available: #{@tensors.keys.join(", ")}")
        start = info.data_offsets[0]
        stop = info.data_offsets[1]
        @data[start, stop - start]
      end

      # Convenience: returns tensor data as a Slice(Float32).
      # Raises if the tensor dtype is not F32.
      def tensor_f32(name : String) : Slice(Float32)
        info = @tensors[name]? || raise Error.new("Tensor '#{name}' not found")
        raise Error.new("Expected F32 tensor, got #{info.dtype}") unless info.dtype == DType::F32

        raw = tensor_bytes(name)
        # raw.size should equal element_count * 4 — validate to catch corrupt files
        expected = info.element_count * 4
        raise Error.new("Byte size mismatch: expected #{expected}, got #{raw.size}") \
          unless raw.size == expected

        Slice(Float32).new(raw.to_unsafe.as(Float32*), info.element_count)
      end

      # Returns a single row of a 2-D F32 tensor as a Slice(Float32) (zero-copy).
      # Intended for looking up rows in the embedding matrix.
      def tensor_f32_row(name : String, row : Int) : Slice(Float32)
        info = @tensors[name]? || raise Error.new("Tensor '#{name}' not found")
        raise Error.new("Expected F32 tensor, got #{info.dtype}") unless info.dtype == DType::F32
        raise Error.new("Expected 2-D tensor, got #{info.shape.size}-D") \
          unless info.shape.size == 2

        rows = info.shape[0]
        cols = info.shape[1]
        raise Error.new("Row #{row} out of bounds (tensor has #{rows} rows)") \
          if row < 0 || row >= rows

        byte_start = info.data_offsets[0] + row.to_i64 * cols * 4
        Slice(Float32).new(
          (@data.to_unsafe + byte_start).as(Float32*),
          cols
        )
      end

      # -----------------------------------------------------------------------
      private def self.parse_header(json : String, data_block_size : Int64) : Hash(String, TensorInfo)
        parsed = JSON.parse(json)
        obj = parsed.as_h? || raise Error.new("Header JSON is not an object")

        tensors = {} of String => TensorInfo

        obj.each do |key, val|
          next if key == "__metadata__"

          tensor_obj = val.as_h? || raise Error.new("Tensor entry '#{key}' is not an object")
          tensors[key] = parse_tensor(key, data_block_size, tensor_obj)
        end

        tensors
      end

      private def self.parse_tensor(key, data_block_size, tensor_obj)
        dtype_str = tensor_obj["dtype"]?.try(&.as_s?) ||
                    raise Error.new("Missing 'dtype' in tensor '#{key}'")

        shape_arr = tensor_obj["shape"]?.try(&.as_a?) ||
                    raise Error.new("Missing 'shape' in tensor '#{key}'")
        shape = shape_arr.map(&.as_i64)

        offsets_arr = tensor_obj["data_offsets"]?.try(&.as_a?) ||
                      raise Error.new("Missing 'data_offsets' in tensor '#{key}'")
        raise Error.new("'data_offsets' must have exactly 2 elements in '#{key}'") \
          unless offsets_arr.size == 2

        off_start = offsets_arr[0].as_i64
        off_end = offsets_arr[1].as_i64

        raise Error.new("Negative offset in tensor '#{key}'") if off_start < 0 || off_end < 0
        raise Error.new("Inverted offsets in tensor '#{key}'") if off_start > off_end
        raise Error.new("Offset out of bounds in tensor '#{key}': end=#{off_end} > data_block_size=#{data_block_size}") \
          if off_end > data_block_size

        dtype = DType.from_string(dtype_str)
        offsets = {off_start, off_end}
        TensorInfo.new(dtype, shape, offsets)
      end
    end
  end
end
