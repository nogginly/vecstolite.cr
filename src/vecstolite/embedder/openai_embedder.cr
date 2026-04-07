require "http/client"
require "json"
require "uri"

require "../vector_embedder"

# HTTP client for the OpenAI v1/embeddings API (and any compatible endpoint,
# e.g. Ollama, LM Studio, llama.cpp server, Azure OpenAI).
#
# Request:
#   POST /v1/embeddings
#   Authorization: Bearer <api_key>
#   Content-Type: application/json
#
#   { "model": "text-embedding-3-small", "input": "your text", "encoding_format": "float" }
#
# Response:
#   {
#     "object": "list",
#     "data": [ { "object": "embedding", "index": 0, "embedding": [0.123, ...] } ],
#     "model": "text-embedding-3-small",
#     "usage": { "prompt_tokens": 8, "total_tokens": 8 }
#   }
module Vecstolite
  class OpenAIEmbedder
    include VectorEmbedder

    class Error < Exception; end

    class HttpError < Error
      getter status : Int32

      def initialize(message : String, @status : Int32)
        super(message)
      end
    end

    DEFAULT_BASE_URL = "https://api.openai.com"
    DEFAULT_MODEL    = "text-embedding-3-small"
    DEFAULT_TIMEOUT  = 30.seconds

    getter base_url : String
    getter model : String
    getter dimensions : Int32

    private getter api_key : String

    # *base_url* lets you point at any OpenAI-compatible server:
    #   - OpenAI:    "https://api.openai.com"        (default)
    #   - Ollama:    "http://localhost:11434"
    #   - LM Studio: "http://localhost:1234"
    #   - Azure:     "https://<resource>.openai.azure.com/openai"
    #
    # *api_key* is sent as "Authorization: Bearer <key>".
    # Pass an empty string for local servers that don't require auth.
    #
    # *dimensions* is forwarded when non-nil; only supported by
    # text-embedding-3 and later models.
    def initialize(
      @dimensions : Int32,
      @api_key : String,
      @model : String = DEFAULT_MODEL,
      base_url : String = DEFAULT_BASE_URL,
      timeout : Time::Span = DEFAULT_TIMEOUT,
    )
      @base_url = base_url.rstrip("/")
      @timeout = timeout
    end

    # Embed a single string.  Returns the embedding as Embedding.
    def embed(text : String) : Embedding
      parse_response(post(build_request_body(text)))
    end

    # ---------------------------------------------------------------------------
    private def build_request_body(text : String) : String
      obj = {
        "model"           => JSON::Any.new(@model),
        "input"           => JSON::Any.new(text),
        "encoding_format" => JSON::Any.new("float"),
      } of String => JSON::Any

      if d = @dimensions
        obj["dimensions"] = JSON::Any.new(d.to_i64)
      end

      JSON::Any.new(obj).to_json
    end

    private def post(body : String) : String
      uri = URI.parse("#{@base_url}/v1/embeddings")

      HTTP::Client.new(uri) do |client|
        client.connect_timeout = @timeout
        client.read_timeout = @timeout

        headers = HTTP::Headers{
          "Content-Type"  => "application/json",
          "Authorization" => "Bearer #{@api_key}",
        }

        response = client.post(uri.request_target, headers: headers, body: body)

        unless response.success?
          raise HttpError.new(
            "HTTP #{response.status_code}: #{response.body.strip}",
            response.status_code
          )
        end

        response.body
      end
    end

    private def parse_response(body : String) : Embedding
      root = JSON.parse(body)

      data = root["data"]?.try(&.as_a?) ||
             raise Error.new("Response missing 'data' array")
      raise Error.new("Response 'data' array is empty") if data.empty?

      embedding_any = data[0]["embedding"]?.try(&.as_a?) ||
                      raise Error.new("Response missing 'data[0].embedding'")

      Embedding.new(embedding_any.size) { |i| embedding_any[i].as_f32 }
    end
  end
end
