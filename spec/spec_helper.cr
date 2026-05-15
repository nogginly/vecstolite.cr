require "spectator"
require "wiretap"

require "../src/vecstolite"

Spectator.configure do |config|
  config.fail_blank # Fail on no tests.
  config.randomize  # Randomize test order.
end

Wiretap.configure do |c|
  c.transcript_dir = "spec/fixtures/transcripts"

  # If in CI, don't record, fail if not found
  c.record_mode = ENV["CI"]? ? :none : :once
end
