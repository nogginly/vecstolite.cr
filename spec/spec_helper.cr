require "spectator"
require "wiretap"

require "../src/vecstolite"

Spectator.configure do |config|
  config.fail_blank # Fail on no tests.
  config.randomize  # Randomize test order.
end

Wiretap.configure do |c|
  # `:once` only under `RECORD=1`, and never in CI. Otherwise a missing
  # transcript fails the run instead of quietly reaching for the network —
  # which on the paid endpoints would also be a bill.
  #
  # Compared rather than `try`ed: `ENV["RECORD"]?.try` yields nil when the
  # variable is unset, which is falsy, which would record on exactly the plain
  # run this is meant to protect.
  never_record = ENV["CI"]? || ENV["RECORD"]? != "1"
  c.record_mode = never_record ? :none : :once
end
