require "spectator"

require "../src/vecstolite"

Spectator.configure do |config|
  config.fail_blank # Fail on no tests.
  config.randomize  # Randomize test order.
end
