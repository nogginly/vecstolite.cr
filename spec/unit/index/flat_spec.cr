require "../../spec_helper"

Spectator.describe Vecstolite::Index::Flat do
  alias Repo = Vecstolite::Repository
  alias Flat = Vecstolite::Index::Flat

  let(dims) { 3 }

  # Unit vectors on the three axes, plus one between x and y.
  private def unit(x : Float32, y : Float32, z : Float32) : Vecstolite::Embedding
    norm = Math.sqrt(x * x + y * y + z * z).to_f32
    values = [x / norm, y / norm, z / norm]
    Vecstolite::Embedding.new(3) { |i| values[i] }
  end

  private def seeded_repo : {Repo, Hash(String, Int64)}
    repo = Repo.open(":memory:", dimensions: 3)
    ids = {
      "x"  => repo.insert_entry("x", unit(1_f32, 0_f32, 0_f32)),
      "y"  => repo.insert_entry("y", unit(0_f32, 1_f32, 0_f32)),
      "z"  => repo.insert_entry("z", unit(0_f32, 0_f32, 1_f32)),
      "xy" => repo.insert_entry("xy", unit(1_f32, 1_f32, 0_f32)),
    }
    {repo, ids}
  end

  it "returns the nearest entries in order" do
    repo, ids = seeded_repo
    flat = Flat.new(repo)

    hits = flat.search(unit(1_f32, 0_f32, 0_f32), k: 2)
    expect(hits.size).to eq 2
    expect(hits[0].entry_id).to eq ids["x"]
    expect(hits[1].entry_id).to eq ids["xy"]
    expect(hits[0].score).to be > hits[1].score
    repo.close
  end

  it "scores an exact match at 1.0" do
    repo, _ = seeded_repo
    flat = Flat.new(repo)

    hit = flat.search(unit(0_f32, 0_f32, 1_f32), k: 1).first
    expect(hit.score).to be_close(1.0_f32, 1e-5)
    repo.close
  end

  it "ranks every entry when k exceeds the corpus" do
    repo, _ = seeded_repo
    flat = Flat.new(repo)

    hits = flat.search(unit(1_f32, 0_f32, 0_f32), k: 99)
    expect(hits.size).to eq 4
    scores = hits.map(&.score)
    expect(scores).to eq scores.sort.reverse
    repo.close
  end

  it "skips tombstoned entries" do
    repo, ids = seeded_repo
    repo.tombstone(ids["x"])
    flat = Flat.new(repo)

    hits = flat.search(unit(1_f32, 0_f32, 0_f32), k: 4)
    expect(hits.map(&.entry_id)).not_to contain ids["x"]
    expect(hits.size).to eq 3
    repo.close
  end

  it "honours an allowed set" do
    repo, ids = seeded_repo
    flat = Flat.new(repo)

    allowed = Set{ids["y"], ids["z"]}
    hits = flat.search(unit(1_f32, 0_f32, 0_f32), k: 4, allowed: allowed)
    expect(hits.map(&.entry_id).to_set).to eq allowed
    repo.close
  end

  it "returns nothing for an empty store or a non-positive k" do
    repo = Repo.open(":memory:", dimensions: dims)
    flat = Flat.new(repo)

    expect(flat.search(unit(1_f32, 0_f32, 0_f32), k: 3)).to be_empty
    expect(flat.size).to eq 0

    repo.insert_entry("x", unit(1_f32, 0_f32, 0_f32))
    expect(flat.search(unit(1_f32, 0_f32, 0_f32), k: 0)).to be_empty
    repo.close
  end

  it "needs no graph, so nothing to persist or rebuild" do
    repo, _ = seeded_repo
    flat = Flat.new(repo)

    expect(flat.kind).to eq :flat
    expect(flat.fully_persisted?).to be true
    expect(flat.size).to eq 4

    flat.flush
    flat.clear
    expect(repo.node_count).to eq 0
    expect(flat.size).to eq 4
    repo.close
  end
end
