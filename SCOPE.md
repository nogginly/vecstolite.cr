# Scope

Outstanding work, tracked in two buckets. **Completed items are deleted, not
ticked** — this file is a worklist, not a changelog. It should grow through the
early phases and dissolve as the design settles.

- **MUST FIX** — blocks progress, or is cheap now and expensive later.
- **WILL FIX** — real, but deliberately not now.

Anything settled belongs in code comments or design documentation; anything
outstanding belongs here, because nobody greps a codebase for open questions.

---

## MUST FIX

**Benchmark results (`DESIGN.md` §13).** The suites exist and `crossover` has
reported. Still to run at 10,000 entries: `cache` and `footprint`, which
produce the README sizing rule, then `ingest`, `restart` and `compact` for the
baseline in `DEVELOPMENT.md`. W1 ran with a 0.5 MB budget against a corpus of
tens of thousands — about 170 nodes at ~3 KB each — and nobody ever measured
what that cost.

**`page_size` is inside the schema freeze.** SQLite fixes it at creation, so it
cannot be changed by a later release without recreating the database. We
currently leave it at the 4096 default, which fits a 768-dimension Float32
vector. Confirm against benchmark results, then record the decision in
`DESIGN.md` §5.1.

**README sizing guidance.** `README.md` has a marked gap under "Memory" for a
budget-per-thousand-entries rule, waiting on the `cache` and `footprint`
benchmarks.

---

## WILL FIX

**`upsert` (`DESIGN.md` §14, question 2 — deferred from 0.7.0).** A changed
text means a new vector and a new graph position, so the entry's id *must*
change — and an `upsert` that returns a different id than the entry had is a
trap for any caller holding the old one. Delete-then-add says the same thing
in two lines, with the new id impossible to miss. Adding it later is a new
method, not a breaking change; revisit if idempotent re-ingest (W2b) wants it.

**Metadata filtering (`DESIGN.md` §8).** The `Filter` AST over `json_extract`,
with selectivity routing between post-filtered HNSW and an exact scan over
matching rows. Deferred to after 0.7.0: it is net new capability rather than
core, and no current workload needs it. `meta` is already JSON and the
selectivity routing touches no schema, so nothing about deferring it is
expensive. W2b (chunked document knowledge) is the likely first consumer.

**int8 vector quantisation (`DESIGN.md` §5.3).** The `encoding` metadata key is
already written, so this needs no migration. A 768-dimension vector drops from
3 KB to 768 bytes: 4× less storage, 4× more nodes per byte of cache. The
largest single lever for W2a and W2b, whose constraint is memory per agent
instance. Costs some recall, which the §7 harness can quantify.

**Automatic compaction.** W2a accumulates tombstones with no natural
maintenance window, and `compact!` being invisible to callers now (stable ids)
makes triggering it automatically safe. Needs a threshold with a number behind
it — benchmark 6.

**Crash recovery validation.** The atomicity argument is structural: every
operation writes its graph metadata inside the transaction that writes the rows
it describes. No test exercises an actual mid-transaction failure, which would
need fault injection into the repository rather than a black-box spec.

**Readonly stores cannot rebuild.** Opening readonly with an unsaved graph, or
with a different index strategy, raises rather than serving reads. Building the
graph in memory without persisting it would work, and is the obvious fix if a
readonly consumer appears.

**Suite runtime.** 28 seconds, dominated by the 500-entry volume specs and the
2,000-vector recall harness. Tag them, or shrink the corpora, if it starts
costing more than it catches.

**`StaticEmbedder#embed_all`.** Still one at a time. Cheap to batch through the
tokenizer and the embedding table, and worth doing when W2b's bursty document
ingest is real.

**`Index::Flat` allocates a vector per row.** Scanning 1,000 entries of 768
dimensions takes 6.72 ms, where the arithmetic alone — some 750k multiply-adds
— should take well under a millisecond. The cost is per-row: a SQLite fetch
plus `unpack_vector` building a fresh 3 KB slice that is discarded
immediately. A reusable buffer through `Repository#each_live_vector` would
likely take most of it back. Matters for the recall harness and for anyone
using Flat on a store of any size; does not affect HNSW, which reads a handful
of nodes per query.

**Drop the `simd` dependency.** Exactly one function is used — `dot`, at three
call sites (`vector_embedder.cr`, `index/strategy.cr`,
`embedder/static_embedder.cr`) — and only its scalar path, since CPU-specific
SIMD measured slower once Crystal 1.20 moved to an LLVM that auto-vectorises
scalar maths well. Upstream's scalar `dot`
(`spider-gazelle/simd`, `src/simd/scalar.cr:154`) is the naive bounds-checked
loop `acc += a[i] * b[i]`, with no fast-math or unrolling, so a local
`Vecstolite::Vec.dot` using `unsafe_fetch` should at worst match it.

Do it behind the existing `VECM` indirection, so switching back to hardware
SIMD stays a one-line change, and measure before removing the shard rather
than assuming: LLVM will only vectorise a float reduction if allowed to
reorder the additions, which strict IEEE semantics forbid. int8 quantisation
will need its own dot product regardless, and a small internal maths module
gives it somewhere to live.

**The hybrid storage engine (`DESIGN.md` §5.2).** Vectors and neighbours in an
append-only mmap'd side file, with SQLite keeping all truth about offsets.
Reachable because `Repository` owns every SQL statement. Only worth it if
benchmarks say so, and nothing currently says so.
