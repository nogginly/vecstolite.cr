# Vecstolite benchmarks

- Crystal 1.21.0
- embedder: vecstolite/lexical, 768 dimensions
- corpus sizes: 100000

## Cache sizing

| corpus | budget | hit rate | 50 queries | per query |
|-------:|--------|---------:|-------------------:|----------:|
| 100000 | 0.5 MB | 7.6% | 216.67 ms | 4.33 ms |
| 100000 | 5 MB | 23.7% | 184.84 ms | 3.70 ms |
| 100000 | 50 MB | 47.5% | 127.68 ms | 2.55 ms |
| 100000 | all | n/a | 18.41 ms | 0.37 ms |
# Vecstolite benchmarks

- Crystal 1.21.0
- embedder: vecstolite/lexical, 768 dimensions
- corpus sizes: 100000

## Per-instance footprint

| corpus | cache | heap after queries | database |
|-------:|-------|-------------------:|---------:|
| 100000 | lru 0.5 MB | 0.0 MB | 416.3 MB |
| 100000 | lru 5 MB | 0.0 MB | 416.3 MB |
| 100000 | memory | 257.0 MB | 416.3 MB |
| 100000 | disk | 0.0 MB | 416.3 MB |
# Vecstolite benchmarks

- Crystal 1.21.0
- embedder: vecstolite/lexical, 768 dimensions
- corpus sizes: 100000

## Open and first query

| corpus | cache | exit | open | first query |
|-------:|-------|------|-----:|------------:|
| 100000 | lru 0.5 MB | clean | 7.02 ms | 4.95 ms |
| 100000 | lru 0.5 MB | unclean | 3.69 ms | 4.56 ms |
| 100000 | memory | clean | 310.90 ms | 0.50 ms |
| 100000 | memory | unclean | 143978.20 ms | 0.51 ms |
| 100000 | disk | clean | 3.92 ms | 4.93 ms |
| 100000 | disk | unclean | 3.72 ms | 4.78 ms |
# Vecstolite benchmarks

- Crystal 1.21.0
- embedder: vecstolite/lexical, 768 dimensions
- corpus sizes: 100000

## Flat against HNSW

Two recall figures, because they disagree when scores tie. *ids* is
the share of exact results the graph also returned. *scores* is the
share of graph results scoring at least as well as the exact k-th —
an equally good answer that happens to be a different entry counts
here but not above. Sparse embeddings tie constantly, so *ids*
understates quality on them.

| corpus | strategy | 50 queries | per query | recall (ids) | recall (scores) | database |
|-------:|----------|-------------------:|----------:|-------------:|----------------:|---------:|
| 100000 | flat | 27119.45 ms | 542.39 ms | 1.000 | 1.000 | 400.7 MB |
| 100000 | hnsw | 19.32 ms | 0.39 ms | 0.200 | 0.472 | 417.1 MB |
