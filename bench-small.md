# Vecstolite benchmarks

- Crystal 1.21.0
- embedder: vecstolite/lexical, 768 dimensions
- corpus sizes: 1000, 10000

## Cache sizing

| corpus | budget | hit rate | 50 queries | per query |
|-------:|--------|---------:|-------------------:|----------:|
| 1000 | 0.5 MB | 25.6% | 62.70 ms | 1.25 ms |
| 1000 | 5 MB | 94.7% | 8.23 ms | 0.16 ms |
| 1000 | 50 MB | 94.7% | 8.17 ms | 0.16 ms |
| 1000 | all | n/a | 6.54 ms | 0.13 ms |
| 10000 | 0.5 MB | 6.2% | 194.93 ms | 3.90 ms |
| 10000 | 5 MB | 34.3% | 144.59 ms | 2.89 ms |
| 10000 | 50 MB | 79.0% | 50.06 ms | 1.00 ms |
| 10000 | all | n/a | 15.70 ms | 0.31 ms |

## Per-instance footprint

| corpus | cache | heap after queries | database |
|-------:|-------|-------------------:|---------:|
| 1000 | lru 0.5 MB | 0.0 MB | 4.2 MB |
| 1000 | lru 5 MB | 0.0 MB | 4.2 MB |
| 1000 | memory | 0.0 MB | 4.2 MB |
| 1000 | disk | 12.1 MB | 4.2 MB |
| 10000 | lru 0.5 MB | 0.0 MB | 41.6 MB |
| 10000 | lru 5 MB | 0.0 MB | 41.6 MB |
| 10000 | memory | 0.0 MB | 41.6 MB |
| 10000 | disk | 0.0 MB | 41.6 MB |

## Ingest

| corpus | config | style | total | per entry | database |
|-------:|--------|-------|------:|----------:|---------:|
| 1000 | hnsw m=16 memory | bulk | 1023.94 ms | 1.02 ms | 4.2 MB |
| 1000 | hnsw m=16 memory | single | 1126.33 ms | 1.13 ms | 4.2 MB |
| 1000 | hnsw m=8 memory | bulk | 467.65 ms | 0.47 ms | 4.2 MB |
| 1000 | hnsw m=8 memory | single | 569.81 ms | 0.57 ms | 4.2 MB |
| 1000 | hnsw m=16 lru | bulk | 1996.37 ms | 2.00 ms | 4.2 MB |
| 1000 | hnsw m=16 lru | single | 2653.57 ms | 2.65 ms | 4.2 MB |
| 1000 | flat | bulk | 12.63 ms | 0.01 ms | 4.1 MB |
| 1000 | flat | single | 87.57 ms | 0.09 ms | 4.1 MB |
| 10000 | hnsw m=16 memory | bulk | 15182.71 ms | 1.52 ms | 41.7 MB |
| 10000 | hnsw m=16 memory | single | 16118.29 ms | 1.61 ms | 41.6 MB |
| 10000 | hnsw m=8 memory | bulk | 7196.06 ms | 0.72 ms | 41.0 MB |
| 10000 | hnsw m=8 memory | single | 8187.45 ms | 0.82 ms | 41.0 MB |
| 10000 | hnsw m=16 lru | bulk | 44985.17 ms | 4.50 ms | 41.7 MB |
| 10000 | hnsw m=16 lru | single | 68243.06 ms | 6.82 ms | 41.6 MB |
| 10000 | flat | bulk | 113.33 ms | 0.01 ms | 40.1 MB |
| 10000 | flat | single | 983.81 ms | 0.10 ms | 40.0 MB |

## Open and first query

| corpus | cache | exit | open | first query |
|-------:|-------|------|-----:|------------:|
| 1000 | lru 0.5 MB | clean | 3.29 ms | 1.42 ms |
| 1000 | lru 0.5 MB | unclean | 2.85 ms | 3.14 ms |
| 1000 | memory | clean | 4.80 ms | 0.15 ms |
| 1000 | memory | unclean | 1021.92 ms | 0.14 ms |
| 1000 | disk | clean | 6.29 ms | 1.53 ms |
| 1000 | disk | unclean | 1.93 ms | 1.61 ms |
| 10000 | lru 0.5 MB | clean | 6.03 ms | 3.33 ms |
| 10000 | lru 0.5 MB | unclean | 2.92 ms | 3.21 ms |
| 10000 | memory | clean | 20.59 ms | 0.33 ms |
| 10000 | memory | unclean | 15291.57 ms | 0.26 ms |
| 10000 | disk | clean | 3.08 ms | 3.34 ms |
| 10000 | disk | unclean | 2.96 ms | 3.20 ms |

## Flat against HNSW

Two recall figures, because they disagree when scores tie. *ids* is
the share of exact results the graph also returned. *scores* is the
share of graph results scoring at least as well as the exact k-th —
an equally good answer that happens to be a different entry counts
here but not above. Sparse embeddings tie constantly, so *ids*
understates quality on them.

| corpus | strategy | 50 queries | per query | recall (ids) | recall (scores) | database |
|-------:|----------|-------------------:|----------:|-------------:|----------------:|---------:|
| 1000 | flat | 313.06 ms | 6.26 ms | 1.000 | 1.000 | 4.1 MB |
| 1000 | hnsw | 6.07 ms | 0.12 ms | 0.504 | 0.960 | 4.2 MB |
| 10000 | flat | 2418.17 ms | 48.36 ms | 1.000 | 1.000 | 40.1 MB |
| 10000 | hnsw | 15.76 ms | 0.32 ms | 0.712 | 0.960 | 41.7 MB |

## Deletion and compaction

| corpus | tombstones | search (50) | compact! | before | after |
|-------:|-----------:|-------------------:|---------:|-------:|------:|
| 1000 | 10% | 8.00 ms | 901.56 ms | 5.7 MB | 4.1 MB |
| 1000 | 25% | 10.23 ms | 738.17 ms | 4.2 MB | 4.1 MB |
| 1000 | 50% | 13.11 ms | 449.46 ms | 4.2 MB | 4.1 MB |
| 10000 | 10% | 20.86 ms | 13332.28 ms | 40.9 MB | 40.9 MB |
| 10000 | 25% | 28.33 ms | 10711.98 ms | 41.0 MB | 40.9 MB |
| 10000 | 50% | 41.74 ms | 6600.33 ms | 41.2 MB | 40.9 MB |
