# Viz data (cluster nodes & transitions)

The eight files here are the aggregated inputs behind the interactive cluster
explorer (`../docs/`). They cover the full 2×2 of **population × clustering**:

| file stem | population (individuals) | clustering applied |
|---|---|---|
| `*_DK_pop_DK_clusters` | Danish | Danish |
| `*_DK_pop_AT_clusters` | Danish | Austrian |
| `*_AT_pop_DK_clusters` | Austrian | Danish |
| `*_AT_pop_AT_clusters` | Austrian | Austrian |

**`nodes_*.csv`** — one row per cluster (paper numbering, 0–131):
`cluster, mean_age, female_ratio, size, mortality_rate`.
`size` = number of person-year observations; `mortality_rate` is on the Danish
cohort only (blank for the Austrian population).

**`links_*.csv`** — cluster-to-cluster yearly transitions as an edge list:
`source, target, value`, where `value` is the row-stochastic transition
probability (per source cluster, self-loops included).

`nodes_*` also include **`size_alive`** (alive person-years, `year <= yod`) for the Danish-population files. `mortality_rate` is **alive-corrected** — `#deaths / size_alive` — matching the Austrian pipeline; it differs from the crude `#deaths / size` figure reported in the paper for high-mortality clusters.

### Disclosure control
All figures are cluster-level aggregates. Cells whose reconstructable count
(`value × size`, or `mortality_rate × size`) would fall below **5** are removed
(transitions) or blanked (mortality), matching the small-count suppression
required by the underlying Danish and Austrian register data agreements.
