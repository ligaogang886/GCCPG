# Dataset

Two airports, 20 released flights each. Everything is derived from the Amelia-10 trajectory
dataset; the original dataset itself is **not** included.

```
data/<AIRPORT>/
├─ graph_std_skel/          airport movement graph (N = straight, R = turn)
│  ├─ InitMatrixPath.json   segments: name, endpoints, length, polyline   <- primary graph
│  ├─ InitMatrixPath-1m.json same graph densified to 1 m (potential-R matching)
│  ├─ InitPath.json         byte-identical copy of InitMatrixPath.json (kept for legacy code)
│  ├─ path_meta.json        per-segment attributes: kind, class, is_runway, max turn
│  ├─ connection_matrix.csv binary adjacency (1 = two segments share an endpoint)
│  └─ build_report.json     graph statistics (segment counts, lengths, components)
├─ traj_raw/                20 tracks, one file per flight
├─ residuals/               20 point-to-route distance series
├─ full_std_skel/
│  └─ results.json          algorithm output rows, including `best_path`
├─ metrics.csv              20 rows, one per flight, tabular metrics
├─ flights.json             same 20 flights with the full metric record
├─ all_runs.json            score of every run that returned a route (433 / 273 rows)
├─ sample_list.json         track metadata (duration, length, start/end hints)
└─ tids.txt                 the 20 flight ids, in file order
```

## Coordinates and time

- `unity_x` / `unity_z` are **metric** coordinates in the dataset's local frame (metres, x east,
  z north). They are the same frame as the graph polyline `x` / `z`. Use these for any geometric
  computation; `lat` / `lon` are kept only for reference.
- Tracks are 1 Hz, one row per second, ordered in time. The `time` field of the raw dataset is
  only refreshed in blocks (it repeats for ~128 consecutive rows), so the **row index `t`**
  is the reliable time axis: point *i* is at *t0 + i* seconds. `residuals/*.json` uses that index.

## Graphs

`InitMatrixPath.json`:

```json
{"paths": [{"pathname": "N-1",
            "startpoint": "Point (4)", "endpoint": "Point (6)",
            "pathLength": 33.885,
            "points": [{"pointname": 0, "x": 1019.013, "y": 0.0, "z": -211.996, "roa": 0.0}]}]}
```

Segment names start with `N` (straight, largest internal turn < 5°) or `R` (turn) — the same rule
used for every airport, so the labels are comparable across sites. `path_meta.json` adds
`kind`, `cls_name` (`runway` / `stopway` / `taxilane` / `taxiway`), `is_runway` and `maxturn`.

| | KDCA | KLAX |
|---|---|---|
| Segments | 369 | 989 |
| — N / R | 269 / 100 | 614 / 375 |
| Total length | 23.7 km | 91.3 km |
| R share of length | 27.1 % | 37.9 % |
| Endpoints (nodes) | 250 | 652 |
| Graph points / at 1 m | 4 395 / 24 124 | 5 761 / 92 278 |
| Adjacency edges | 792 | 2 188 |

## Tracks (`traj_raw/<tid>.json`)

A list of points, one per second, in the same schema as the pipeline input:

```json
[{"actp": "Track", "time": "1682530048", "t": 0,
  "properties": {"lat": 38.850636, "lon": -77.037674,
                 "unity_x": -156.86, "unity_z": -120.40,
                 "spd_kn": 38.2, "callsign": "1052984",
                 "flio": "A", "rway": ""}}]
```

`flio` is `A` (arrival) or `D` (departure); `callsign` is the flight identifier used by the
dataset (a number, not an airline callsign). The released tracks keep only the fields needed for
route identification and for redrawing the figures.

## Algorithm output (`full_std_skel/results.json`)

`summary` holds the run configuration (thresholds, worker count, run time); `rows` holds one
record per flight. The field that matters is `best_path`: the ordered list of segment names the
algorithm selected. Concatenating those polylines (orienting each segment along the running end
point) reproduces the identified route.

Other row fields: `n_track_points`, `track_len_m`, `dur_s`, `n_candidates`, `median_dist_m`,
`p90_dist_m`, `p95_dist_m`, `max_dist_m`, `cov_10m / cov_15m / cov_30m`,
`lb_mean_dist_m / lb_p95_dist_m` (lower bound against the whole graph), `n_route_paths`,
`route_len_m`, `n_reversal`, plus per-stage timings.

## `metrics.csv`

One row per flight, `rank` = position in the selection.

| column | meaning |
|---|---|
| `tid` | flight id, matches the file names |
| `callsign` | dataset flight identifier |
| `flio` | `A` / `D` |
| `n` | number of track points (= rows in `residuals`) |
| `med`, `p95`, `max` | point-to-route distance: median / 95th percentile / maximum (m) |
| `med_recheck` | median recomputed from `residuals/<tid>.json` (differs from `med` by ≤ 0.09 m, see below) |
| `cov15` | share of points within 15 m (%) |
| `floor` | median distance from the track to the whole graph (m) — geometric lower bound |
| `excess` | `med − floor` (m) |
| `npaths` | number of segments in the identified route |
| `nR` | number of matched R segments behind that route |
| `ncand` | number of candidate routes evaluated |
| `track_len_m`, `route_len_m`, `dur_s` | track length, route length, duration |

`flights.json` repeats these fields and adds `nrev` (segment reversals), `rlen`, `tlen`, `dur`,
`fallback`, and a legacy group `med_res` / `p95_res` / `cov15_res` / `f1` / `recall` /
`precision` kept from an earlier re-matching pass (`f1`, `recall`, `precision` are `null`: the
dataset has no ground-truth route).

`all_runs.json` is the same record for **every** run that returned a route, sorted by `med`.
It is the unbiased distribution: the 20 released flights were picked from its low end.

## `residuals/<tid>.json`

```json
{"tid": "KDCA_1052984_1682530039", "airport": "KDCA", "callsign": "1052984",
 "unit": {"t": "s (1 Hz row index)", "d": "m"},
 "t": [0, 1, 2, ...], "d": [0.91, 1.37, 1.56, ...]}
```

`d[i]` is the distance from track point *i* to the identified route, with the route densified to
1 m. This is exactly what `code/scripts/plot_from_dataset.py` plots.

## Notes

- **`med` vs `med_recheck`.** `med` is the value quoted in the paper figures (computed against
  the 1 m graph used at experiment time). `med_recheck` is recomputed here against the released
  graph and runs 0.008–0.088 m higher (mean +0.04 m on KDCA, +0.03 m on KLAX), which is a
  version difference of the 1 m densification, not a data error. `verify_dataset.py` checks that
  the recomputed median matches `med_recheck` to within 0.02 m.
- **Route seams.** Most routes are node-continuous. A few flights join their first or last
  segment with a straight gap of 20–60 m, and one KLAX flight starts with a 691 m runway segment
  that does not touch the next one; the gap comes from start/end anchoring. It is uncorrelated
  with `med` (r = 0.00 on KDCA, 0.21 on KLAX) and is reported by `verify_dataset.py`.
- **Selection bias.** The 20 released flights are the best-fitting ones, so their median
  (0.7–1.0 m) is roughly half the median over all valid runs (1.8–1.9 m). Use `all_runs.json`
  whenever a distribution rather than an example is needed.

## Licence

The trajectories and graphs derive from Amelia-10 and remain subject to its terms; see
<https://huggingface.co/datasets/AmeliaCMU/Amelia-10>. The derived tables and residuals in this
directory are provided for research use, alongside the MIT-licensed code.
