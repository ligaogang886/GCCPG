# Taxiway Route Identification from ADS-B Trajectories

Reference implementation and a 40-flight benchmark for recovering the **taxi route an aircraft
actually followed** from its surface trajectory, on an airport movement graph whose segments are
labelled **N** (straight) or **R** (turn).

Given a noisy 1 Hz ADS-B track and a pre-built graph, the pipeline

1. anchors the start and end segments of the track,
2. collects **potential R** segments — turn segments the aircraft is likely to have used,
3. completes candidate routes between them with an angle-constrained shortest path,
4. scores every candidate by the point-to-route distance and keeps the best one.

No map-matching library, no learned weights, no per-airport tuning: the same thresholds are used
for both airports in this release.

```
ADS-B track ──► start / end anchoring ──► potential R set ──► candidate routes ──► best route
                (first & last point)      (15 m grid match    (angle ≥ 110°      (min Σ point-to-
                                           + 10° turn filter)   between segments)  route distance)
```

---

## Repository layout

```
taxi-route-identification/
├─ code/
│  ├─ core/        algorithm: graph, potential R, route completion, scoring, viewer
│  ├─ pipeline/    dataset adaptation: graph build, track cleaning, batch runner
│  ├─ analysis/    selection of the released flights from a full run
│  └─ scripts/     verify_dataset.py, plot_from_dataset.py   (run out of the box)
└─ data/
   ├─ KDCA/        graph, 20 tracks, identified routes, residuals, metrics
   └─ KLAX/        same
```

Field-by-field description of every file under `data/`: [`data/README.md`](data/README.md).
What each script does: [`code/README.md`](code/README.md).

---

## Data at a glance

Both airports come from the [Amelia-10](https://huggingface.co/datasets/AmeliaCMU/Amelia-10)
surface-trajectory dataset. For each airport 500 flights were sampled, run through the pipeline,
and a subset of 20 flights with complete geometry is released here.

| | KDCA (Reagan National) | KLAX (Los Angeles Intl.) |
|---|---|---|
| Graph segments | 369 (N 269 / R 100) | 989 (N 614 / R 375) |
| Graph length | 23.7 km | 91.3 km |
| Runs / routes returned | 500 / 433 | 500 / 273 |
| Routes with a matched R | 389 | 265 |
| Released flights | 20 | 20 |
| Track points released | 6 277 | 8 202 |
| Median distance, released flights | 0.69 – 0.98 m | 0.65 – 1.05 m |
| Median distance, all non-fallback routes | 1.86 m | 1.85 m |
| Distance ECDF, P50 / P90 / ≤3 m | 0.91 m / 2.33 m / 94.4 % | 0.98 m / 2.72 m / 92.4 % |

The released flights are the ones whose geometry is complete; they are **not** a random sample.
`data/<AIRPORT>/all_runs.json` holds the score of every run that returned a route (433 / 273
rows), which is the unbiased distribution to quote: its median is 1.86 m / 1.85 m, about twice
the released subset.

Per-flight metrics: `data/KDCA/metrics.csv`, `data/KLAX/metrics.csv`.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. check every released file for consistency  (no downloads, ~1 s)
python code/scripts/verify_dataset.py

# 2. redraw the point-to-route ECDF from the released residuals
python code/scripts/plot_from_dataset.py --airport both --out ecdf.png

# 3. re-run the algorithm on the released flights of one airport
cd code/pipeline
python run_amelia_par.py KDCA --out-root ../../data --graph-subdir graph_std_skel --workers 4
```

Step 1 prints, per airport, the number of flights, points and graph segments, plus the largest
deviation between the tabulated median residual and the value recomputed from
`data/<AIRPORT>/residuals/` (currently ≤ 0.005 m). Step 3 takes about 6 s per flight on a
laptop CPU and writes `data/<AIRPORT>/<tag>/results.json`.

---

## Metrics

Every track point is assigned its distance to the identified route (graph edges densified to 1 m):

| metric | meaning |
|---|---|
| `med` | median point-to-route distance (m) — headline accuracy |
| `p95`, `max` | tail of the same distribution |
| `cov15` | share of points within 15 m |
| `floor` | median distance from the track to the **whole graph** — the geometric lower bound no routing method can beat |
| `excess` | `med − floor`, i.e. what route selection itself costs |
| `fallback` | `true` when the returned route contains no matched R (degenerate shortest path) |

`excess` is the metric to compare across airports: `med` alone is dominated by how far the
aircraft actually flies from the centreline.

---

## Algorithm

`code/core` implements four stages; `config.py` holds every threshold.

| stage | module | what it does |
|---|---|---|
| 0 | `start_end_extractor.py` | infer departure/arrival type, runway and start/end segments |
| 1 | `coordinate_transform.py` | latitude/longitude → local metric frame |
| 2 | `potential_r_matcher.py` | 15 m grid match against R segments, keep points whose 5-point window turns more than 10° |
| 3 | `path_combinator.py` + `graph_pathfinder.py` | branch on the potential-R sequence, complete each branch by shortest path, reject transitions whose inter-segment angle is below 110° |
| 4 | `best_path_search.py` | score candidates by Σ point-to-route distance, return the minimum |

Key parameters (`code/core/config.py`): `R_MATCH_DIST = 15 m`, `R_TOP_K = 3`,
`ANGLE_THRESHOLD = 10°`, `PATH_ANGLE_THRESHOLD = 110°`, plus a reversal-triggered fallback that
re-runs a flight at `ANGLE_THRESHOLD_ESCALATED = 7°` when the route enters and leaves the same
segment twice.

Airport-specific inputs are limited to the graph and the coordinate frame. `code/pipeline`
contains the Amelia-10 adapter: it rebuilds the graph as an N/R segment set, cleans the tracks,
and patches exactly three points of `main.py` (identity coordinate transform because the dataset
already ships `unity_x/unity_z`, data-driven start/end anchoring because the dataset has no stand
geometry, and the coordinate source of the potential-R extractor). The four graph operations above
are reused unchanged.

---

## Reproducing the experiment

```bash
# graph: Amelia-10 MultiDiGraph -> N/R segment set
python code/pipeline/build_paths_amelia.py kdca
python code/pipeline/build_paths_amelia.py klax

# tracks: clean + sample 500 flights, convert to the algorithm's track format
python code/pipeline/prep_traj_amelia.py kdca --export
python code/pipeline/prep_traj_amelia.py klax --export

# run all 500 flights (4 workers, ~6 s per flight)
python code/pipeline/run_amelia_par.py KDCA --graph-subdir graph_std_skel --workers 4
python code/pipeline/run_amelia_par.py KLAX --graph-subdir graph_std_skel --workers 4

# rank the runs and write the 20 released flights
python code/analysis/pick_flights.py
```

The pipeline scripts read the dataset root from `ROUTE_ID_ROOT` (default: the repository root),
so the downloaded Amelia-10 raw files must be laid out as expected by
`prep_traj_amelia.py` — see the header of that file.

---

## Data source and licence

Trajectories and airport graphs derive from **Amelia-10** (Carnegie Mellon University):

- dataset: <https://huggingface.co/datasets/AmeliaCMU/Amelia-10>
- project page: <https://ameliacmu.github.io>
- paper: *Amelia: A Large-Scale Dataset for Airport Surface Movement* (arXiv:2407.21185)
- tools: <https://github.com/AmeliaCMU/AmeliaScenes>

Only the 40 released flights, their graphs and the derived results are redistributed here; the
full dataset (~13.5 GB, five airports) is **not** included. Please follow the original dataset
licence when reusing the trajectories.

Code in this repository is released under the MIT licence ([`LICENSE`](LICENSE)).

---

## Citation

```bibtex
@misc{taxi_route_id_2026,
  title  = {Taxiway Route Identification from ADS-B Trajectories},
  year   = {2026},
  note   = {Code and a 40-flight benchmark on KDCA and KLAX},
  url    = {https://github.com/<your-account>/taxi-route-identification}
}
```

Please also cite the Amelia-10 dataset above.

---

## Limitations

- The graph is built from the dataset's OSM-derived `MultiDiGraph`; junctions are sharp single
  vertices, so turn segments are shorter and fewer than in a survey-grade taxiway map.
- `f1` / `recall` / `precision` fields are `null`: Amelia-10 provides no ground-truth route, so
  the evaluation is purely geometric.
- A few released routes join their first or last segment with a straight jump of 20–60 m
  (one flight 691 m) instead of a shared node; the jump comes from start/end anchoring and is
  uncorrelated with `med` (r = 0.00–0.21). The seams are listed by `verify_dataset.py`.
