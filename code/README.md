# Code

```
code/
├─ core/       algorithm — graph operations, potential R, route completion, scoring
├─ pipeline/   Amelia-10 adaptation — graph build, track cleaning, batch runner
├─ analysis/   selection of the released flights from a full run
├─ viewer/     single-file HTML viewer generator
└─ scripts/    verify_dataset.py, plot_from_dataset.py (self-contained, run from the repo root)
```

All modules resolve the dataset root as

```python
ROOT = os.environ.get("ROUTE_ID_ROOT") or <repository root>
```

so nothing depends on the working directory. Set `ROUTE_ID_ROOT` only if the data lives
elsewhere.

## core — the algorithm

| file | role |
|---|---|
| `main.py` | entry point: batch over a track directory, writes the five-stage outputs |
| `config.py` | every threshold and path (edit here, nothing else) |
| `start_end_extractor.py` | departure/arrival type, runway, start/end segment |
| `coordinate_transform.py` | latitude/longitude → local metric frame (6-parameter affine) |
| `potential_r_matcher.py` | 15 m grid match against R segments + 10° turn filter |
| `angle_constraints.py` | inter-segment angle at the shared endpoint, 110° legality rule |
| `graph_pathfinder.py` | Dijkstra / A* completion between two segments |
| `path_combinator.py` | branch on the potential-R sequence, keep legal combinations |
| `best_path_search.py` | score candidates by Σ point-to-route distance |
| `param_validate.py` | sweep the angle threshold and compare routes |
| `visualize.py` | static HTML viewer from the `output/` tree |

Run on one track (input format: the track JSON described in `data/README.md`):

```bash
cd code/core
python main.py --input-dir <track_dir> --single <tid>.json --output-dir <out_dir>
```

Requires the five graph files (`InitMatrixPath.json`, `InitMatrixPath-1m.json`,
`InitStand.json`-style stand data, runway data, adjacency matrix) as described in
`config.py`. Amelia-10 provides no stand/runway files, which is why the pipeline below patches
start/end extraction instead of using `main.py` directly.

## pipeline — Amelia-10 adaptation

| file | role |
|---|---|
| `build_paths_amelia.py` | `MultiDiGraph` → N/R segment set; splits at degree≠2 nodes and at curvature transitions (0.45 °/m), then classifies with the 5° rule |
| `prep_traj_amelia.py` | track cleaning L0–L4 (aircraft only, break at gaps > 30 s, cut the taxiing portion, drop off-graph ends) and export of a 500-flight sample |
| `amelia_adapter.py` | coordinate frame + start/end anchoring helpers |
| `run_amelia.py` | single-process runner; patches three points of `main.py` |
| `run_amelia_par.py` | same, multi-process, with `--resume` and `--tids` |

```bash
python code/pipeline/build_paths_amelia.py kdca
python code/pipeline/prep_traj_amelia.py kdca --export
python code/pipeline/run_amelia_par.py KDCA --graph-subdir graph_std_skel --workers 4
```

`run_amelia_par.py` expects `<ROOT>/<AIRPORT>/{graph_std_skel, traj_raw, sample_list.json}` —
which is exactly the layout under `data/`, so the released flights can be re-run with:

```bash
cd code/pipeline
python run_amelia_par.py KDCA --out-root ../../data --graph-subdir graph_std_skel \
       --tids ../../data/KDCA/tids.txt --tag rerun --workers 4
```

The three patched points, in short: the dataset already ships metric coordinates (no affine
transform), it has no stand geometry (start/end segments come from the first/last track point),
and the potential-R extractor takes its coordinates from those two. The graph operations —
angle constraints, path finding, combination, scoring — are used unchanged.

## analysis

| file | role |
|---|---|
| `pick_flights.py` | rank every run by `med + 0.02·p95`, drop fallbacks, write `flights.json`, `tids.txt` and `all_runs.json` |

It reads `<ROOT>/<AIRPORT>/full_std_skel/results.json`, so it only runs after a full 500-flight
pass; the released `data/` already contains its output.

## viewer

`gen_viewer_cn.py` builds a self-contained HTML viewer (all data embedded, no server): layer
toggles, point-size / sampling-step / route-width / label-size / legend-scale sliders, and PNG
(1×, 2×) plus SVG vector export.

```bash
python code/viewer/gen_viewer_cn.py --ap KDCA --traj traj_raw --tids ../KDCA/tids.txt \
       --out KDCA_viewer.html
```

## scripts

| file | role |
|---|---|
| `verify_dataset.py` | cross-checks every released file: ids, point counts, route continuity, recomputed median |
| `plot_from_dataset.py` | point-to-route ECDF from `data/<AIRPORT>/residuals/` |

Both run with the released data only, no download:

```bash
python code/scripts/verify_dataset.py
python code/scripts/plot_from_dataset.py --airport both --out ecdf.png
```

## Dependencies

`numpy`, `scipy`, `pandas`, `networkx`, `matplotlib` (see `requirements.txt`). Developed with
Python 3.12 / numpy 1.26; no GPU and no learned model.
