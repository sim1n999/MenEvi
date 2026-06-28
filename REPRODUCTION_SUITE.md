# Reproduction Suite

This file summarizes the public reproduction tasks shipped with EviMem. Each task has a semantic folder name and can be launched either directly or through `scripts/run_all_main.sh`.

| Task | Folder | Default inventory | Purpose |
|---|---|---:|---|
| Length-curve evaluation | `length_curve` | 32K/64K/128K/256K x 789 | Evaluate EviMem across MemLens context lengths |
| Strong-baseline evaluation | `strong_baselines` | 32K x 789 | Run matched BM25, caption, direct LVLM, and flat multimodal baselines |
| Automatic-routing evaluation | `automatic_routing` | 32K x 789 | Measure question-only route selection |
| Component-ablation evaluation | `component_ablations` | 32K x 789 | Measure visual-source, storage, graph-edge, and gate components |
| Efficiency profiling | `efficiency_profile` | 32K/64K/128K/256K x 789 | Measure wall time, observed GPU memory, and storage |

Run the full public suite:

```bash
source configs/local_paths.env
bash scripts/run_all_main.sh
```

Run one task:

```bash
source configs/local_paths.env
bash scripts/run_evimem_length_curve.sh
bash scripts/run_strong_baselines.sh
bash scripts/run_routing_analysis.sh
bash scripts/run_component_ablations.sh
bash scripts/run_efficiency_profile.sh
```