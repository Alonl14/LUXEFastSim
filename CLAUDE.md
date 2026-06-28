# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A WGAN-GP fast-simulation generator for the LUXE experiment. It learns the
distribution of background particles (photons/neutrons/electrons, selected by
PDG code) hitting a detector plane and generates synthetic events far faster
than full Monte Carlo. The detector face is split into three spatial **regions**
that are trained as separate models: `outer1`, `outer2`, `inner`.

## Commands

There is no build system, test suite, or linter. It is a set of scripts run
against a Python venv (`./venv`, torch + sklearn + pandas + scipy + numba +
psutil + matplotlib).

Train all three regions (the main entrypoint):
```bash
python FastSimCluster.py <num_epochs> <output_dir> <config_stamp>
# e.g.  python FastSimCluster.py 200 Output/run_42/ _cfg_v5.json
```
`config_stamp` is the suffix appended to each region name to find its config,
so `_cfg_v5.json` loads `Config/inner_cfg_v5.json`, `Config/outer1_cfg_v5.json`,
`Config/outer2_cfg_v5.json`. Regions train in order outer1 → outer2 → inner.

Analyze / evaluate a finished run:
```bash
python AnalyzeRun.py <run_id> [--calculate_BED] [--save_df] [--plot_metrics] [--plot_results]
# run_id is the suffix of Output/run_<run_id>/ (or the cluster GAN_Output dir)
```

## Pipeline architecture

The training flow is a small chain — read these four files together to
understand a run:

1. **`FastSimCluster.py`** — launcher. Loads the three region configs, patches
   in `outputDir`/`numEpochs`, builds a trainer per region, runs them
   sequentially, then dumps per-region training-curve logs as `.npy`
   (`KL_*`, `D_*`, `G_*`, `GP_*`, grad norms) into the output dir.
2. **`trainerFactory.create_trainer(cfg)`** — builds the `ParticleDataset`,
   does an 80/20 train/val `random_split`, constructs `DataLoader`s, picks the
   device, instantiates `Generator2`/`Discriminator2` wrapped in
   `nn.DataParallel`, sets up Adam optimizers, and returns a `Trainer`. Note it
   reassembles a *flat* `cfgDict` that the `Trainer` consumes — different keys
   than the JSON config.
3. **`trainer.Trainer.run()`** — the WGAN-GP loop: `nCrit` critic steps per
   generator step, gradient penalty (`compute_gradient_penalty`, or R1 if
   `gradMetric != 'norm'`), decaying instance noise on real+fake, LR cosine
   warm-up, periodic KL logging, checkpoint-on-best-validation-D, and a
   flat-slope early stop. Best models are saved as
   `<outputDir><dataGroup>_Gen_model.pt` / `_Disc_model.pt`.
4. **`dataset.ParticleDataset`** — reads the region CSV, calls
   `utils.add_features` to compute derived physics columns, selects only
   `cfg["features"]`, applies per-feature transforms from the registry
   (`log`, `flip`) in listed order, then fits a sklearn `QuantileTransformer`
   (→ normal) used during training and inverted during generation.

`Generator`/`Discriminator` (v1) are legacy; the active models are
`Generator2` (BatchNorm+ReLU MLP) and `Discriminator2` (LayerNorm critic, no
final bias — a proper WGAN critic). Loading code in
`utils.generate_fake_real_dfs` tries `Generator` first and falls back to
`Generator2` on a state-dict mismatch.

## Evaluation (`utils.check_run`)

`AnalyzeRun.py` → `utils.check_run` regenerates fake+real DataFrames for all
three regions, applies the **region-defining spatial cuts** (inner / outer1 /
outer2 are carved out by `xx`/`yy`/`rx` thresholds in `check_run`), and can
compute the Energy Distance / "BED" two-sample test
(`get_batch_ed_histograms`, numba-accelerated `euclidean_distance_matrix`), KL
divergence (`get_kld`), and 1-D KS distances, plus correlation/feature plots.

## Gotchas

- **Feature column names have leading spaces** (`" xx"`, `" pzz"`, `" time"`,
  etc.) — this matches the input CSV headers and is load-bearing throughout
  `dataset.py`, `utils.add_features`, and the config `features` maps. Preserve
  them exactly.
- **Hardcoded cluster paths.** `FastSimCluster.py` reads configs from
  `/srv01/agrp/alonle/LUXEFastSim/Config`; `AnalyzeRun.py` points at
  `/storage/agrp/alonle/LUXE_FastSim/GAN_Output`; config `data_path` fields
  point at `/storage/agrp/...`. For local runs, `utils.fix_path` rewrites
  `data_path` when `check_run` is called with `path=None`.
- **Models are `nn.DataParallel`-wrapped**, so saved state dicts have a
  `module.` prefix — load into a `DataParallel`-wrapped net, not a bare one.
- `add_features` derives energy/angles from base kinematics using the
  particle's mass looked up in `utils.pdg_dict` / `particle_dict` by the
  config `pdg`, so the config `pdg` must match the input sample.
- `Config/inner_cfg_local.json` and `*_local.json` variants are for CPU/local
  testing (small device, local paths).

## Layout notes

- `Config/` — per-region, per-version JSON configs (`{region}_cfg_v{N}.json`).
- `Output/run_*/` — trained checkpoints, copied configs, and `.npy` log curves.
- `Z_nonPipeline/`, `PaperStuff.ipynb`, `make_paper_plots.py`,
  `MakePDGPlots.py` — exploratory notebooks and paper-figure scripts, not part
  of the training pipeline.
- `ArchiveRuns/`, `WGAN_run*_DetId31/` — archived run artifacts and figures.
