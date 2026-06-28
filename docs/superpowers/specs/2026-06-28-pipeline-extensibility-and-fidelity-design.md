# LUXE FastSim — Pipeline Extensibility & Disc-Surface Fidelity

**Date:** 2026-06-28
**Status:** Approved design, ready for implementation planning

## Problem

Two struggles with the current WGAN-GP fast simulator, treated as symptoms of the
same per-region/per-particle architecture:

1. **Adding a new particle type is hard.** A particle is baked into three places
   per region: input CSVs split by both `pdg` and region (`{pdg}_{region}.csv`),
   a config per region that hardcodes `pdg` + `data_path`, and a from-scratch
   training of three separate region GANs. Adding a species means duplicating
   configs, re-splitting data, and hand-authoring per-feature transforms.
2. **Disc-surface marginals are slightly off.** The generated per-feature (1D)
   marginal distributions at the detector ("disc") surface are slightly
   smoothed/shifted; these small errors compound when the sample is propagated
   through Geant4 downstream.

## Goals & Non-Goals

**Goals**
- One parameterized command to add a particle type: auto data-split, auto config,
  auto-launch of the three region trainings, no source edits.
- Improve fidelity (measured by BED) using only changes that are ~free at
  generation time.
- Single source of truth for region boundaries shared by data-split and eval.

**Non-Goals (YAGNI — documented fallbacks only)**
- Merging the three regions into one model.
- Cross-particle conditioning (single conditional GAN over species).
- Changing the GAN loss or network architecture.
- Any inference-time refiner (normalizing flow / diffusion / learned corrector).

## Constraints (from brainstorming)

- **Scope:** one unified redesign covering both struggles.
- **Architecture:** keep three separate region models per particle.
- **Fidelity symptom:** slightly-off 1D **marginals** (not seams, not
  correlations — correlations are reportedly fine).
- **Inference budget:** essentially none. Fidelity fixes must be training-time or
  near-free post-hoc.
- **Extensibility shape:** automated *per-particle* pipeline (separate models),
  not a conditional model.
- **Transforms:** one shared default transform template reused across particles,
  with per-particle override when needed.
- **Success metric:** Energy Distance (BED), the existing multivariate two-sample
  test.

## Chosen Approach (A)

Config-driven pipeline refactor + generator EMA + post-hoc per-feature marginal
calibration. Rejected alternatives: B (A + training-quality overhaul — open-ended,
higher risk) and C (A + learned inference-time refiner — violates the inference
budget). B/C remain documented fallbacks if A's BED gains fall short.

---

## Part 1 — Extensibility: pipeline & config model

### Single entrypoint
A new `run_particle.py <pdg> [--epochs N] [--tag v6] [--regions inner outer1 outer2]`
replaces the hand-managed `FastSimCluster.py <epochs> <out> <stamp>` flow. It:

1. Resolves the input sample for `pdg` — uses pre-split `{pdg}_{region}.csv` if
   present, otherwise splits them on the fly from the master sample via a shared
   region-split function.
2. Materializes the three region configs from a base template + optional per-pdg
   override, injecting `pdg`, mass (from `pdg_dict`), `data_path`, and the
   region's spatial cuts.
3. Launches the three region trainings in the existing `outer1 → outer2 → inner`
   order, reusing `create_trainer` + `Trainer.run`.
4. Writes resolved `cfg_{region}.json`, models, and `.npy` log curves into the run
   dir in the layout `utils.check_run` already expects.

### Config model
Replace the 18 `{region}_cfg_v{N}.json` files with:
- `base_cfg.json` — shared hyperparameters, the shared default transform template,
  and the three region definitions (`dataGroup` + spatial-cut bounds).
- `overrides/{pdg}.json` — optional; only the fields a given particle differs on.

### Single source of truth for region boundaries
Region spatial cuts are currently duplicated and **inconsistent** between
`MakePDGPlots.py` (data split) and `utils.check_run` (eval filter) — e.g. the
split uses `yy >= 500` / `yy > 400` while `check_run` uses `yy >= 520` /
`yy <= 520`. Define each region's bounds once; both the splitter and the
evaluator import them so training data and evaluation always agree.

### Path cleanup
`cfg_dir`, the `GAN_Output` dir, and `data_path` roots move to a small resolvable
config (CLI/env with defaults) so the same code runs locally and on the cluster
without editing source.

---

## Part 2 — Fidelity: two near-free levers

Both target BED; both add ~zero inference cost.

### Lever 1 — Generator EMA (training-time only)
`Trainer` maintains a shadow copy of the generator weights, updated each generator
step as `ema = decay·ema + (1−decay)·live` (decay ≈ 0.999). The **EMA generator**
is validated, checkpointed on best validation, and saved as
`{region}_Gen_model.pt`. At inference it is an ordinary generator — no extra cost —
but its samples are markedly less noisy than any single training snapshot.

### Lever 2 — Post-hoc per-feature marginal calibration
Generation currently assumes the generator output is exactly standard-normal per
feature and feeds it straight into `QuantileTransformer.inverse_transform`; small
deviations from normal are amplified into off marginals. Fix:

- After training, push a large sample through the (EMA) generator to measure its
  **actual** per-feature output distribution in transformed space.
- Fit a 1D monotonic map per feature (empirical-CDF / a second
  `QuantileTransformer`) sending the generator's real marginal onto the target the
  inverse transform expects. Persist it next to the model (`{region}_calib.pkl`).
- Generation becomes: `generate → per-feature calibrate → QT.inverse →
  inverse log/flip`. One 1D interpolation per feature — essentially free.
- Each map is monotonic and per-feature, so it cannot disturb the joint/correlation
  structure (rank correlations preserved); it only sharpens marginals — exactly the
  reported symptom, which BED rewards.

**Ordering:** calibration is fit *after* EMA is selected as the deployed generator,
so it corrects the actual deployed marginals.

---

## Part 3 — Validation, success criteria, testing, migration

### Headline metric
Per-region **BED** (`get_batch_ed_histograms`) must beat the current v5 models on
the **photon (pdg 22)** baseline. Report BED at three stages — current model,
+EMA, +EMA+calibration — to attribute each lever's contribution.

### Guardrails
- Per-feature KS (`get_distance` / `plot_1d`) must not regress on any feature.
- Correlation plots must not visibly degrade.

### Acceptance criteria
- `run_particle.py 22` reproduces the current photon pipeline end-to-end, then
  shows improved BED with EMA + calibration enabled.
- Adding a new particle is a single command: split data, materialize 3 configs
  from template + override, train, calibrate, and report BED — no source edits,
  no per-region/per-version JSON authoring.

### Testing
- **Config materialization:** template + override → 3 valid region configs with
  correct `pdg` / mass / `data_path` / cuts.
- **Region split:** shared bounds partition the plane with no overlap and no gaps;
  splitter and `check_run` use the same bounds (regression test for the current
  inconsistency).
- **Calibration:** round-trip on a synthetic known distribution recovers the
  target marginal to tolerance; monotonicity preserves a known correlation.
- **EMA:** update math correct; saved model is the EMA weights, not the live ones.

### Migration
Convert existing v5 configs into one `base_cfg.json` + a photon override. Keep a
one-shot check that the refactored photon run reproduces current outputs *before*
enabling the fidelity levers, so the extensibility and fidelity changes are
validated independently.
