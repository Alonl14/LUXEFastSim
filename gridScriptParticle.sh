#! /bin/bash
#PBS -m n
#PBS -l walltime=72:00:00 -l io=3

# Runs the new per-particle pipeline (run_particle.py) on the cluster.
# Trains all three regions for one PDG, with EMA + post-hoc marginal calibration.
#
# PBS -v variables:
#   parname1 -> PDG code (e.g. 22, 11, 2112)
#   parname2 -> project working directory (repo checkout, where run_particle.py lives)
#   parname3 -> output directory for this run (e.g. /.../GAN_Output/run_123/)
#   parname4 -> epochs
#   parname5 -> (optional) data root override (dir holding {pdg}_{region}.csv)

echo "[grid] Setting up Conda…"
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common

Pdg="${parname1}"
Directory="${parname2}"
OutputDir="${parname3}"
NEpoch="${parname4}"
DataRoot="${parname5:-}"

echo "[grid] PDG      : ${Pdg}"
echo "[grid] WORKDIR  : ${Directory}"
echo "[grid] OUTDIR   : ${OutputDir}"
echo "[grid] EPOCHS   : ${NEpoch}"
[[ -n "${DataRoot}" ]] && echo "[grid] DATA_ROOT: ${DataRoot}"

cd "${Directory}"
echo "[grid] PWD = ${PWD}"

EXTRA_ARGS=()
if [[ -n "${DataRoot}" ]]; then
  EXTRA_ARGS+=(--data-root "${DataRoot}")
fi

echo "[grid] Launching training…"
time python3 run_particle.py "${Pdg}" \
  --epochs "${NEpoch}" \
  --output-dir "${OutputDir}" \
  "${EXTRA_ARGS[@]}"
echo "[grid] Done."
