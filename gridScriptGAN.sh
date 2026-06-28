#! /bin/bash
#PBS -m n
#PBS -l walltime=72:00:00 -l io=3

echo "[grid] Setting up Conda…"
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common

# --------- PBS -v variables ----------
# parname1 -> epochs
# parname2 -> project working directory (where fastsim_cluster_main.py lives)
# parname3 -> output directory for this run (e.g. /.../GAN_Output/run_123/)
# parname4 -> config STAMP (e.g. cfg_cluster OR _cfg_cluster.json)
# parname5 -> (optional) config directory path (defaults to trainer main’s built-in)
n_epoch="${parname1}"
Directory="${parname2}"
OutputDir="${parname3}"
ConfigStamp="${parname4}"
CfgDir="${parname5:-}"

# Normalize stamp: add leading "_" and trailing ".json" if missing
normalize_stamp() {
  local s="$1"
  [[ "${s}" != _* ]] && s="_${s}"
  [[ "${s}" != *.json ]] && s="${s}.json"
  printf "%s" "$s"
}
CONFIG_STAMP="$(normalize_stamp "$ConfigStamp")"

echo "[grid] EPOCHS   : ${n_epoch}"
echo "[grid] WORKDIR  : ${Directory}"
echo "[grid] OUTDIR   : ${OutputDir}"
echo "[grid] CFG_STAMP: ${CONFIG_STAMP}"
[[ -n "${CfgDir}" ]] && echo "[grid] CFG_DIR  : ${CfgDir}"

echo "[grid] cd → ${Directory}"
cd "${Directory}"
echo "[grid] PWD = ${PWD}"

# Build optional args
EXTRA_ARGS=()
if [[ -n "${CfgDir}" ]]; then
  EXTRA_ARGS+=(--cfg_dir "${CfgDir}")
fi

echo "[grid] Launching training…"
if ((${#EXTRA_ARGS[@]})); then
  time python3 FastSimCluster.py \
    "${n_epoch}" "${OutputDir}" "${CONFIG_STAMP}" \
    "${EXTRA_ARGS[@]}"
else
  time python3 FastSimCluster.py \
    "${n_epoch}" "${OutputDir}" "${CONFIG_STAMP}"
fi
echo "[grid] Done."


