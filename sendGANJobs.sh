#! /bin/bash

# Usage:
#   ./sendGANJobs.sh [epochs] [cfg_stamp] [optional_cfg_dir]
# Examples:
#   ./sendGANJobs.sh 50 cfg_cluster
#   ./sendGANJobs.sh 50 _cfg_v7.json /srv01/agrp/alonle/LUXEFastSim/Config

nepoch="${1:-1}"
cfg_stamp="${2:-cfg_cluster}"          # can be "cfg_cluster" or "_cfg_cluster.json"
cfg_dir_opt="${3:-}"                   # optional override of config directory

runid=1
b=1

DESTINATION="/storage/agrp/alonle/LUXE_FastSim/ClusterLogs"
OUTPUT="/storage/agrp/alonle/LUXE_FastSim/GAN_Output"

# Find the next free run id
while [[ -d "${DESTINATION}/run_${runid}" ]]; do
  runid=$(( runid + b ))
done

echo "runid: ${runid}"
if (( runid != 1 )); then
  echo "Run id taken, creating directories for run_${runid}"
fi

mkdir -p "${DESTINATION}/run_${runid}/"
mkdir -p "${OUTPUT}/run_${runid}/"

PRESENTDIRECTORY="$(pwd)"

# Build PBS -v variable list
VARS="parname1=${nepoch},parname2=${PRESENTDIRECTORY},parname3=${OUTPUT}/run_${runid}/,parname4=${cfg_stamp}"
if [[ -n "${cfg_dir_opt}" ]]; then
  VARS+=",parname5=${cfg_dir_opt}"
fi

# Submit
qsub \
  -l ngpus=1,mem=32gb \
  -v "${VARS}" \
  -q N \
  -N "run_${runid}" \
  -o "${DESTINATION}/run_${runid}" \
  -e "${DESTINATION}/run_${runid}" \
  gridScriptGAN.sh

sleep 1s

