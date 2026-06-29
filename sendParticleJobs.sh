#! /bin/bash

# Submit a per-particle training job (new run_particle.py pipeline: EMA + calibration).
#
# Usage:
#   ./sendParticleJobs.sh <pdg> [epochs] [optional_data_root]
# Examples:
#   ./sendParticleJobs.sh 22 150
#   ./sendParticleJobs.sh 11 150 /storage/agrp/alonle/LUXE_FastSim/GAN_InputSample

pdg="${1:?Usage: ./sendParticleJobs.sh <pdg> [epochs] [data_root]}"
nepoch="${2:-150}"
data_root_opt="${3:-}"

runid=1
b=1

DESTINATION="/storage/agrp/alonle/LUXE_FastSim/ClusterLogs"
OUTPUT="/storage/agrp/alonle/LUXE_FastSim/GAN_Output"

# Find the next free run id
while [[ -d "${DESTINATION}/run_${runid}" ]]; do
  runid=$(( runid + b ))
done

echo "runid: ${runid}  pdg: ${pdg}  epochs: ${nepoch}"
mkdir -p "${DESTINATION}/run_${runid}/"
mkdir -p "${OUTPUT}/run_${runid}/"

PRESENTDIRECTORY="$(pwd)"

VARS="parname1=${pdg},parname2=${PRESENTDIRECTORY},parname3=${OUTPUT}/run_${runid}/,parname4=${nepoch}"
if [[ -n "${data_root_opt}" ]]; then
  VARS+=",parname5=${data_root_opt}"
fi

qsub \
  -l ngpus=1,mem=32gb \
  -v "${VARS}" \
  -q N \
  -N "run_${runid}_pdg${pdg}" \
  -o "${DESTINATION}/run_${runid}" \
  -e "${DESTINATION}/run_${runid}" \
  gridScriptParticle.sh

sleep 1s
