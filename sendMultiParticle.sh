#! /bin/bash

# Submit one training job per particle through the new run_particle.py pipeline
# (EMA + marginal calibration). Each call to sendParticleJobs.sh grabs the next
# free run id, so the particles land in separate run dirs.
#
# Usage:
#   ./sendMultiParticle.sh [epochs] [optional_data_root] [pdg ...]
# Examples:
#   ./sendMultiParticle.sh                      # 150 epochs, default data root, pdgs 22 11 2112
#   ./sendMultiParticle.sh 200                  # 200 epochs, default pdgs
#   ./sendMultiParticle.sh 150 "" 22 2112       # only photon + neutron
#   ./sendMultiParticle.sh 150 /storage/agrp/alonle/LUXE_FastSim/GAN_InputSample 11

nepoch="${1:-150}"
data_root_opt="${2:-}"
pdgs=("${@:3}")
if [[ ${#pdgs[@]} -eq 0 ]]; then
  pdgs=(22 11 2112)
fi

echo "Submitting ${#pdgs[@]} job(s): pdgs=${pdgs[*]} epochs=${nepoch} data_root='${data_root_opt}'"

for pdg in "${pdgs[@]}"; do
  echo "=== submitting pdg ${pdg} ==="
  if [[ -n "${data_root_opt}" ]]; then
    ./sendParticleJobs.sh "${pdg}" "${nepoch}" "${data_root_opt}"
  else
    ./sendParticleJobs.sh "${pdg}" "${nepoch}"
  fi
done
