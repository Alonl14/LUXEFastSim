#! /bin/bash

### how many jobs you want to submit, if -1, then submits jobs you set
run_id=${1:-"1"}                # First argument: run_id (default to "1" if not provided)
setting=${2:-"default"}         # Second argument: "default" or "make_df" (default to "default")

### the place where the output and error file of the grid will live
DESTINATION="/storage/agrp/alonle/LUXE_FastSim/ClusterLogs"

echo "run_id: $run_id"
echo "setting: $setting"
OUTPUT="/storage/agrp/alonle/LUXE_FastSim/GAN_Output"
mkdir -p ${DESTINATION}/analysis_run_${run_id}/

#### from where you are submitting jobs
PRESENTDIRECTORY=${PWD}
export IOTHROTTLE_VERBOSE=1
#### submit jobs to the PBS system with run_id, PRESENTDIRECTORY, and setting
qsub -l ncpus=1,mem=1280gb -v parname1=${run_id},parname2=${PRESENTDIRECTORY},parname3=${setting} \
     -q N -N "analysis_run_${run_id}" \
     -o "${DESTINATION}/analysis_run_${run_id}" \
     -e "${DESTINATION}/analysis_run_${run_id}" \
     gridScriptAnalyze.sh

#### submit jobs to the PBS system
sleep 1s

    

