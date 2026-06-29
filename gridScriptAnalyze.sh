#! /bin/bash
#PBS -m n
#PBS -l walltime=72:00:00 -l io=3

#### script that runs the python script, the MadGraph generator
echo "Installing python3>>>>>>"
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common
# Set default behavior
calculate_BED="--calculate_BED"
save_df=""
plot_metrics="--plot_metrics"
plot_results="--plot_results"

# Read settings from parname3
if [ "${parname3}" == "metrics" ]; then
    calculate_BED=""
    save_df=""
    plot_metrics="--plot_metrics"
    plot_results=""
fi
if [ "${parname3}" == "noBED" ]; then
    calculate_BED=""
    save_df=""
    plot_metrics="--plot_metrics"
    plot_results="--plot_results"
fi
if [ "${parname3}" == "all" ]; then
    calculate_BED=""
    save_df="--save_df"
    plot_metrics="--plot_metrics"
    plot_results="--plot_results"
fi
export MALLOC_ARENA_MAX=2
export PYTORCH_ALLOC_CONF=max_split_size_mb:256
run_id=${parname1}
Directory=${parname2}
cd ${Directory}
echo "I am now in ${PWD}, running analysis for run ${run_id}"

# Pass the flags to the Python script
time python3 AnalyzeRun.py ${run_id} ${calculate_BED} ${save_df} ${plot_metrics} ${plot_results}

