#!/bin/bash

if [ -z "$1" ]; then
    echo "No folder provided. Please provide a folder as input."
    return 1
fi

export CONFIG_DIR="$1"

for config_file in "${CONFIG_DIR}"/inner_*.json; do
    # Check if the config file exists to avoid wildcard failures
    if [ -f "$config_file" ]; then
        # Extract the base file name without the path
        config_filename=$(basename "$config_file")

        echo "Submitting job for configuration: ${config_filename:5}"

        # Run the qsub command with the current config file
        source sendGANJobs.sh 150 ${config_filename:5}
    else
        echo "No configuration files found matching 'inner_*.json' in ${CONFIG_DIR}"
    fi
done
