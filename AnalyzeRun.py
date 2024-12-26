import sys
import time
import utils
import argparse
#
# beg_time = time.localtime()
# print(f"Starting timer at : {utils.get_time(beg_time)}")
# cluster_output = "/storage/agrp/alonle/GAN_Output"
# utils.check_run(sys.argv[1], path=cluster_output,
#                 calculate_BED=True, save_df=False,
#                 plot_metrics=True, plot_results=True)
# print(f"Done! Time elapsed : {utils.get_time(beg_time, time.localtime())}")

parser = argparse.ArgumentParser(description="Run the GAN output processing script.")

# Positional argument for the main parameter
parser.add_argument('run_id', type=str, help="Run ID to process")

# Optional flags for boolean variables
parser.add_argument('--calculate_BED', action='store_true', help="Enable calculation of BED")
parser.add_argument('--save_df', action='store_true', help="Save DataFrame to output")
parser.add_argument('--plot_metrics', action='store_true', help="Plot metrics")
parser.add_argument('--plot_results', action='store_true', help="Plot results")

# Parse arguments
args = parser.parse_args()

# Start timing
beg_time = time.localtime()
print(f"Starting timer at : {utils.get_time(beg_time)}")

# Call utils.check_run with arguments
cluster_output = "/storage/agrp/alonle/GAN_Output"
utils.check_run(args.run_id, path=cluster_output,
                calculate_BED=args.calculate_BED,
                save_df=args.save_df,
                plot_metrics=args.plot_metrics,
                plot_results=args.plot_results)

# Print elapsed time
print(f"Done! Time elapsed : {utils.get_time(beg_time, time.localtime())}")
