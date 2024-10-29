import sys
import time
import utils

beg_time = time.localtime()
print(f"Starting timer at : {utils.get_time(beg_time)}")
cluster_output = "/storage/agrp/alonle/GAN_Output"
utils.check_run(sys.argv[1], path=cluster_output)
print(f"Done! Time elapsed : {utils.get_time(beg_time, time.localtime())}")
