import os
import time

print_loc = 'File'  # to print to stdout, change 'File' to anything else. For example 'std'


os.makedirs('log_traces', exist_ok=True)        
timestr = time.strftime("%Y%m%d-%H%M%S")
filename = timestr