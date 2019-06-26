from time import time
from utils import timeutils
#import timeutils

def progress_update(size_processed, size_all, start_time, curr_perc_int, step_perc=1):
    """
    update and print current progress.
    e.g.
    curr_perc_int, _ = pu.progress_update(size_processed, size_all, start_time, curr_perc_int)
    returns: curr_perc_int_new, deltatime_str
    """
    curr_perc_new = size_processed*100.0 / size_all
    curr_perc_int_new = int(curr_perc_new)
    if curr_perc_int_new != curr_perc_int and curr_perc_int_new % step_perc == 0:
        deltatime = time() - start_time
        deltatime_str = timeutils.deltatime_str(deltatime)
        est_end = deltatime / curr_perc_new * 100
        est_time_str = timeutils.deltatime_str(est_end - deltatime)
        print('({}/{}) {}% in {} - estimated end in {}'.format(size_processed, size_all, curr_perc_int_new, deltatime_str, est_time_str))
        return curr_perc_int_new, deltatime_str
    else:
        return curr_perc_int_new, None

