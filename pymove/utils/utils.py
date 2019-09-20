# TODO: Andreza e Arina
from __future__ import division
import time
import math
import folium
import datetime
import numpy as np
import pandas as pd

from IPython.display import display
from ipywidgets import IntProgress, HTML, VBox
from pandas._libs.tslibs.timestamps import Timestamp

"""main labels """
dic_labels = {"id" : 'id', 'lat' : 'lat', 'lon' : 'lon', 'datetime' : 'datetime'}

dic_features_label = {'tid' : 'tid', 'dist_to_prev' : 'dist_to_prev', "dist_to_next" : 'dist_to_next', 'dist_prev_to_next' : 'dist_prev_to_next', 
                    'time_to_prev' : 'time_to_prev', 'time_to_next' : 'time_to_next', 'speed_to_prev': 'speed_to_prev', 'speed_to_next': 'speed_to_next',
                    'period': 'period', 'day': 'day', 'index_grid_lat': 'index_grid_lat', 'index_grid_lon' : 'index_grid_lon',
                    'situation':'situation'}

#esses dois ultimos nao sei onde ficam
def log_progress(sequence, every=None, size=None, name='Items'):
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

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
        deltatime = time.time() - start_time
        deltatime_str_ = deltatime_str(deltatime)
        est_end = deltatime / curr_perc_new * 100
        est_time_str = deltatime_str(est_end - deltatime)
        print('({}/{}) {}% in {} - estimated end in {}'.format(size_processed, size_all, curr_perc_int_new, deltatime_str_, est_time_str))
        return curr_perc_int_new, deltatime_str # aqui era pra ser deltatime_str_ não?
    else:
        return curr_perc_int_new, None

