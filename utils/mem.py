import re
import os
import psutil
import pwd
import pandas as pd
import json
import resource

def get_proc_info():
    UID = 1

    regex = re.compile(r'.+kernel-(.+)\.json')
    port_regex = re.compile(r'port=(\d+)')
    
    pids = [pid for pid in os.listdir('/proc') if pid.isdigit()]

    # memory info from psutil.Process
    df_mem = []

    for pid in pids:
        try:
            ret = open(os.path.join('/proc', pid, 'cmdline'), 'rb').read()
            ret_str = ret.decode('utf-8')
        except IOError:  # proc has already terminated
            continue

        # jupyter notebook processes
        if len(ret_str) > 0 and ('jupyter' in ret_str or 'ipython' in ret_str) and 'kernel' in ret_str:
            # kernel
            kernel_ID = re.sub(regex, r'\1', ret_str)[0:-1]
            #kernel_ID = filter(lambda x: x in string.printable, kernel_ID)

            # memory
            process = psutil.Process(int(pid))
            mem = process.memory_info()[0] / float(1e9)

            # user name for pid
            for ln in open('/proc/{0}/status'.format(int(pid))):
                if ln.startswith('Uid:'):
                    uid = int(ln.split()[UID])
                    uname = pwd.getpwuid(uid).pw_name

            # user, pid, memory, kernel_ID
            df_mem.append([uname, pid, mem, kernel_ID])

    df_mem = pd.DataFrame(df_mem)
    df_mem.columns = ['user', 'pid', 'memory_GB', 'kernel_ID']
    return df_mem

def get_session_info(sessions_str):
    sessions = json.loads(sessions_str)
    df_nb = []
    kernels = []
    for sess in sessions:
        kernel_ID = sess['kernel']['id']
        if kernel_ID not in kernels:
            notebook_path = sess['notebook']['path']
            df_nb.append([kernel_ID, notebook_path])
            kernels.append(kernel_ID)

    df_nb = pd.DataFrame(df_nb)
    df_nb.columns = ['kernel_ID', 'notebook_path']
    return df_nb

def stats(sessions_str):
    df_mem = get_proc_info()
    df_nb = get_session_info(sessions_str)

    # joining tables
    df = pd.merge(df_nb, df_mem, on=['kernel_ID'], how='right')
    df = df.sort_values('memory_GB', ascending=False)
    del(df_mem)
    del(df_nb)
    return df.reset_index(drop=True)

def mem():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return mem # used memory in MB
