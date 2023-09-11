import numpy as np
import os, sys
import json




def get_job_node_list_slurm_rwth():
    host_list_val       = eval(os.environ['R_WLM_ABAQUSHOSTLIST'])

    host_list           = []
    for x in host_list_val:
        host_list.append(x[0])
    host_list = list(set(host_list))

    return host_list

def build_tf_config():
    # general settings
    port_range_start    = 23456
    tasks_per_node      = int(os.environ['SLURM_NTASKS_PER_NODE'])

    # create worker list
    list_hosts          = sorted(get_job_node_list_slurm_rwth())
    list_workers        = []
    for host in list_hosts:
        for i in range(tasks_per_node):
            list_workers.append(f"{host}:{port_range_start+i}")

    # create config and set environment variable
    tf_config = {
        'cluster': {
            'worker': list_workers
        },
        'task': {'type': 'worker', 'index': int(os.environ['RANK'])}
    }

    str_dump = json.dumps(tf_config)
    print(str_dump)

if __name__ == '__main__':
    # actual building the config
    build_tf_config()
