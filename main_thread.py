# -*- coding:utf-8 -*-

import init_path
from dqsy.parameter import ParameterGained
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Based on Deep Learning Automatic Data Analysis '
                                                 'Platform for Oilfield Monitoring Video')
    parser.add_argument('--aas_dept_id', dest='aas_dept_id', help='aas_dept_id')
    parser.add_argument('--aas_obs_id', dest='aas_obs_id', help='aas_obs_id')
    parser.add_argument('--fct_id', dest='fct_id', help='fct_id')
    parser.add_argument('--stream', dest='stream', help='stream')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    aas_dept_id = args.aas_dept_id
    aas_obs_id = args.aas_obs_id
    fct_id = args.fct_id
    stream = args.stream
    parameter_gained = ParameterGained(fct_id, aas_obs_id, aas_dept_id, stream)
    parameter_gained.get_new_data()
