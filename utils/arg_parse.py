import argparse
import yaml


def arg_paser():
    # parse the config for console arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--params", default=None, help='Base set of params to use for experiment')

    parser.add_argument("--save_path", default=None, help='path to save all results')

    # update able params
    parser.add_argument("--update_params", default={}, type=yaml.load, help='Update the parameters dictionary.')

    return parser.parse_args()