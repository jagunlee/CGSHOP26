import json, copy, time, random
import fast_data as fast
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # read initial solution
    input_file = args.data
    dt = fast.FastData(input_file)

