"""
Outputs all evaluation jobs that need to be run to a shell script.

Update ARGUMENTS with the sweep of parameters you want to run.

Usage:
    python -m relpose.eval.eval_driver eval_jobs.sh
"""
import argparse
import itertools

BASE_CMD = "python -m relpose.eval.eval_joint "
ARGUMENTS = {
    "checkpoint": ["data/pretrained_co3dv2"],
    "mode": ["mst", "sequential", "coord_asc"],
    "num_frames": [20, 10, 5, 3],
    "dataset": ["co3d"],
    "categories_type": ["unseen", "seen"],
    "index": [0, 1, 2, 3],
    "skip": [4],
}


def dict_product(dicts):
    """
    https://stackoverflow.com/a/40623158

    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def main(output_path):
    with open(output_path, "w") as f:
        for args in dict_product(ARGUMENTS):
            f.write(BASE_CMD + " ".join([f"--{k} {v}" for k, v in args.items()]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, default="eval_jobs.sh")
    args = parser.parse_args()
    main(args.output_path)
