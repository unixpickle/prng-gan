import argparse
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=str, default="gzip_u8_4096")
    parser.add_argument("--smoothing", type=float, default=0.99)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("paths", nargs="+", type=str)
    args = parser.parse_args()

    plt.figure()
    for path in args.paths:
        xs, ys = read_plot(path, args.field)
        plt.plot(xs, smooth_ema(ys, args.smoothing), label=os.path.basename(path))
    plt.legend()
    plt.xlabel("step")
    plt.ylabel(args.field)
    plt.savefig(args.output)


def read_plot(path: str, field: str) -> Tuple[List[int], List[int]]:
    expr = re.compile(f".*step ([0-9]*):.* {field}=([0-9-.]+)( .*)?")
    results = {}
    with open(path, "r") as f:
        for row in f:
            row = row.rstrip("\n")
            match = expr.match(row)
            if match:
                results[int(match[1])] = float(match[2])

    xs = sorted(results.keys())
    ys = [results[x] for x in xs]

    return xs, ys


def smooth_ema(ys: List[float], rate: float) -> List[float]:
    if not len(ys):
        return ys
    y = ys[0]
    result = []
    for val in ys:
        y = y * rate + val * (1 - rate)
        result.append(y)
    return result


if __name__ == "__main__":
    main()
