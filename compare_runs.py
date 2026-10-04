import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.visualization import plot_comparison_multi


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MARL training curves")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["runs/ippo_n5", "runs/maddpg_n5", "runs/mappo_n5"],
    )
    parser.add_argument("--out", type=str, default="runs/comparison_marl.png")
    parser.add_argument("--window", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = ["ippo", "maddpg", "mappo"]
    run_dirs = {}
    for i, run_dir in enumerate(args.runs):
        label = labels[i] if i < len(labels) else f"run{i}"
        run_dirs[label] = run_dir

    plot_comparison_multi(run_dirs, args.out, window=args.window)
    print(f"Comparison plot saved to {args.out}")


if __name__ == "__main__":
    main()
