"""Test suite for utils.visualization — exercises every public API with real & synthetic data."""
import os
import sys
import tempfile
import csv
import shutil
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from utils.visualization import (
    plot_training_curve,
    plot_comparison_multi,
    TrainingVisualizer,
    visualize_runs,
)


def _make_synthetic_csv(path: str, episodes: int = 200, seed: int = 0):
    """Write a realistic-looking train_log.csv for testing."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5, episodes)
    trend = np.linspace(-60, -20, episodes)
    rewards = trend + noise
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "team_reward", "length"])
        w.writeheader()
        for ep in range(1, episodes + 1):
            w.writerow({"episode": ep, "team_reward": round(float(rewards[ep - 1]), 4), "length": 25})


def _find_real_csv():
    """Return the first real train_log.csv found under runs/, or None."""
    for root, _, files in os.walk(os.path.join(ROOT, "runs")):
        if "train_log.csv" in files:
            return os.path.join(root, "train_log.csv")
    return None


def test_standalone_plot_training_curve(tmpdir):
    csv_path = os.path.join(tmpdir, "train_log.csv")
    _make_synthetic_csv(csv_path)
    out = os.path.join(tmpdir, "curve.png")
    plot_training_curve(csv_path, out, window=10, title="Test Curve")
    assert os.path.exists(out), "plot_training_curve did not produce output"


def test_standalone_plot_comparison_multi(tmpdir):
    dirs = {}
    for algo in ("ippo", "maddpg", "mappo"):
        d = os.path.join(tmpdir, algo)
        _make_synthetic_csv(os.path.join(d, "train_log.csv"), seed=["ippo", "maddpg", "mappo"].index(algo))
        dirs[algo] = d
    out = os.path.join(tmpdir, "comparison.png")
    plot_comparison_multi(dirs, out, window=10)
    assert os.path.exists(out), "plot_comparison_multi did not produce output"


def test_visualizer_plot_training_curve(tmpdir):
    csv_path = os.path.join(tmpdir, "train_log.csv")
    _make_synthetic_csv(csv_path)
    viz = TrainingVisualizer(save_dir=tmpdir)
    path = viz.plot_training_curve(csv_path, title="Viz Curve", window=15, save_name="viz_curve")
    assert os.path.exists(path), f"TrainingVisualizer.plot_training_curve did not produce output: {path}"


def test_visualizer_plot_comparison(tmpdir):
    run_data = {}
    for algo in ("ippo", "maddpg"):
        p = os.path.join(tmpdir, f"{algo}.csv")
        _make_synthetic_csv(p, seed=["ippo", "maddpg"].index(algo))
        run_data[algo] = p
    viz = TrainingVisualizer(save_dir=tmpdir)
    path = viz.plot_comparison(run_data, save_name="viz_cmp")
    assert os.path.exists(path), f"TrainingVisualizer.plot_comparison did not produce output: {path}"


def test_visualizer_plot_statistics(tmpdir):
    run_data = {}
    for algo in ("ippo", "mappo"):
        p = os.path.join(tmpdir, f"{algo}.csv")
        _make_synthetic_csv(p, seed=["ippo", "mappo"].index(algo))
        run_data[algo] = p
    viz = TrainingVisualizer(save_dir=tmpdir)
    path = viz.plot_statistics(run_data, save_name="viz_stats")
    assert os.path.exists(path), f"TrainingVisualizer.plot_statistics did not produce output: {path}"


def test_visualizer_generate_gif(tmpdir):
    csv_path = os.path.join(tmpdir, "train_log.csv")
    _make_synthetic_csv(csv_path)
    viz = TrainingVisualizer(save_dir=tmpdir)
    path = viz.generate_gif(csv_path, save_name="viz_gif", duration=1.0)
    assert os.path.exists(path), f"TrainingVisualizer.generate_gif did not produce output: {path}"


def test_visualizer_create_report(tmpdir):
    run_data = {}
    for algo in ("ippo", "maddpg"):
        p = os.path.join(tmpdir, f"{algo}.csv")
        _make_synthetic_csv(p, seed=["ippo", "maddpg"].index(algo))
        run_data[algo] = p
    viz = TrainingVisualizer(save_dir=tmpdir)
    path = viz.create_report(run_data, report_name="viz_report")
    assert os.path.exists(path), f"TrainingVisualizer.create_report did not produce output: {path}"
    with open(path, "r") as f:
        html = f.read()
    assert "<html" in html, "Report does not look like valid HTML"


def test_visualize_runs(tmpdir):
    d1 = os.path.join(tmpdir, "test_ippo_n3")
    d2 = os.path.join(tmpdir, "test_mappo_n3")
    _make_synthetic_csv(os.path.join(d1, "train_log.csv"), seed=0)
    _make_synthetic_csv(os.path.join(d2, "train_log.csv"), seed=2)
    out_dir = os.path.join(tmpdir, "output")
    visualize_runs([d1, d2], output_dir=out_dir)
    assert os.path.exists(os.path.join(out_dir, "algorithm_comparison.png"))
    assert os.path.exists(os.path.join(out_dir, "training_statistics.png"))
    assert os.path.exists(os.path.join(out_dir, "training_report.html"))
    print("  visualize_runs produced all 3 expected files")


def test_with_real_data(tmpdir):
    real = _find_real_csv()
    if real is None:
        print("  (skipped — no real training logs found under runs/)")
        return "skipped"

    viz = TrainingVisualizer(save_dir=tmpdir)
    path = viz.plot_training_curve(real, title="Real Data Curve", save_name="real_curve")
    assert os.path.exists(path)
    print(f"  real-data plot saved to {path}")
    return "ok"


def main():
    tmpdir = os.path.join(ROOT, "test_visualizations")
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs(tmpdir, exist_ok=True)

    tests = [
        ("standalone plot_training_curve", test_standalone_plot_training_curve),
        ("standalone plot_comparison_multi", test_standalone_plot_comparison_multi),
        ("TrainingVisualizer.plot_training_curve", test_visualizer_plot_training_curve),
        ("TrainingVisualizer.plot_comparison", test_visualizer_plot_comparison),
        ("TrainingVisualizer.plot_statistics", test_visualizer_plot_statistics),
        ("TrainingVisualizer.generate_gif", test_visualizer_generate_gif),
        ("TrainingVisualizer.create_report", test_visualizer_create_report),
        ("visualize_runs", test_visualize_runs),
        ("with real training data", test_with_real_data),
    ]

    passed = 0
    failed = 0

    print("=" * 56)
    print("  utils.visualization — rendering test suite")
    print("=" * 56)
    print(f"  output dir: {tmpdir}\n")

    for name, fn in tests:
        try:
            result = fn(tmpdir)
            if result == "skipped":
                print(f"  [{name}] — skipped")
            else:
                print(f"  [{name}] PASS")
                passed += 1
        except Exception as e:
            print(f"  [{name}] FAIL — {e}")
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed, {len(tests) - passed - failed} skipped")
    print(f"  Artifacts saved to: {tmpdir}")
    print("=" * 56)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
