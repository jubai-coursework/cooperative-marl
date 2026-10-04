"""Unified visualization module for MARL training — standalone functions + TrainingVisualizer class."""
import os
from typing import Dict, List, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


# ═══════════════════════════════════════════════════════════════════
#  Standalone functions (compatible with original plotting.py API)
# ═══════════════════════════════════════════════════════════════════

def plot_training_curve(
    csv_path: str,
    save_path: str,
    window: int = 20,
    title: str = "Training Curve",
    reward_col: str = "team_reward",
):
    """Plot a single training reward curve and save to disk."""
    df = pd.read_csv(csv_path)
    rewards = df[reward_col].values
    episodes = df["episode"].values
    moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode Team Reward")
    plt.plot(episodes, moving_avg, label=f"Moving Avg (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Team Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_multi(
    run_dirs: Dict[str, str],
    save_path: str,
    window: int = 20,
    title: str = "IPPO vs MADDPG vs MAPPO",
):
    """Plot multiple training curves on the same axes and save to disk."""
    plt.figure(figsize=(12, 6))
    colors = {"ippo": "blue", "maddpg": "red", "mappo": "green"}

    for label, run_dir in run_dirs.items():
        csv_path = os.path.join(run_dir, "train_log.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        rewards = df["team_reward"].values
        episodes = df["episode"].values
        moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        color = colors.get(label, None)
        plt.plot(episodes, rewards, alpha=0.15, color=color)
        plt.plot(
            episodes,
            moving_avg,
            linewidth=2,
            color=color,
            label=f"{label.upper()} (window={window})",
        )

    plt.xlabel("Episode")
    plt.ylabel("Team Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  TrainingVisualizer class
# ═══════════════════════════════════════════════════════════════════

class TrainingVisualizer:
    """Higher-level visualization tool that saves into a managed directory."""

    COLORS = {
        "ippo": "#1f77b4",
        "maddpg": "#ff7f0e",
        "mappo": "#2ca02c",
        "qmix": "#d62728",
        "vdn": "#9467bd",
    }

    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── per-algorithm curve ──────────────────────────────────────

    def plot_training_curve(
        self,
        csv_path: str,
        title: str = "Training Curve",
        window: int = 20,
        show: bool = False,
        save_name: Optional[str] = None,
    ) -> str:
        """Plot a single training curve, delegating to the standalone function."""
        if save_name:
            save_path = str(self.save_dir / f"{save_name}.png")
        else:
            save_path = str(self.save_dir / f"{title.replace(' ', '_').lower()}.png")

        plot_training_curve(csv_path, save_path, window=window, title=title)

        if show:
            plt.imshow(plt.imread(save_path))
            plt.axis("off")
            plt.show()

        return save_path

    # ── multi-algorithm comparison ───────────────────────────────

    def plot_comparison(
        self,
        run_data: Dict[str, str],
        title: str = "Algorithm Comparison",
        window: int = 20,
        show: bool = False,
        save_name: str = "algorithm_comparison",
    ) -> str:
        """Compare multiple training runs.  Values can be CSV paths or run directories."""
        save_path = str(self.save_dir / f"{save_name}.png")

        fig, ax = plt.subplots(figsize=(14, 7))
        for algo_name, path in run_data.items():
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping {algo_name}")
                continue

            df = pd.read_csv(path)
            rewards = df["team_reward"].values
            episodes = df["episode"].values
            moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

            color = self.COLORS.get(algo_name.lower(), "#7f7f7f")
            ax.plot(episodes, rewards, alpha=0.15, color=color)
            ax.plot(episodes, moving_avg, linewidth=2.5, color=color, label=algo_name.upper())

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Team Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return save_path

    # ── statistics bars + boxplot ────────────────────────────────

    def plot_statistics(
        self,
        run_data: Dict[str, str],
        show: bool = False,
        save_name: str = "training_statistics",
    ) -> str:
        """Bar chart + boxplot comparing final-100-episode statistics."""
        stats = []
        all_rewards = []
        labels = []

        for algo_name, csv_path in run_data.items():
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            rewards = df["team_reward"].values[-100:]
            all_rewards.append(rewards)
            labels.append(algo_name.upper())
            stats.append({
                "algorithm": algo_name.upper(),
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "max": np.max(rewards),
                "min": np.min(rewards),
                "final": rewards[-1] if len(rewards) > 0 else 0,
            })

        stats_df = pd.DataFrame(stats)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].bar(stats_df["algorithm"], stats_df["mean"], yerr=stats_df["std"], capsize=5)
        axes[0].set_title("Mean Reward (Last 100 Episodes)", fontsize=12)
        axes[0].set_ylabel("Reward", fontsize=10)

        if all_rewards:
            axes[1].boxplot(all_rewards, labels=labels)
        axes[1].set_title("Reward Distribution (Last 100 Episodes)", fontsize=12)

        save_path = str(self.save_dir / f"{save_name}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        plt.close()

        return save_path

    # ── animated GIF ─────────────────────────────────────────────

    def generate_gif(
        self,
        csv_path: str,
        title: str = "Training Progress",
        window: int = 20,
        duration: float = 3.0,
        save_name: str = "training_progress",
    ) -> str:
        """Animated GIF showing reward curve building up over episodes."""
        df = pd.read_csv(csv_path)
        rewards = df["team_reward"].values
        episodes = df["episode"].values
        moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values

        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], linewidth=2, label="Moving Avg")
        scatter, = ax.plot([], [], alpha=0.3, label="Episode Reward")

        ax.set_xlim(0, episodes[-1])
        ax.set_ylim(min(rewards) - 10, max(rewards) + 10)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Team Reward")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        num_frames = 50
        frame_indices = np.linspace(0, len(episodes) - 1, num_frames).astype(int)

        def update(frame_idx):
            idx = frame_indices[frame_idx]
            line.set_data(episodes[:idx + 1], moving_avg[:idx + 1])
            scatter.set_data(episodes[:idx + 1], rewards[:idx + 1])
            return line, scatter

        anim = FuncAnimation(
            fig, update, frames=num_frames, interval=duration * 1000 / num_frames, blit=True
        )

        save_path = str(self.save_dir / f"{save_name}.gif")
        anim.save(save_path, writer=PillowWriter(fps=num_frames / duration))
        plt.close()

        return save_path

    # ── HTML report ──────────────────────────────────────────────

    def create_report(
        self,
        run_data: Dict[str, str],
        report_name: str = "training_report",
    ) -> str:
        """Create a self-contained HTML report with comparison and statistics."""
        report_path = str(self.save_dir / f"{report_name}.html")

        comparison_img = self.plot_comparison(run_data, show=False)
        stats_img = self.plot_statistics(run_data, show=False)

        rows = ""
        for algo_name, csv_path in run_data.items():
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                rewards = df["team_reward"].values[-100:]
                rows += f"""
                <tr>
                    <td>{algo_name.upper()}</td>
                    <td>{np.mean(rewards):.2f}</td>
                    <td>{np.std(rewards):.2f}</td>
                    <td>{np.max(rewards):.2f}</td>
                    <td>{np.min(rewards):.2f}</td>
                </tr>"""

        report_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #555; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }}
        .section {{ margin: 30px 0; }}
        img {{ max-width: 100%; border-radius: 8px; }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .stats-table th, .stats-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .stats-table th {{ background: #1f77b4; color: white; }}
        .stats-table tr:hover {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Agent RL Training Report</h1>
        <div class="section">
            <h2>Algorithm Comparison</h2>
            <img src="{os.path.basename(comparison_img)}" alt="Comparison Plot">
        </div>
        <div class="section">
            <h2>Training Statistics</h2>
            <img src="{os.path.basename(stats_img)}" alt="Statistics Plot">
        </div>
        <div class="section">
            <h2>Detailed Statistics</h2>
            <table class="stats-table">
                <tr><th>Algorithm</th><th>Mean Reward</th><th>Std</th><th>Max</th><th>Min</th></tr>
                {rows}
            </table>
        </div>
        <div class="section">
            <p style="text-align: center; color: #888; font-size: 14px;">
                Generated by MARL Visualization Module
            </p>
        </div>
    </div>
</body>
</html>
"""
        with open(report_path, "w") as f:
            f.write(report_content)

        return report_path


# ═══════════════════════════════════════════════════════════════════
#  Convenience entry point
# ═══════════════════════════════════════════════════════════════════

def visualize_runs(run_dirs: List[str], output_dir: str = "visualizations") -> None:
    """Discover CSVs under run directories and generate all visuals + report."""
    visualizer = TrainingVisualizer(output_dir)

    run_data = {}
    for run_dir in run_dirs:
        algo_name = os.path.basename(run_dir).split("_")[0]
        csv_path = os.path.join(run_dir, "train_log.csv")
        if os.path.exists(csv_path):
            run_data[algo_name] = csv_path

    if not run_data:
        print("No valid training logs found.")
        return

    print("Generating comparison plot...")
    visualizer.plot_comparison(run_data)

    print("Generating statistics plot...")
    visualizer.plot_statistics(run_data)

    print("Generating training report...")
    visualizer.create_report(run_data)

    print(f"Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MARL Training Visualization")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Paths to run directories")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory")
    args = parser.parse_args()

    visualize_runs(args.run_dirs, args.output_dir)
