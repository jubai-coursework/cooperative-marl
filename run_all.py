"""Train IPPO, MADDPG, MAPPO sequentially and generate comparison plot."""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("MARL_PYTHON", sys.executable)


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_agents = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    algos = [
        ("ippo", f"runs/ippo_n{n_agents}"),
        ("maddpg", f"runs/maddpg_n{n_agents}"),
        ("mappo", f"runs/mappo_n{n_agents}"),
    ]

    for algo, outdir in algos:
        cmd = [
            PYTHON,
            os.path.join(ROOT, "marl_mpe_cooperative.py"),
            "--algo", algo,
            "--n-agents", str(n_agents),
            "--episodes", str(episodes),
            "--outdir", outdir,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

    compare_cmd = [
        PYTHON,
        os.path.join(ROOT, "compare_runs.py"),
        "--runs",
        f"runs/ippo_n{n_agents}",
        f"runs/maddpg_n{n_agents}",
        f"runs/mappo_n{n_agents}",
        "--out",
        f"runs/comparison_marl_n{n_agents}.png",
    ]
    print("Running:", " ".join(compare_cmd))
    subprocess.run(compare_cmd, cwd=ROOT, check=True)
    
    sys.path.insert(0, ROOT)
    from utils.visualization import visualize_runs
    print("Generating visualization report...")
    visualize_runs([
        f"runs/ippo_n{n_agents}",
        f"runs/maddpg_n{n_agents}",
        f"runs/mappo_n{n_agents}",
    ])
    print("All done.")


if __name__ == "__main__":
    main()