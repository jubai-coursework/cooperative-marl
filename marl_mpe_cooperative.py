import argparse
import os
import sys
import shutil

# Ensure project root is on path when run as script
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.settings import IPPOConfig, MADDPGConfig, MAPPOConfig
from envs.mpe_env import get_env_specs, make_env
from trainers.ippo_trainer import IPPOTrainer
from trainers.maddpg_trainer import MADDPGTrainer
from trainers.mappo_trainer import MAPPOTrainer
from utils.visualization import plot_training_curve
from utils.seed import resolve_device, set_seed

TRAINERS = {
    "ippo": (IPPOTrainer, IPPOConfig),
    "maddpg": (MADDPGTrainer, MADDPGConfig),
    "mappo": (MAPPOTrainer, MAPPOConfig),
}


def parse_args():
    if len(sys.argv) > 1:
        # 如果带参数运行，保持原有 argparse 逻辑
        parser = argparse.ArgumentParser(
            description="Cooperative MARL on PettingZoo MPE simple_spread_v3"
        )
        parser.add_argument("--algo", type=str, required=True, choices=["ippo", "maddpg", "mappo"])
        parser.add_argument("--n-agents", type=int, default=5)
        parser.add_argument("--episodes", type=int, default=500)
        parser.add_argument("--outdir", type=str, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default="auto")
        parser.add_argument("--plot-only", action="store_true", help="Only redraw curve from existing log")
        parser.add_argument("--plot-window", type=int, default=20)
        return parser.parse_args()

    # 否则使用交互式菜单
    print("=" * 50)
    print("      合作型多智能体强化学习交互式配置菜单      ")
    print("=" * 50)

    print("\n[1] 算法选择 (algo)")
    print("  1 - IPPO   (Independent PPO)")
    print("  2 - MADDPG (Multi-Agent DDPG)")
    print("  3 - MAPPO  (Multi-Agent PPO)")
    while True:
        algo_choice = input("请输入算法编号 [1/2/3] (回车默认: 3 -> MAPPO): ").strip()
        if not algo_choice:
            algo = "mappo"
            break
        elif algo_choice in ["1", "2", "3"]:
            algo = {"1": "ippo", "2": "maddpg", "3": "mappo"}[algo_choice]
            break
        print("输入无效，请重新输入 1, 2, 或 3。")

    n_agents = input("\n[2] 智能体数量 (n_agents) [回车默认: 5]: ").strip()
    n_agents = int(n_agents) if n_agents.isdigit() else 5

    episodes = input("\n[3] 训练回合数 (episodes) [回车默认: 500]: ").strip()
    episodes = int(episodes) if episodes.isdigit() else 500

    outdir = None

    seed = input("\n[4] 随机种子 (seed) [回车默认: 42]: ").strip()
    seed = int(seed) if seed.isdigit() else 42

    device = "auto"

    plot_only_in = input("\n[5] 是否仅绘制曲线 (plot-only) [y/N, 回车默认: N]: ").strip().lower()
    plot_only = True if plot_only_in in ['y', 'yes'] else False

    plot_window = 20

    print("\n" + "=" * 50)
    print("配置完成！准备运行...")
    print("=" * 50 + "\n")

    return argparse.Namespace(
        algo=algo,
        n_agents=n_agents,
        episodes=episodes,
        outdir=outdir,
        seed=seed,
        device=device,
        plot_only=plot_only,
        plot_window=plot_window
    )


def main():
    args = parse_args()
    outdir = args.outdir or f"runs/{args.algo}_n{args.n_agents}"

    if args.plot_only:
        csv_path = os.path.join(outdir, "train_log.csv")
        plot_path = os.path.join(outdir, "reward_curve.png")
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} does not exist. Cannot plot.")
            return

        plot_training_curve(
            csv_path,
            plot_path,
            window=args.plot_window,
            title=f"{args.algo.upper()} Training on simple_spread_v3",
        )
        print(f"Plot saved to {plot_path}")
        return

    # 默认每次训练前清除已有的训练数据、曲线和模型
    if os.path.exists(outdir):
        print(f"Cleaning output directory: {outdir}")
        shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    env = make_env(n_agents=args.n_agents, seed=args.seed)
    specs = get_env_specs(env)

    trainer_cls, config_cls = TRAINERS[args.algo]
    config = config_cls()
    trainer = trainer_cls(env, specs, outdir, device, config, seed=args.seed)

    print(f"Algorithm: {args.algo.upper()}")
    print(f"Agents: {args.n_agents} | Episodes: {args.episodes}")
    print(f"Obs dim: {specs.obs_dim} | Action dim: {specs.action_dim}")
    print(f"Device: {device}")
    print(f"Output: {outdir}")

    trainer.train(episodes=args.episodes, plot_window=args.plot_window)
    env.close()


if __name__ == "__main__":
    main()
