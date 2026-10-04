"""Evaluate and render trained MARL models — live view, GIF recording, or algorithm comparison."""
import argparse
import os
import sys
import time

import torch
from pettingzoo.mpe import simple_spread_v3

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.mpe_env import get_env_specs
from config.settings import IPPOConfig, MADDPGConfig, MAPPOConfig
from trainers.ippo_trainer import IPPOTrainer
from trainers.maddpg_trainer import MADDPGTrainer
from trainers.mappo_trainer import MAPPOTrainer
from utils.seed import resolve_device, set_seed
from utils.scene_renderer import (
    _frames_to_gif,
    load_trainer_for_render,
    record_episode,
    render_algorithm_comparison,
    create_comparison_grid,
)

TRAINERS = {
    "ippo": (IPPOTrainer, IPPOConfig),
    "maddpg": (MADDPGTrainer, MADDPGConfig),
    "mappo": (MAPPOTrainer, MAPPOConfig),
}


def load_trainer(model_path: str, env, specs, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    algo = checkpoint["algo"]
    config = checkpoint["config"]
    agents_state = checkpoint.get("agents") or checkpoint.get("actors")

    trainer_cls, _ = TRAINERS[algo]
    trainer = trainer_cls(env, specs, os.path.dirname(model_path), device, config)

    if algo == "ippo":
        for i, aid in enumerate(specs.agent_ids):
            trainer.networks[i].load_state_dict(agents_state[aid])
    elif algo in ("maddpg", "mappo"):
        for i, aid in enumerate(specs.agent_ids):
            trainer.actors[i].load_state_dict(agents_state[aid])

    if algo == "ippo":
        for net in trainer.networks:
            net.eval()
    elif algo in ("maddpg", "mappo"):
        for net in trainer.actors:
            net.eval()

    return trainer, algo


def _discover_models():
    """Find all model.pt files under runs/ and return {label: path}."""
    models = {}
    runs_dir = os.path.join(ROOT, "runs")
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir)):
            p = os.path.join(runs_dir, name, "model.pt")
            if os.path.exists(p):
                label = name.replace("_n", " (n=") + ")"
                models[label] = p
    return models


def _pick_model(models, prompt="选择模型"):
    """Let user pick one model from discovered list, or enter a custom path."""
    print(f"\n  已发现的模型:")
    items = list(models.items())
    for i, (label, path) in enumerate(items, 1):
        print(f"    {i} - {label}")
        print(f"        {path}")
    print(f"    {len(items) + 1} - 手动输入路径")
    while True:
        choice = input(f"  请选择 [{prompt}] [1-{len(items) + 1}]: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(items):
                return items[idx - 1][1]
            elif idx == len(items) + 1:
                return input("  请输入模型路径: ").strip()
        print("  输入无效，请重新输入。")


def _pick_models_multi(models):
    """Let user pick multiple models, or 'all'."""
    items = list(models.items())
    print(f"\n  已发现的模型:")
    for i, (label, path) in enumerate(items, 1):
        print(f"    {i} - {label}")
    print("    a - 全部选择")
    while True:
        choice = input("  请选择（多个用逗号分隔，如 1,2,3, 或 a）: ").strip().lower()
        if choice == "a":
            return [p for _, p in items]
        if choice:
            try:
                indices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
                if all(1 <= i <= len(items) for i in indices):
                    return [items[i - 1][1] for i in indices]
            except ValueError:
                pass
        print("  输入无效，请重新输入。")


def parse_args():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Evaluate and render a trained MARL model")
        parser.add_argument("--model-path", type=str, help="Path to a trained model.pt")
        parser.add_argument("--n-agents", type=int, default=5, help="Number of agents")
        parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--fps", type=int, default=15, help="Frames per second for rendering")
        parser.add_argument("--record", action="store_true")
        parser.add_argument("--output-dir", default="visualizations")
        parser.add_argument("--compare", nargs="+")
        parser.add_argument("--labels", nargs="+")
        parser.add_argument("--grid", action="store_true")
        return parser.parse_args()

    # ── 交互式菜单 ──────────────────────────────────────────────
    models = _discover_models()

    print("=" * 56)
    print("        MARL 模型评估与可视化 — 交互式菜单")
    print("=" * 56)

    print("\n[1] 渲染模式")
    print("  1 - 实时窗口播放 (human render)")
    print("  2 - 离线录制 GIF")
    print("  3 - 多算法对比 GIF (并排)")
    print("  4 - 多算法对比网格图 (静态 PNG)")
    while True:
        mode = input("  请选择渲染模式 [1-4] (回车默认: 2): ").strip()
        if not mode:
            mode = "2"; break
        if mode in ("1", "2", "3", "4"):
            break
        print("  输入无效，请重新输入 1-4。")

    record = mode in ("2",)
    compare = mode in ("3", "4")
    grid = mode == "4"

    model_path = None
    compare_paths = []
    labels = []

    if models:
        if compare:
            picked = _pick_models_multi(models)
            compare_paths = picked
            # 从目录名提取算法名
            for p in picked:
                parent = os.path.basename(os.path.dirname(p))
                algo = parent.split("_")[0]
                labels.append(algo)
        else:
            model_path = _pick_model(models, "选择一个模型")
    else:
        if compare:
            print("  未在 runs/ 下发现模型，请手动输入路径。")
            raw = input("  输入模型路径（多个用逗号分隔）: ").strip()
            compare_paths = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            model_path = input("  输入模型路径: ").strip()

    n_agents = input("\n[2] 智能体数量 [回车默认: 5]: ").strip()
    n_agents = int(n_agents) if n_agents.isdigit() else 5

    episodes = input("\n[3] 评估/录制回合数 [回车默认: 3]: ").strip()
    episodes = int(episodes) if episodes.isdigit() else 3

    fps = input("\n[4] 帧率 (FPS) [回车默认: 10]: ").strip()
    fps = int(fps) if fps.isdigit() else 10

    output_dir = input("\n[5] 输出目录 [回车默认: visualizations]: ").strip()
    output_dir = output_dir or "visualizations"

    device = "cpu"

    if not compare:
        labels_arg = None
    else:
        labels_arg = labels

    print("\n" + "=" * 56)
    print("  配置完成！正在运行...")
    print("=" * 56 + "\n")

    return argparse.Namespace(
        model_path=model_path,
        n_agents=n_agents,
        episodes=episodes,
        device=device,
        fps=fps,
        record=record,
        output_dir=output_dir,
        compare=compare_paths if compare else None,
        labels=labels_arg,
        grid=grid,
    )


def run_live_view(args):
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        sys.exit(1)

    set_seed(42)
    device = resolve_device(args.device)

    print(f"Loading env with {args.n_agents} agents and render_mode='human'...")
    env = simple_spread_v3.parallel_env(
        N=args.n_agents, local_ratio=0.5, max_cycles=25,
        continuous_actions=True, render_mode="human",
    )
    env.reset(seed=42)
    specs = get_env_specs(env)

    print(f"Loading model from {args.model_path}...")
    trainer, algo = load_trainer(args.model_path, env, specs, device)
    print(f"Algorithm detected: {algo.upper()}")

    delay = 1.0 / args.fps

    for ep in range(1, args.episodes + 1):
        obs_dict, _ = env.reset(seed=42 + ep)
        ep_reward = 0.0
        steps = 0

        print(f"\n--- Episode {ep} Start ---")
        while True:
            env.render()
            with torch.no_grad():
                actions_dict = trainer.select_actions(obs_dict)
            next_obs_dict, rewards_dict, term, trunc, _ = env.step(actions_dict)

            steps += 1
            step_reward = sum(rewards_dict.values())
            ep_reward += step_reward

            is_done = all(term.values()) or all(trunc.values())
            obs_dict = next_obs_dict
            time.sleep(delay)

            if is_done:
                break

        print(f"--- Episode {ep} End | Steps: {steps} | Total Reward: {ep_reward:.2f} ---")

    print("\nEvaluation finished.")
    env.close()


def run_record(args):
    trainer, specs, algo = load_trainer_for_render(
        args.model_path, n_agents=args.n_agents, device=args.device,
    )
    from pettingzoo.mpe import simple_spread_v3
    env = simple_spread_v3.parallel_env(
        N=args.n_agents, local_ratio=0.5, max_cycles=25,
        continuous_actions=True, render_mode="rgb_array",
    )

    print(f"Recording {args.episodes} episode(s) with {algo.upper()}...")
    for ep in range(args.episodes):
        out = os.path.join(args.output_dir, f"{algo}_ep{ep}.gif")
        frames = record_episode(trainer, env, specs, seed=42 + ep, fps=args.fps, save_path=out)
        print(f"  Saved {out}  ({len(frames)} frames)")

    env.close()


def run_compare(args):
    if not args.compare:
        print("Error: --compare requires at least one model path.")
        return

    labels = args.labels or [f"algo_{i}" for i in range(len(args.compare))]
    paths = dict(zip(labels, args.compare))

    if args.grid:
        out = create_comparison_grid(
            paths, n_agents=args.n_agents, fps=args.fps, save_dir=args.output_dir,
        )
        print(f"Comparison grid saved to {out}")
    else:
        outputs = render_algorithm_comparison(
            paths, n_agents=args.n_agents, episodes=args.episodes,
            fps=args.fps, save_dir=args.output_dir,
        )
        for k, v in outputs.items():
            print(f"  {k}: {v}")


def main():
    args = parse_args()

    if args.compare:
        run_compare(args)
    elif args.model_path and args.record:
        run_record(args)
    elif args.model_path:
        run_live_view(args)
    else:
        print("Error: specify --model-path, --compare, or --model-path --record.")
        sys.exit(1)


if __name__ == "__main__":
    main()
