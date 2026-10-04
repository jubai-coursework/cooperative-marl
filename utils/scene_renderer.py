"""Scene renderer for MPE cooperative MARL — record episodes as GIF and compare algorithms."""
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.mpe_env import get_env_specs, make_env
from config.settings import IPPOConfig, MADDPGConfig, MAPPOConfig

TRAINER_REGISTRY = {
    "ippo": ("trainers.ippo_trainer", "IPPOTrainer", IPPOConfig),
    "maddpg": ("trainers.maddpg_trainer", "MADDPGTrainer", MADDPGConfig),
    "mappo": ("trainers.mappo_trainer", "MAPPOTrainer", MAPPOConfig),
}


def _import_trainer(algo: str):
    mod_path, cls_name, _ = TRAINER_REGISTRY[algo]
    mod = __import__(mod_path, fromlist=[cls_name])
    return getattr(mod, cls_name)


def load_trainer_for_render(model_path: str, n_agents: int = 5, device: str = "cpu"):
    """Load a saved model and return (trainer, specs, algo)."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    algo = checkpoint["algo"]
    config = checkpoint["config"]
    agents_state = checkpoint.get("agents") or checkpoint.get("actors")

    from pettingzoo.mpe import simple_spread_v3
    env = simple_spread_v3.parallel_env(
        N=n_agents, local_ratio=0.5, max_cycles=25,
        continuous_actions=True, render_mode="rgb_array",
    )
    env.reset()
    specs = get_env_specs(env)

    trainer_cls = _import_trainer(algo)
    trainer = trainer_cls(env, specs, os.path.dirname(model_path), device, config)

    if algo == "ippo":
        for i, aid in enumerate(specs.agent_ids):
            trainer.networks[i].load_state_dict(agents_state[aid])
            trainer.networks[i].eval()
    elif algo in ("maddpg", "mappo"):
        for i, aid in enumerate(specs.agent_ids):
            trainer.actors[i].load_state_dict(agents_state[aid])
            trainer.actors[i].eval()

    trainer.algo_name = algo
    return trainer, specs, algo


def _frames_to_gif(frames: List[np.ndarray], save_path: str, fps: int = 10):
    """Convert a list of (H,W,3) uint8 frames to an animated GIF."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    pil_images = [Image.fromarray(f) for f in frames]
    pil_images[0].save(
        save_path, save_all=True, append_images=pil_images[1:],
        duration=int(1000 / fps), loop=0,
    )


def record_episode(
    trainer,
    env,
    specs,
    seed: int = 42,
    fps: int = 10,
    save_path: Optional[str] = None,
) -> List[np.ndarray]:
    """Run one episode and return rendered frames.  If save_path is given, also write a GIF."""
    frames: List[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    done = False

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        with torch.no_grad():
            actions = trainer.select_actions(obs)
        obs, _, term, trunc, _ = env.step(actions)
        done = all(term.values()) or all(trunc.values())

    if save_path:
        _frames_to_gif(frames, save_path, fps=fps)

    return frames


def render_algorithm_comparison(
    model_paths: Dict[str, str],
    n_agents: int = 5,
    episodes: int = 1,
    fps: int = 10,
    save_dir: str = "visualizations",
) -> Dict[str, str]:
    """Record the same seed episode for each algorithm and produce a side-by-side GIF per episode.

    Returns a dict mapping algo name → comparison GIF path.
    """
    os.makedirs(save_dir, exist_ok=True)
    from pettingzoo.mpe import simple_spread_v3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainers = {}
    envs = {}

    for algo, path in model_paths.items():
        trainer, specs, _ = load_trainer_for_render(path, n_agents=n_agents, device=device)
        trainers[algo] = trainer
        envs[algo] = simple_spread_v3.parallel_env(
            N=n_agents, local_ratio=0.5, max_cycles=25,
            continuous_actions=True, render_mode="rgb_array",
        )

    outputs = {}

    for ep in range(episodes):
        seed = 42 + ep
        episode_frames: Dict[str, List[np.ndarray]] = {}

        for algo in model_paths:
            frames = record_episode(trainers[algo], envs[algo], specs, seed=seed, fps=fps)
            episode_frames[algo] = frames

            solo_path = os.path.join(save_dir, f"{algo}_ep{ep}.gif")
            _frames_to_gif(frames, solo_path, fps=fps)

        n_cols = len(model_paths)
        if n_cols > 0:
            min_len = min(len(f) for f in episode_frames.values())
            combined = []
            for t in range(min_len):
                row_frames = [episode_frames[algo][t] for algo in model_paths]
                combined.append(np.hstack(row_frames))

            cmp_path = os.path.join(save_dir, f"comparison_ep{ep}.gif")
            _frames_to_gif(combined, cmp_path, fps=fps)
            outputs[f"episode_{ep}"] = cmp_path

    for e in envs.values():
        e.close()

    return outputs


def create_comparison_grid(
    model_paths: Dict[str, str],
    n_agents: int = 5,
    seed: int = 42,
    fps: int = 10,
    save_dir: str = "visualizations",
) -> str:
    """Grid layout: one row per algorithm, showing key frames across a single episode."""
    os.makedirs(save_dir, exist_ok=True)
    from pettingzoo.mpe import simple_spread_v3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_frames: Dict[str, List[np.ndarray]] = {}
    envs_to_close = []

    for algo, path in model_paths.items():
        trainer, specs, _ = load_trainer_for_render(path, n_agents=n_agents, device=device)
        env_ = simple_spread_v3.parallel_env(
            N=n_agents, local_ratio=0.5, max_cycles=25,
            continuous_actions=True, render_mode="rgb_array",
        )
        envs_to_close.append(env_)
        frames = record_episode(trainer, env_, specs, seed=seed, fps=fps)
        all_frames[algo] = frames

    for e in envs_to_close:
        e.close()

    n_algos = len(all_frames)
    if n_algos == 0:
        return ""

    max_frames = max(len(f) for f in all_frames.values())
    key_indices = np.linspace(0, max_frames - 1, min(5, max_frames)).astype(int)

    rows = []
    for algo in model_paths:
        if algo not in all_frames:
            continue
        frames = all_frames[algo]
        row_parts = []
        for idx in key_indices:
            if idx < len(frames):
                row_parts.append(frames[idx])
            else:
                row_parts.append(np.zeros_like(frames[-1]))
        rows.append(np.hstack(row_parts))

    grid = np.vstack(rows) if rows else np.zeros((100, 100, 3), dtype=np.uint8)

    save_path = os.path.join(save_dir, "comparison_grid.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    Image.fromarray(grid).save(save_path)

    return save_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scene renderer for MARL MPE environments")
    parser.add_argument("--model-path", type=str, help="Path to a single model.pt for recording")
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--compare", nargs="+", help="Paths to multiple model.pt files for comparison")
    parser.add_argument("--algo-labels", nargs="+", help="Labels for comparison models")
    parser.add_argument("--output-dir", default="visualizations")
    parser.add_argument("--grid", action="store_true", help="Produce a static grid image instead of GIF")
    args = parser.parse_args()

    if args.compare:
        labels = args.algo_labels or [f"algo_{i}" for i in range(len(args.compare))]
        paths = dict(zip(labels, args.compare))
        if args.grid:
            out = create_comparison_grid(paths, n_agents=args.n_agents, save_dir=args.output_dir)
            print(f"Comparison grid saved to {out}")
        else:
            outputs = render_algorithm_comparison(
                paths, n_agents=args.n_agents, episodes=args.episodes,
                fps=args.fps, save_dir=args.output_dir,
            )
            for k, v in outputs.items():
                print(f"  {k}: {v}")
    elif args.model_path:
        from pettingzoo.mpe import simple_spread_v3

        trainer, specs, algo = load_trainer_for_render(
            args.model_path, n_agents=args.n_agents,
        )
        env = simple_spread_v3.parallel_env(
            N=args.n_agents, local_ratio=0.5, max_cycles=25,
            continuous_actions=True, render_mode="rgb_array",
        )
        for ep in range(args.episodes):
            out = os.path.join(args.output_dir, f"{algo}_ep{ep}.gif")
            record_episode(trainer, env, specs, seed=42 + ep, fps=args.fps, save_path=out)
            print(f"Saved {out}")
        env.close()
    else:
        parser.print_help()
