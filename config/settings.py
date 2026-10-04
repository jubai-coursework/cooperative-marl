from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 2000
    n_agents: int = 5
    device: str = "auto"
    outdir: str = "runs/ippo_n5"
    plot_window: int = 20


@dataclass
class IPPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    max_episode_steps: int = 25


@dataclass
class MADDPGConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 100000
    batch_size: int = 1024
    noise_std: float = 0.2
    noise_min: float = 0.01
    noise_decay: float = 0.998
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    hidden_dim: int = 128
    update_actor_interval: int = 2


@dataclass
class MAPPOConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    max_episode_steps: int = 25
