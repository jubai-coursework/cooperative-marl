from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from pettingzoo.mpe import simple_spread_v3


@dataclass
class EnvSpecs:
    agent_ids: List[str]
    n_agents: int
    obs_dim: int
    action_dim: int
    action_low: float
    action_high: float
    global_obs_dim: int
    global_action_dim: int
    max_cycles: int = 25


def make_env(n_agents: int = 5, seed: Optional[int] = None):
    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=True,
    )
    env.reset(seed=seed)
    return env


def get_env_specs(env) -> EnvSpecs:
    agent_ids = sorted(env.possible_agents)
    n_agents = len(agent_ids)
    obs_space = env.observation_space(agent_ids[0])
    act_space = env.action_space(agent_ids[0])
    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(act_space.shape))
    action_low = float(act_space.low[0])
    action_high = float(act_space.high[0])
    return EnvSpecs(
        agent_ids=agent_ids,
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        global_obs_dim=n_agents * obs_dim,
        global_action_dim=n_agents * action_dim,
    )


def flatten_obs(obs_dict: Dict[str, np.ndarray], agent_ids: List[str]) -> np.ndarray:
    return np.concatenate([obs_dict[aid].astype(np.float32) for aid in agent_ids], axis=-1)


def flatten_actions(action_dict: Dict[str, np.ndarray], agent_ids: List[str]) -> np.ndarray:
    return np.concatenate([action_dict[aid].astype(np.float32) for aid in agent_ids], axis=-1)


def stack_obs(obs_dict: Dict[str, np.ndarray], agent_ids: List[str]) -> np.ndarray:
    return np.stack([obs_dict[aid].astype(np.float32) for aid in agent_ids], axis=0)


def actions_from_stack(
    actions: np.ndarray,
    agent_ids: List[str],
    action_dim: int,
) -> Dict[str, np.ndarray]:
    return {
        aid: actions[i * action_dim : (i + 1) * action_dim]
        for i, aid in enumerate(agent_ids)
    }


def team_reward(rewards_dict: Dict[str, float]) -> float:
    if not rewards_dict:
        return 0.0
    return float(np.mean(list(rewards_dict.values())))


def episode_done(terminations: Dict[str, bool], truncations: Dict[str, bool]) -> bool:
    if not terminations:
        return True
    return all(terminations.values()) or all(truncations.values())


def agent_dones(terminations: Dict[str, bool], truncations: Dict[str, bool]) -> Dict[str, bool]:
    return {aid: terminations.get(aid, False) or truncations.get(aid, False) for aid in terminations}
