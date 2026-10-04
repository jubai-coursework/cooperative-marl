import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.settings import IPPOConfig
from envs.mpe_env import agent_dones, flatten_obs
from memory.rollout_buffer import RolloutBuffer
from models.ppo_nets import ActorCritic
from trainers.base import BaseTrainer


class IPPOTrainer(BaseTrainer):
    algo_name = "ippo"

    def __init__(self, env, specs, outdir: str, device, config: IPPOConfig, seed: int = 42):
        super().__init__(env, specs, outdir, device, seed)
        self.config = config
        self.agent_ids = specs.agent_ids
        self.n_agents = specs.n_agents

        self.networks: List[ActorCritic] = []
        self.optimizers: List[optim.Adam] = []
        self.buffers: List[RolloutBuffer] = []

        buf_size = config.max_episode_steps + 1
        for _ in range(self.n_agents):
            net = ActorCritic(specs.obs_dim, specs.action_dim, config.hidden_dim).to(device)
            self.networks.append(net)
            self.optimizers.append(optim.Adam(net.parameters(), lr=config.lr))
            self.buffers.append(RolloutBuffer(buf_size, specs.obs_dim, specs.action_dim))

        self._last_log_probs: Dict[str, float] = {}
        self._last_values: Dict[str, float] = {}

    def select_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        actions = {}
        for i, aid in enumerate(self.agent_ids):
            state = obs_dict[aid].astype(np.float32)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mu, std, value = self.networks[i](state_t)
            dist = torch.distributions.Normal(mu, std)
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).sum(dim=-1)
            action = np.clip(
                action_t.squeeze(0).cpu().numpy(),
                self.specs.action_low,
                self.specs.action_high,
            )
            actions[aid] = action.astype(np.float32)
            self._last_log_probs[aid] = log_prob.item()
            self._last_values[aid] = value.item()
        return actions

    def store_transition(
        self,
        obs_dict,
        actions_dict,
        rewards_dict,
        next_obs_dict,
        term,
        trunc,
    ):
        dones = agent_dones(term, trunc)
        for i, aid in enumerate(self.agent_ids):
            done = float(dones.get(aid, False))
            self.buffers[i].add(
                obs_dict[aid].astype(np.float32),
                actions_dict[aid].astype(np.float32),
                float(rewards_dict[aid]),
                done,
                self._last_log_probs[aid],
                self._last_values[aid],
            )

    def _update_agent(self, agent_idx: int):
        cfg = self.config
        net = self.networks[agent_idx]
        opt = self.optimizers[agent_idx]
        buf = self.buffers[agent_idx]

        if len(buf) == 0:
            return

        with torch.no_grad():
            last_state = torch.FloatTensor(buf.states[len(buf) - 1]).unsqueeze(0).to(self.device)
            _, _, last_value = net(last_state)
        buf.compute_returns_and_advantages(last_value.item(), cfg.gamma, cfg.gae_lambda)

        states, actions, old_log_probs, returns, advantages = buf.get()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, cfg.batch_size):
                end = min(start + cfg.batch_size, dataset_size)
                batch_idx = indices[start:end]

                log_probs, values, entropy = net.evaluate(
                    states[batch_idx], actions[batch_idx]
                )
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = (
                    torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                    * advantages[batch_idx]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, returns[batch_idx])
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy.mean()
                )

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

        buf.clear()

    def on_episode_end(self):
        for i in range(self.n_agents):
            self._update_agent(i)

    def save(self, path: str):
        torch.save(
            {
                "algo": "ippo",
                "agents": {aid: self.networks[i].state_dict() for i, aid in enumerate(self.agent_ids)},
                "config": self.config,
            },
            path,
        )
