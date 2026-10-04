from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.settings import MAPPOConfig
from envs.mpe_env import agent_dones, flatten_obs
from memory.rollout_buffer import RolloutBuffer
from models.mappo_nets import CentralizedCritic, GaussianActor
from trainers.base import BaseTrainer


class MAPPOTrainer(BaseTrainer):
    algo_name = "mappo"

    def __init__(self, env, specs, outdir: str, device, config: MAPPOConfig, seed: int = 42):
        super().__init__(env, specs, outdir, device, seed)
        self.config = config
        self.agent_ids = specs.agent_ids
        self.n_agents = specs.n_agents

        self.actors: List[GaussianActor] = []
        self.actor_optimizers: List[optim.Adam] = []
        self.buffers: List[RolloutBuffer] = []

        buf_size = config.max_episode_steps + 1
        for _ in range(self.n_agents):
            actor = GaussianActor(specs.obs_dim, specs.action_dim, config.hidden_dim).to(device)
            self.actors.append(actor)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=config.actor_lr))
            self.buffers.append(RolloutBuffer(buf_size, specs.obs_dim, specs.action_dim))

        self.critic = CentralizedCritic(specs.global_obs_dim, config.hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self._global_values: List[float] = []
        self._last_log_probs: Dict[str, float] = {}

    def _global_obs_tensor(self, obs_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        g = flatten_obs(obs_dict, self.agent_ids)
        return torch.FloatTensor(g).unsqueeze(0).to(self.device)

    def select_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        actions = {}
        global_t = self._global_obs_tensor(obs_dict)
        with torch.no_grad():
            global_value = self.critic(global_t).item()
        self._global_values.append(global_value)

        for i, aid in enumerate(self.agent_ids):
            state = obs_dict[aid].astype(np.float32)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mu, std = self.actors[i](state_t)
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
        return actions

    def store_transition(self, obs_dict, actions_dict, rewards_dict, next_obs_dict, term, trunc):
        dones = agent_dones(term, trunc)
        # Use centralized value from rollout (stored at select_actions time)
        value_idx = len(self.buffers[0])  # current step index before add
        global_value = self._global_values[value_idx] if value_idx < len(self._global_values) else 0.0

        for i, aid in enumerate(self.agent_ids):
            done = float(dones.get(aid, False))
            self.buffers[i].add(
                obs_dict[aid].astype(np.float32),
                actions_dict[aid].astype(np.float32),
                float(rewards_dict[aid]),
                done,
                self._last_log_probs[aid],
                global_value,
            )

    def _compute_global_returns(self, buf: RolloutBuffer, last_global_value: float):
        """GAE using centralized values stored in buffer.values."""
        cfg = self.config
        last_gae = 0.0
        for t in reversed(range(buf.ptr)):
            if t == buf.ptr - 1:
                next_value = last_global_value
                next_non_terminal = 1.0 - buf.dones[t]
            else:
                next_value = buf.values[t + 1]
                next_non_terminal = 1.0 - buf.dones[t]

            delta = buf.rewards[t] + cfg.gamma * next_value * next_non_terminal - buf.values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            buf.advantages[t] = last_gae
            buf.returns[t] = last_gae + buf.values[t]

        valid = buf.advantages[: buf.ptr]
        mean = np.mean(valid)
        std = np.std(valid)
        if std > 1e-8:
            buf.advantages[: buf.ptr] = (valid - mean) / (std + 1e-8)

    def on_episode_end(self):
        cfg = self.config
        self._global_values.clear()

        # Bootstrap last global value from final obs of last transition in buffer 0
        if len(self.buffers[0]) > 0:
            # Reconstruct last global obs from stored per-agent states at last step
            last_states = []
            for i in range(self.n_agents):
                last_states.append(self.buffers[i].states[len(self.buffers[i]) - 1])
            last_global = np.concatenate(last_states, axis=-1)
            last_global_t = torch.FloatTensor(last_global).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_global_value = self.critic(last_global_t).item()
        else:
            last_global_value = 0.0

        for i in range(self.n_agents):
            buf = self.buffers[i]
            if len(buf) == 0:
                continue
            self._compute_global_returns(buf, last_global_value)

        self._update_critic()

        for i in range(self.n_agents):
            buf = self.buffers[i]
            if len(buf) == 0:
                continue
            self._update_actor(i, buf)

        for buf in self.buffers:
            buf.clear()

    def _update_actor(self, agent_idx: int, buf: RolloutBuffer):
        cfg = self.config
        actor = self.actors[agent_idx]
        opt = self.actor_optimizers[agent_idx]

        states, actions, old_log_probs, _, advantages = buf.get()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)

        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, cfg.batch_size):
                end = min(start + cfg.batch_size, dataset_size)
                batch_idx = indices[start:end]

                log_probs, entropy = actor.evaluate(states[batch_idx], actions[batch_idx])
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = (
                    torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                    * advantages[batch_idx]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                loss = policy_loss - cfg.entropy_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                opt.step()

    def _update_critic(self):
        cfg = self.config
        all_global_obs = []
        all_returns = []

        for i in range(self.n_agents):
            buf = self.buffers[i]
            if len(buf) == 0:
                continue
            for t in range(buf.ptr):
                # Rebuild global obs from all agents at timestep t
                parts = [self.buffers[j].states[t] for j in range(self.n_agents)]
                all_global_obs.append(np.concatenate(parts, axis=-1))
                all_returns.append(buf.returns[t])

        if not all_global_obs:
            return

        global_obs_t = torch.FloatTensor(np.array(all_global_obs)).to(self.device)
        returns_t = torch.FloatTensor(all_returns).to(self.device)

        for _ in range(cfg.update_epochs):
            values = self.critic(global_obs_t)
            value_loss = nn.MSELoss()(values, returns_t)
            self.critic_optimizer.zero_grad()
            (cfg.value_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
            self.critic_optimizer.step()

    def save(self, path: str):
        torch.save(
            {
                "algo": "mappo",
                "actors": {aid: self.actors[i].state_dict() for i, aid in enumerate(self.agent_ids)},
                "critic": self.critic.state_dict(),
                "config": self.config,
            },
            path,
        )
