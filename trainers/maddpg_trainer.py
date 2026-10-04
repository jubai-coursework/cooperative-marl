from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.settings import MADDPGConfig
from envs.mpe_env import episode_done, flatten_actions, flatten_obs, team_reward
from memory.replay_buffer import ReplayBuffer
from models.maddpg_nets import DDPGActor, MADDPGCritic
from trainers.base import BaseTrainer


def _soft_update(source: nn.Module, target: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class MADDPGTrainer(BaseTrainer):
    algo_name = "maddpg"

    def __init__(self, env, specs, outdir: str, device, config: MADDPGConfig, seed: int = 42):
        super().__init__(env, specs, outdir, device, seed)
        self.config = config
        self.agent_ids = specs.agent_ids
        self.n_agents = specs.n_agents
        self.total_steps = 0
        self.current_noise = config.noise_std

        self.actors: List[DDPGActor] = []
        self.actor_targets: List[DDPGActor] = []
        self.actor_optimizers: List[optim.Adam] = []

        for _ in range(self.n_agents):
            actor = DDPGActor(
                specs.obs_dim,
                specs.action_dim,
                config.hidden_dim,
                specs.action_low,
                specs.action_high,
            ).to(device)
            target = DDPGActor(
                specs.obs_dim,
                specs.action_dim,
                config.hidden_dim,
                specs.action_low,
                specs.action_high,
            ).to(device)
            target.load_state_dict(actor.state_dict())
            self.actors.append(actor)
            self.actor_targets.append(target)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=config.actor_lr))

        self.critic = MADDPGCritic(
            specs.global_obs_dim, specs.global_action_dim, config.hidden_dim
        ).to(device)
        self.critic_target = MADDPGCritic(
            specs.global_obs_dim, specs.global_action_dim, config.hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.buffer = ReplayBuffer(
            config.buffer_size, specs.global_obs_dim, specs.global_action_dim
        )

    def select_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        actions = {}
        for i, aid in enumerate(self.agent_ids):
            state = obs_dict[aid].astype(np.float32)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actors[i](state_t).squeeze(0).cpu().numpy()
            noise = np.random.normal(0, self.current_noise, size=action.shape)
            action = np.clip(action + noise, self.specs.action_low, self.specs.action_high)
            actions[aid] = action.astype(np.float32)
        return actions

    def store_transition(self, obs_dict, actions_dict, rewards_dict, next_obs_dict, term, trunc):
        global_obs = flatten_obs(obs_dict, self.agent_ids)
        global_next = flatten_obs(next_obs_dict, self.agent_ids)
        global_actions = flatten_actions(actions_dict, self.agent_ids)
        reward = team_reward(rewards_dict)
        done = float(episode_done(term, trunc))

        self.buffer.push(global_obs, global_actions, reward, global_next, done)
        self.total_steps += 1

        if len(self.buffer) >= self.config.warmup_steps:
            self._update()

    def on_episode_end(self):
        # Decay exploration noise
        if hasattr(self.config, 'noise_decay') and hasattr(self.config, 'noise_min'):
            self.current_noise = max(self.config.noise_min, self.current_noise * self.config.noise_decay)

    def _update(self):
        cfg = self.config
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_action_parts = []
            for i in range(self.n_agents):
                obs_i = next_states[:, i * self.specs.obs_dim : (i + 1) * self.specs.obs_dim]
                next_action_parts.append(self.actor_targets[i](obs_i))
            next_global_actions = torch.cat(next_action_parts, dim=-1)
            target_q = self.critic_target(next_states, next_global_actions).squeeze(-1)
            target_value = rewards + (1 - dones) * cfg.gamma * target_q

        current_q = self.critic(states, actions).squeeze(-1)
        critic_loss = nn.MSELoss()(current_q, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()

        if self.total_steps % cfg.update_actor_interval == 0:
            for i in range(self.n_agents):
                actor_action_parts = []
                for j in range(self.n_agents):
                    obs_j = states[:, j * self.specs.obs_dim : (j + 1) * self.specs.obs_dim]
                    if j == i:
                        actor_action_parts.append(self.actors[j](obs_j))
                    else:
                        start = j * self.specs.action_dim
                        end = (j + 1) * self.specs.action_dim
                        actor_action_parts.append(actions[:, start:end].detach())
                global_actor_actions = torch.cat(actor_action_parts, dim=-1)
                actor_loss = -self.critic(states, global_actor_actions).mean()

                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), cfg.max_grad_norm)
                self.actor_optimizers[i].step()

        for i in range(self.n_agents):
            _soft_update(self.actors[i], self.actor_targets[i], cfg.tau)
        _soft_update(self.critic, self.critic_target, cfg.tau)

    def save(self, path: str):
        torch.save(
            {
                "algo": "maddpg",
                "actors": {aid: self.actors[i].state_dict() for i, aid in enumerate(self.agent_ids)},
                "critic": self.critic.state_dict(),
                "config": self.config,
            },
            path,
        )
