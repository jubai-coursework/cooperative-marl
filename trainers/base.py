import os
from abc import ABC, abstractmethod

from envs.mpe_env import episode_done, team_reward
from utils.logger import EpisodeLogger
from utils.visualization import plot_training_curve


class BaseTrainer(ABC):
    algo_name: str = "base"

    def __init__(self, env, specs, outdir: str, device, seed: int = 42):
        self.env = env
        self.specs = specs
        self.outdir = outdir
        self.device = device
        self.seed = seed
        os.makedirs(outdir, exist_ok=True)
        self.logger = EpisodeLogger(os.path.join(outdir, "train_log.csv"))

    @abstractmethod
    def select_actions(self, obs_dict):
        pass

    @abstractmethod
    def store_transition(self, obs_dict, actions_dict, rewards_dict, next_obs_dict, term, trunc):
        pass

    @abstractmethod
    def on_episode_end(self):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    def train(self, episodes: int, plot_window: int = 20):
        total_steps = 0
        for episode in range(1, episodes + 1):
            obs, _ = self.env.reset(seed=self.seed + episode)
            ep_team_reward = 0.0
            ep_length = 0
            done = False

            while not done:
                actions = self.select_actions(obs)
                next_obs, rewards, term, trunc, _ = self.env.step(actions)
                ep_team_reward += team_reward(rewards)
                ep_length += 1
                total_steps += 1

                self.store_transition(obs, actions, rewards, next_obs, term, trunc)
                obs = next_obs
                done = episode_done(term, trunc)

            self.on_episode_end()
            self.logger.log(episode, ep_team_reward, ep_length)

            if episode % 10 == 0 or episode == 1:
                print(
                    f"[{self.algo_name}] Episode {episode}/{episodes} | "
                    f"team_reward={ep_team_reward:.3f} | steps={total_steps}"
                )

        self.logger.save()
        model_path = os.path.join(self.outdir, "model.pt")
        self.save(model_path)

        csv_path = os.path.join(self.outdir, "train_log.csv")
        plot_path = os.path.join(self.outdir, "reward_curve.png")
        plot_training_curve(
            csv_path,
            plot_path,
            window=plot_window,
            title=f"{self.algo_name.upper()} Training on simple_spread_v3",
        )
        print(f"[{self.algo_name}] Saved log -> {csv_path}")
        print(f"[{self.algo_name}] Saved model -> {model_path}")
        print(f"[{self.algo_name}] Saved curve -> {plot_path}")
