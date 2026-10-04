import csv
import os
from typing import Dict, List


class EpisodeLogger:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.episodes: List[Dict] = []
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    def log(self, episode: int, team_reward: float, length: int = 0):
        self.episodes.append(
            {"episode": episode, "team_reward": team_reward, "length": length}
        )

    def save(self):
        with open(self.save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "team_reward", "length"])
            writer.writeheader()
            writer.writerows(self.episodes)
