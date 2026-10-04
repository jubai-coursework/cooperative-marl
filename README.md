# 合作型多智能体强化学习：IPPO / MADDPG / MAPPO 对比

在 PettingZoo 的 MPE `simple_spread_v3` 环境上实现并对比三种合作型多智能体算法：IPPO、MADDPG、MAPPO。

实验设置与曲线解读要点见 [EXPERIMENT_ANALYSIS.md](EXPERIMENT_ANALYSIS.md)。

## 结构

| 路径 | 说明 |
|---|---|
| `envs/mpe_env.py` | MPE 环境封装 |
| `trainers/ippo_trainer.py` | IPPO 训练器 |
| `trainers/maddpg_trainer.py` | MADDPG 训练器 |
| `trainers/mappo_trainer.py` | MAPPO 训练器 |
| `models/ppo_nets.py` / `maddpg_nets.py` / `mappo_nets.py` | 三种算法的网络定义 |
| `memory/rollout_buffer.py` / `replay_buffer.py` | On-policy 轨迹缓冲 / Off-policy 回放池 |
| `config/settings.py` | 超参数配置 |
| `run_all.py` | 依次训练三种算法并生成对比图 |
| `play.py` | 加载模型回放：实时渲染、录制 GIF、算法对比 |
| `compare_runs.py` | 汇总各算法训练日志并绘图 |
| `runs/` | 训练产物：模型权重、奖励曲线、训练日志 |

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 依次训练三种算法（默认 2000 回合、5 个智能体）
# 可选位置参数：回合数、智能体数量
python run_all.py

# 回放已训练的模型
python play.py

# 汇总各次运行的对比曲线
python compare_runs.py
```

> `run_all.py` 读取环境变量 `MARL_PYTHON` 作为解释器路径，未设置时使用当前 Python。
> 仓库内的 `python_env.bat` 是针对本地 conda 环境的启动脚本，可参照其内容配置自己的环境。

## 结果

`runs/` 下已附带三种算法的训练产物（`model.pt`、`reward_curve.png`、`train_log.csv`），以及汇总对比图 `runs/comparison_marl_n5.png`。
