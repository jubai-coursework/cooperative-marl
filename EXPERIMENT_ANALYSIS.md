# 实验结果对比分析（模板）

训练完成后，将 `runs/comparison_marl_n5.png` 插入报告，并参考以下要点撰写分析。

## 1. 算法差异

| 算法 | 策略 | 价值函数 | 数据效率 |
|------|------|----------|----------|
| IPPO | 各 agent 独立 π(a\|o_i) | 局部 V(o_i) | On-policy，样本利用率较低 |
| MADDPG | 确定性 μ(o_i) | 中心化 Q(o,a) | Off-policy + 回放，样本效率较高 |
| MAPPO | 各 agent 独立 π(a\|o_i) | 中心化 V(o_1..o_N) | On-policy，利用全局信息减方差 |

## 2. 曲线解读要点

- **IPPO**：各 agent 独立学习，环境非平稳性明显，曲线可能波动较大。
- **MADDPG**：Critic 可见全局信息，通常更稳定；需注意 warmup 期与探索噪声对前期曲线的影响。
- **MAPPO**：共享全局 baseline，协作信号更强，中后期往往优于 IPPO。

## 3. 调参建议（PDF 第七节）

若奖励波动大：增加 `--episodes` 至 800–1000，或降低 PPO/DDPG 学习率。

## 4. 复现命令

```bash
python marl_mpe_cooperative.py --algo ippo   --n-agents 5 --episodes 500 --outdir runs/ippo_n5
python marl_mpe_cooperative.py --algo maddpg --n-agents 5 --episodes 500 --outdir runs/maddpg_n5
python marl_mpe_cooperative.py --algo mappo  --n-agents 5 --episodes 500 --outdir runs/mappo_n5
python compare_runs.py --out runs/comparison_marl.png
```

或一键训练：

```bash
python run_all.py 500 5
```
