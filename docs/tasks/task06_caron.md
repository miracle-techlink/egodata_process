# TASK ⑥ — Mamba 时序模型训练（负责人：Caron / 工位 `/home/caron@Caron`）

## 你的使命

用 Mamba 对连续交互关系做时序建模，训练出**意图预测模型**：公共数据集预训练 → 机电场景微调，并产出基线对照与 gaze 消融实验。

## 输入 → 输出

| | 内容 |
|---|---|
| 输入 | ④ 样本 + ⑤ `build_encoder/build_decoder` |
| 输出 | `data/ego_hoi/runs/<run>/best.pt` + 训练日志 + 指标报告 |

## 当前状态

- ✅ 环境预装中：`conda activate caron_mamba`（torch cu126 + causal-conv1d + mamba-ssm，编译状态见 `/home/caron@Caron/MIGRATION_NOTE.md` 后续或问 Daniel）
- ✅ 参考代码：`refs/mamba`（官方仓库）
- ⚠️ **GPU 当前未挂载**——训练前确认 `nvidia-smi` 有输出
- ⚠️ 你之前 `envs/pose_codec` 是 prefix 环境已失效，请删掉用 `caron_mamba`（规范见 `MIGRATION_NOTE.md`）
- ⏳ 等 ④⑤ 产出（等料期间：跑通 mamba-ssm import + 随机数据冒烟）

## 训练规范

- yaml config 驱动；bf16 混合精度；ckpt 含 config + 契约版本
- **显存预算（24G）**：W=32/bs=64/d128 起步；OOM 顺序：grad checkpointing → batch 减半 → 梯度累积
- 训练前 `nvidia-smi` 看占用，错峰

## 实验矩阵（任务书 4/6 项）

1. 公共数据集预训练（ego_pose_dataset，注意其当前是空壳——状态见 `.agent/memory/04_resources.md`）
2. 机电场景微调（freeze encoder）
3. LSTM/GRU 同结构基线（验证 Mamba 收益）
4. ±gaze 消融（配合 arc 的预留接口）
5. 指标：三路 Top-1 acc、宏 F1、**可提前预测时长分布**（课题核心）

## 实施步骤

1. **D1**：环境自检（torch + mamba-ssm import）→ 随机数据 10-step 冒烟
2. **D2**：与 arc 联调前向/反向；与 xw 对齐样本加载
3. **D3**：M3 里程碑——公共数据集预训练出指标

## 验收标准

- [ ] 冒烟：10-step loss 下降 + 显存记录
- [ ] 基线 vs Mamba 对照表 + gaze 消融表
- [ ] 更新进度日志，提 PR（分支 `caron/mamba-train`）

## 求助与参考

- 手册：`.agent/skills/mamba-training/SKILL.md`（安装/预算/调优细节）
- https://github.com/state-spaces/mamba

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：LSTM 基线已在 L40 GPU 跑通（loss 正常下降）；mamba_ssm 本机必须源码编译（glibc 2.31），FORCE_BUILD + sm_89 补丁 + --no-deps 的完整流程见 mamba-training skill，wheel 备份在 `/root/gpufree-data/wheels/ubuntu2004/`。

1. **Mamba 骨干真实训练**：编译验证后跑 `configs/mamba_base.yaml`，对比 LSTM 基线同数据同结构，记录 loss/acc/显存（L40 46G 比 4090 宽裕，batch 可以放大）
2. **真实数据预训练**：①②③④ 的真实 samples 出来后先做小规模预训练，再谈迁移机电场景；公共数据预训练数据集 ego_pose_dataset（ModelScope）目前仍是空壳，暂以 EgoDemo 为主
3. **实验矩阵补齐**：宏 F1、提前预测时长分布等指标报告脚本还没写，是下一步（配合 ⑦）
4. 注意：不要升级本机 torch（2.6.0 定格），装任何包先确认不会连带升级 torch（uv 加 --no-deps 或锁定）
