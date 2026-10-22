# TASK ⑥ — Mamba 时序模型训练（负责人：Caron / 工位 `/home/caron@Caron`）

## 你的使命

用 Mamba 对连续交互关系做时序建模，训练出**意图预测模型**：公共数据集预训练 → 机电场景微调，并产出基线对照与 gaze 消融实验。

## 输入 → 输出

| | 内容 |
|---|---|
| 输入 | ④ 样本 + ⑤ `build_encoder/build_decoder` |
| 输出 | `data/ego_hoi/runs/<run>/best.pt` + 训练日志 + 指标报告 |

## 当前状态

- ✅ `conda activate caron_mamba`：torch 2.6.0+cu124、causal_conv1d 1.7.0、mamba_ssm 2.2.6.post3、transformers 4.56.2，版本已定格
- ✅ L40 GPU 可用；sm_89 本机 wheel 已备份，FP32/BF16 前向反向和 3 epoch 训练冒烟通过
- ✅ 参考代码：`refs/mamba`；⑤ 编码接口与 ④ 合成样本已联调
- ⏳ 下一步是真实样本上的 Mamba vs LSTM 对照，不再等 GPU 或编译环境

## 训练规范

- yaml config 驱动；bf16 混合精度；ckpt 含 config + 契约版本
- **显存预算（L40 46G）**：W=32/bs=64/d128 起步；OOM 顺序：grad checkpointing → batch 减半 → 梯度累积
- 训练前 `nvidia-smi` 看占用，错峰

## 实验矩阵（任务书 4/6 项）

1. 公共数据集预训练（当前以 EgoDemo 子集为主；ModelScope ego_pose_dataset 仍是空壳）
2. 机电场景微调（freeze encoder）
3. LSTM/GRU 同结构基线（验证 Mamba 收益）
4. ±gaze 消融（配合 arc 的预留接口）
5. 指标：三路 Top-1 acc、宏 F1、**可提前预测时长分布**（课题核心）

## 实施步骤

1. **D1**：环境自检（torch + mamba-ssm import）→ 随机数据冒烟 ✅
2. **D2**：与 arc 联调前向/反向；与 xw 对齐样本加载 ✅（合成数据）
3. **D3**：真实公共数据预训练出指标，并完成同数据 Mamba/LSTM 对照

## 验收标准

- [x] 冒烟：loss 下降 + 显存记录（3 epoch train loss 4.3740→1.4405；峰值 699 MiB）
- [ ] 基线 vs Mamba 对照表 + gaze 消融表
- [ ] 更新进度日志，提 PR（分支 `caron/mamba-train`）

## 求助与参考

- 手册：`.agent/skills/mamba-training/SKILL.md`（安装/预算/调优细节）
- https://github.com/state-spaces/mamba

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：Mamba 与 LSTM 均已在 L40 跑通；Mamba 的 FP32/BF16 前向反向、3 epoch 训练和 checkpoint 重载均通过。当前合成样本标签分布单一，准确率 1.0 只说明链路可运行，不能作为模型效果结论。

1. **真实对照实验**：④ 的真实 samples 产出后，使用相同 split、编码器、训练轮数与评价指标跑 `configs/mamba_base.yaml` 和 LSTM 基线，记录 loss/acc/宏 F1/显存
2. **真实数据预训练**：①②③④ 的真实 samples 先小规模过拟合，再扩大训练；暂以 EgoDemo 为主
3. **实验矩阵补齐**：宏 F1、提前预测时长分布等指标报告脚本还没写，是下一步（配合 ⑦）；之后再做 gaze 消融
4. **依赖不要升级**：torch 2.6.0、transformers 4.56.2 定格；装 mamba wheel 必须 `uv pip install --no-deps --reinstall`，详见 skill
