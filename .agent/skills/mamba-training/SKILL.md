---
name: mamba-training
description: 任务⑥ 基于 mamba-ssm 训练时序意图预测模型：公共数据集预训练 → 机电场景微调。负责人 Caron。
---

# Mamba 时序模型训练（mamba-training）

## 输入 / 输出

- 输入：④ 的 `samples/` + ⑤ 的 `build_encoder/build_decoder`
- 输出：`data/ego_hoi/runs/<run_name>/best.pt` + 训练日志（**checkpoint 存数据盘，根盘会爆**）

## 安装（4090 + cu121，独立环境）

```bash
conda create -n caron_mamba python=3.10 -y && conda activate caron_mamba
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install causal-conv1d>=1.4.0 mamba-ssm   # 编译需 nvcc；卡住就下预编译 wheel
```

- 验证：`python -c "import mamba_ssm; from mamba_ssm import Mamba"`（import 报错九成是 torch/cuda 版本不匹配）
- 官方仓库：https://github.com/state-spaces/mamba

## 训练规范

- 配置驱动：`configs/<run_name>.yaml`（d_model、层数、lr、bs、W），命令行只传 config 路径
- 混合精度 bf16（4090 原生支持）；`torch.compile` 可选
- 日志：tensorboard 或 wandb（离线 `WANDB_MODE=offline`），存 `runs/<run_name>/`
- checkpoint 含：model / optimizer / epoch / config / 契约版本（`03_data_pipeline.md` 顶部日期）

## 24GB 显存预算（经验起点）

| W(窗口) | batch | d_model | 备注 |
|--------|-------|---------|------|
| 32 | 64 | 128 | 初版安全起点 |
| 128 | 32 | 256 | OOM 先开 grad checkpointing，再降 batch |

OOM 顺序：grad checkpointing → batch 减半 → 梯度累积补回等效 batch。

## 训练策略（对应任务书 4–6）

1. **公共数据集预训练**：ego_pose_dataset（ModelScope，已处理好），量大先子集跑通
2. **机电场景微调**：等 ①–④ 产出自建数据后 freeze encoder 微调
3. 基线对照：LSTM/GRU 同结构一路，验证 Mamba 收益（实验验证用）
4. 消融：± gaze 先验（配合 ⑤ 的预留接口）

## 协作

- 接口问题找 ⑤arc；样本问题找 ④xw；冒烟失败先跑 `tests/check_contracts.py` 定位到具体环节
