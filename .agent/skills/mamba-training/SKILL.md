---
name: mamba-training
description: 任务⑥ 基于 mamba-ssm 训练时序意图预测模型：公共数据集预训练 → 机电场景微调。负责人 Caron。
---

# Mamba 时序模型训练（mamba-training）

## 输入 / 输出

- 输入：④ 的 `samples/` + ⑤ 的 `build_encoder/build_decoder`
- 输出：`data/ego_hoi/runs/<run_name>/best.pt` + 训练日志（**checkpoint 存数据盘，根盘会爆**）

## 安装（✅ 2026-10-15 已预装验证，直接用）

`conda activate caron_mamba` 即可，已装好：

- **torch 2.6.0+cu126**（⚠️ 勿升到 2.14：mamba-ssm 没有配套预编译 wheel，源码编译与新版 torch 不兼容）
- causal_conv1d 1.5.2 + mamba_ssm 2.2.6.post3（GitHub releases 预编译 wheel，四元组 `cu12torch2.6 abiFALSE cp310`）

如需重建：

```bash
conda create -n caron_mamba python=3.10 -y && conda activate caron_mamba
uv pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126
# 预编译 wheel（服务器走 gh-proxy 前缀下载）：
uv pip install causal_conv1d-1.5.2+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
               mamba_ssm-2.2.6.post3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -c "from mamba_ssm import Mamba; print('ok')"
```

**踩过的坑**：`pip/uv 直装 mamba-ssm` 会触发源码编译，且 build isolation 默认把 torch 隔离掉导致失败；即便加 `--no-build-isolation`，对 torch 2.14 也编不过。**一律用预编译 wheel + 配套 torch 版本**。

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
