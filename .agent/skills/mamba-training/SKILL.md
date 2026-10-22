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
# torch 走清华 pypi（⚠️ download.pytorch.org 从本服务器下载会停滞，勿用官方 cu126 index）
uv pip install torch==2.6.0 torchvision==0.21.0
# 预编译 wheel 数据盘有备份，直接装：
uv pip install /root/gpufree-data/wheels/causal_conv1d-1.5.2+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
               /root/gpufree-data/wheels/mamba_ssm-2.2.6.post3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -c "from mamba_ssm import Mamba; print('ok')"
```

**踩过的坑**：① `uv/pip 直装 mamba-ssm` 会源码编译且 build isolation 隔离 torch → 失败；② torch 2.14 无配套 wheel 且编译不兼容；③ download.pytorch.org 从本服务器拉不动 → torch 用清华 pypi（自带 cu124 runtime，与 cu12 wheel 兼容，`torch.__version__` 显示 2.6.0+cu124 属正常）。wheel 备份在 `/root/gpufree-data/wheels/`（含 sha 命名，勿删）。

**GPU 未挂载时的预期现象**：容器没挂 GPU（无 `/dev/nvidia*`）时，`from mamba_ssm import Mamba` 会报 `RuntimeError: 0 active drivers`（triton autotuner 初始化需要 CUDA driver）——**这不是安装错误**。此时用 `uv pip show mamba-ssm` + 确认 `site-packages/mamba_ssm/*.so` 存在 + `import torch` 成功作为安装到位的判据；GPU 挂载后再跑完整 import。daniel_test 环境也装了同一套（torch 2.6 + 两个 wheel）。

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

---

## Ubuntu 20.04 源码编译路线（2026-10-22 新服务器实测，必走）

新服务器 L40 是 Ubuntu 20.04（glibc 2.31），官方预编译 wheel 要 glibc 2.32 → **import 报 `GLIBC_2.32 not found`**。预编译 wheel 在本机不可用，源码编译步骤（已在 caron_mamba/daniel_test 落地）：

```bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS=8
export MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE   # 关键！否则 setup.py 会偷下载预编译包
# torch 保持 2.6.0，不要升级
cd <mamba源码> && python setup.py bdist_wheel
# 安装时必须 --no-deps，否则 uv 会顺手把 torch 升到最新版破坏兼容：
uv pip install --no-deps --reinstall dist/*.whl
```

三个坑（都踩过）：
1. `MAMBA_FORCE_BUILD=TRUE` 不设 → "Guessing wheel URL" 下载预编译包冒充编译产物，glibc 错误依旧
2. setup.py 硬编码 gencode 列表（75/80/87/90…）**没有 sm_89（L40）** → 要在 `arch=compute_80` 行后补 `arch=compute_89,code=sm_89`，否则 no kernel image
3. 重装 wheel 不加 `--no-deps` → torch 被升级 2.6→2.14，需降回

编译产物备份：`/root/gpufree-data/wheels/ubuntu2004/`（causal_conv1d 1.7.0 + mamba_ssm 2.2.6.post3，sm_89 专用）。
