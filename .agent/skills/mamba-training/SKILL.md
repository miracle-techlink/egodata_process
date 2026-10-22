---
name: mamba-training
description: 任务⑥ 基于 mamba-ssm 训练时序意图预测模型：公共数据集预训练 → 机电场景微调。负责人 Caron。
---

# Mamba 时序模型训练（mamba-training）

## 输入 / 输出

- 输入：④ 的 `samples/` + ⑤ 的 `build_encoder/build_decoder`
- 输出：`data/ego_hoi/runs/<run_name>/best.pt` + 训练日志（**checkpoint 存数据盘，根盘会爆**）

## 安装（2026-10-22：L40 / Ubuntu 20.04）

`conda activate caron_mamba`，已验证的兼容组合：

- **torch 2.6.0+cu124**（勿升级；重装 mamba 相关 wheel 时必须加 `--no-deps`）
- causal_conv1d 1.7.0 + mamba_ssm 2.2.6.post3：本机源码编译，包含 **sm_89**
- transformers 4.56.2 + tokenizers 0.22.2 + huggingface-hub 0.36.2（transformers 5.x 已删除 Mamba 所需的 `GreedySearchDecoderOnlyOutput`，不能直接使用）

本机 glibc 2.31 不兼容官方预编译 CUDA wheel，旧 wheel 已清理，不要再按旧文档安装。使用本机编译备份恢复：

```bash
conda activate caron_mamba
uv pip install "transformers==4.56.2" "tokenizers==0.22.2" "huggingface-hub==0.36.2"
uv pip install --no-deps --reinstall \
  /root/gpufree-data/wheels/ubuntu2004/causal_conv1d-1.7.0-cp310-cp310-linux_x86_64.whl \
  /root/gpufree-data/wheels/ubuntu2004/mamba_ssm-2.2.6.post3-cp310-cp310-linux_x86_64.whl
python -c "import torch; from mamba_ssm import Mamba; print(torch.__version__, torch.cuda.is_available())"
```

需要从源码重建时，严格按本文末尾的“Ubuntu 20.04 源码编译路线”执行；不要直接 `pip install mamba-ssm`。`daniel_test` 使用同一组兼容版本。

## 训练规范

- 配置驱动：`configs/<run_name>.yaml`（d_model、层数、lr、bs、W），命令行只传 config 路径
- 混合精度 bf16（L40 原生支持）；`torch.compile` 可选
- 日志：tensorboard 或 wandb（离线 `WANDB_MODE=offline`），存 `runs/<run_name>/`；未装 tensorboard 时现有代码回退到 `metrics.jsonl`
- checkpoint 含：model / optimizer / epoch / config / 契约版本（`03_data_pipeline.md` 顶部日期）

## 显存预算（经验起点）

| W(窗口) | batch | d_model | 备注 |
|--------|-------|---------|------|
| 32 | 64 | 128 | 初版安全起点 |
| 128 | 32 | 256 | OOM 先开 grad checkpointing，再降 batch |

OOM 顺序：grad checkpointing → batch 减半 → 梯度累积补回等效 batch。

## 训练策略（对应任务书 4–6）

1. **公共数据集预训练**：当前以已下载的 EgoDemo 机电类子集为主；ModelScope 的 ego_pose_dataset 仍是空壳，不能写成已有训练数据
2. **机电场景微调**：等真实 ①–④ 产出自建数据后 freeze encoder 微调
3. 基线对照：LSTM/GRU 同结构一路，验证 Mamba 收益（实验验证用）
4. 消融：± gaze 先验（配合 ⑤ 的预留接口）
5. 合成数据只验证代码链路；其标签分布单一，冒烟准确率不能作为模型效果结论

## 协作

- 接口问题找 ⑤arc；样本问题找 ④xw；冒烟失败先跑 `tests/check_contracts.py` 定位到具体环节

---

## Ubuntu 20.04 源码编译路线（2026-10-22 新服务器实测）

新服务器 L40 是 Ubuntu 20.04（glibc 2.31），官方预编译 wheel 要 glibc 2.32 → **import 报 `GLIBC_2.32 not found`**。预编译 wheel 在本机不可用，源码编译步骤（已在 caron_mamba/daniel_test 落地）：

```bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda MAX_JOBS=8
export MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE   # 否则 setup.py 会下载预编译包
cd <mamba源码> && python setup.py bdist_wheel
uv pip install --no-deps --reinstall dist/*.whl
```

四个坑（均已实测）：
1. `MAMBA_FORCE_BUILD=TRUE` 不设 → "Guessing wheel URL" 下载预编译包冒充编译产物，glibc 错误依旧
2. setup.py 硬编码 gencode 列表（75/80/87/90…）**没有 sm_89（L40）** → 要在 `arch=compute_80` 行后补 `arch=compute_89,code=sm_89`，否则 no kernel image
3. 重装 wheel 不加 `--no-deps` → torch 被升级 2.6→2.14，需降回
4. transformers 5.x → import 缺 `GreedySearchDecoderOnlyOutput`；按上方兼容组合定格 4.56.2

编译产物备份：`/root/gpufree-data/wheels/ubuntu2004/`（causal_conv1d 1.7.0 + mamba_ssm 2.2.6.post3，sm_89 专用，勿删）。临时源码构建目录已清理；需要重建时重新拉取对应源码并应用 sm_89 补丁。

## L40 验证记录（2026-10-22）

- causal_conv1d CUDA 前向/反向通过；Mamba FP32、BF16 前向/反向及梯度有限性检查通过
- `cuobjdump` 确认两个 CUDA 扩展均含 sm_89 cubin
- 3 epoch 合成数据训练冒烟：train loss 4.3740 → 1.4405，val loss 2.9758 → 0.9296，峰值显存 699 MiB
- `best.pt` / `last.pt` 已重载验证：`/home/daniel/pipeline_run/runs/mamba_smoke/`
