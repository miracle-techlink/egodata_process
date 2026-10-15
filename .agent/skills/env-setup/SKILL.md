---
name: env-setup
description: 通用技能：云端 GPU 服务器训练环境快速搭建与下载加速（镜像源/aria2c/uv/torch·flash-attn·deepspeed 预编译 wheels/HF·GitHub 加速/BBR）。源自团队《云端GPU服务器环境配置指南》(gpu-rsi-plugin)。
---

# 环境配置与加速（env-setup）

> 上游完整文档：https://github.com/miracle-techlink/gpu-rsi-plugin （2026-09 版）
> 本 skill 是其在项目内的可执行精华版。**大原则：一律预编译 wheel，不源码编译；大文件一律 aria2c，不 wget；缓存放数据盘。**

## 本服务器现状（2026-10-15 已落地）

- ✅ Miniconda `/opt/miniconda3`（清华 conda 源）、pip/uv **阿里云源**（`/etc/pip.conf` + `/root/.bashrc`；实测 41MB/s > 华为 20 > 清华 6）
- ✅ aria2c、uv 已全局安装；`HF_ENDPOINT`、`UV_INDEX_URL` 等已写入 `/root/.bashrc`
- ⚠️ BBR/sysctl 容器内无权限，跳过；Docker 加速不涉及（我们就在容器里）
- ⚠️ GitHub 直连 TLS 不稳 → clone/pull 用 `gh-proxy.com` 前缀（见下，已实测），**push 绝不走代理前缀（会泄露 token）**
- ⛔ **容器 cgroup 内存限 2GB：conda install 解依赖大的包（如 ffmpeg）会被 OOM kill 静默退出** → 装包一律 uv（Python 包）或 apt（系统工具），conda 只用来 create 空环境

## 1. 镜像源速查

```bash
# pip（已全局配置 /etc/pip.conf → 阿里云）
# uv（~/.bashrc 已配置）
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export UV_CONCURRENT_DOWNLOADS=100
export UV_HTTP_RETRIES=5
export UV_COMPILE_BYTECODE=1
export UV_CACHE_DIR=/root/gpufree-data/uv-cache   # 缓存放数据盘，二次装环境秒级
```

## 2. aria2c 多线程下载（模型/数据集/大 wheel 必备）

```bash
aria2c -x16 -s16 -k1M -c -d /path/to/dir -o filename <URL>
# -c 断点续传：中断后重跑同一条命令即可续（.aria2 控制文件别删）
# 服务器不支持 Range 时自动退化单线程，正常
```

## 3. HuggingFace / ModelScope / GitHub 加速

```bash
# HF（~/.bashrc 已配置 HF_ENDPOINT=https://hf-mirror.com，from_pretrained 零改动生效）
pip install "huggingface_hub[hf_xet]"        # Xet 分块去重+续传，新版 hub 自动启用
# hf_transfer 仅单次冷下载超大文件时临时开（与 Xet 互斥，勿常驻）：
# export HF_HUB_ENABLE_HF_TRANSFER=1

# 国产模型优先 ModelScope（国内 CDN）
modelscope download --model Qwen/Qwen2.5-7B-Instruct

# GitHub clone / release 下载（只读场景）
git clone https://gh-proxy.com/https://github.com/org/repo.git
aria2c -x16 -s16 -c https://gh-proxy.com/https://github.com/org/repo/releases/download/v1/x.whl
# 前缀可用性（2026-10-15 本服务器实测）：gh-proxy.com ✅ 可用；ghproxy.com 挂起、mirror.ghproxy.com DNS 解析失败
# ⛔ push 不走代理前缀——第三方代理会看到你的凭据
```

## 4. 训练底层 wheels（uv 安装，拒绝源码编译）

装包统一 `uv pip install`（在 conda env 内执行，比 pip 快一个量级），加 `--only-binary :all:` 杜绝意外触发源码编译。

```bash
# 4.1 torch（triton 随 torch 自动装匹配版，勿单独装）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 4.2 flash-attn：只装预编译 wheel。先查四元组：
python -c "import torch,sys; print(torch.__version__, torch.version.cuda, sys.version_info[:2], torch._C._GLIBCXX_USE_CXX11_ABI)"
# 按 torch×CUDA×python×cxx11abi 去 release 页选 whl（abi 选错 import 报符号错）：
# https://github.com/Dao-AILab/flash-attention/releases
aria2c -x16 -s16 -c https://ghproxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install ./flash_attn-*.whl

# 4.3 deepspeed（ZeRO/offload 无需编译 op，直接装）
uv pip install deepspeed && ds_report

# 4.4 mamba-ssm / causal-conv1d（caron 用）：同样优先预编译 wheel，见 mamba-training skill
```

版本速查（2026-09 时点，执行前以官方兼容矩阵复核）：torch 2.6 ↔ triton 3.2 / cu126；flash-attn 2.7.4。

## 5. 进阶手段（按需，细节见上游文档）

- **Git LFS 按需拉模型**：`GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 <repo>` 后 `git lfs pull --include="*.safetensors,config.json,tokenizer*"`
- **数据集**：探索用 `streaming=True`；单机训练前预下载 + `HF_DATASETS_OFFLINE=1`；缓存目录 `HF_HOME` 指数据盘
- **conda 解依赖慢**：conda 26 已默认 libmamba 求解器；新项目可用 pixi
- **万不得已源码编译**（自定义 CUDA op）：`apt install ccache`，`export CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache MAX_JOBS=$(nproc)`，二次编译小时级→分钟级
- **对象存储互传**：大量小文件 s5cmd；跨云 rclone；同区域走内网 endpoint

## 环境验收（装完必跑）

```bash
python - <<'EOF'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| gpu:", torch.cuda.is_available())
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", (x @ x).sum().item() != 0)
EOF
# 需要 flash_attn/deepspeed 时再 import 对应包验证
```
