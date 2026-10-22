# 02 — 服务器使用规范

## 连接（2026-10-22 起用新机器）

```bash
ssh -p 30314 root@120.209.70.195
# 密码见群消息（不要写进任何会提交到 git 的文件）
```

- 机型：**NVIDIA L40 46GB**（sm_89，驱动 580/CUDA 13.0，nvcc 12.1 在 `/usr/local/cuda`）
- 系统：**Ubuntu 20.04（glibc 2.31）**——⚠️ 很多官方预编译 CUDA wheel（mamba_ssm 等）要求 glibc 2.32+，本机必须源码编译，见 mamba-training skill
- 系统 Python 3.10，conda 装在 `/opt/miniconda3`（已配镜像）；uv/ffmpeg/aria2/git-lfs 已装
- 旧 4090 机器因 GPU 无法挂载已弃用，不要再用于环境配置、下载或训练

## 磁盘布局（重要：根盘只有 30GB）

| 挂载点 | 容量 | 用途 |
|--------|------|------|
| `/`（overlay） | 30G | 系统、代码、个人目录 —— **不要放大文件** |
| `/root/gpufree-data` | **492G** | **持久数据盘**：数据集 + 所有 conda 环境 |
| `/data/tini` | 1T（宿主机共享，73% 已用） | 勿依赖，非本实例专属 |

```
/root/gpufree-data/
├── conda/{envs,pkgs}     # 所有人的 conda 环境都装这里（.condarc 已配好）
└── datasets/             # 数据集统一放这里
```

**空间纪律**：原始视频、模型权重、checkpoint、conda 环境一律放数据盘；每人完工后清理中间产物；`df -h` 定期自查。根盘写满会导致全实例不可用。

## 工作目录

- 每人只在 `/home/<自己的目录>` 下放本人任务相关内容（目录格式 `<缩写>@<群昵称>`，分配见 `01_team.md`；总览表在 `/home/README.md`）
- **公用项目主代码只有一份中央 clone：`/workspace/egodata_process`**（origin 走 gh-proxy 只读拉取；push 由 Daniel 从 Mac 统一进行）
- 工位不放整仓 clone；保留 `README.md` + `TASK.md`（实体副本）+ `data` 软链，以及本人任务的实验/测试代码、参考仓库和产物
- 在工位编译、测试本人任务模块；验证后由 Daniel 整理进中央 `src/` 对应模块，不在服务器上 push

## conda 环境规范（避免编译冲突）

1. **严禁**往 base 环境装任何包
2. 每人创建自己的环境，命名 `<目录名>_<用途>`，如：
   ```bash
   conda create -n ash_hands python=3.10 -y
   conda activate ash_hands
   ```
3. 需要 CUDA 编译的包（mamba-ssm、nvdiffrast 等）装在自己环境里，出问题删环境重来，不影响别人
4. pip/uv 已全局配置阿里云镜像（实测最快，41MB/s）；**装包优先 `uv pip install`**（uv 已全局安装，快一个量级），如 `uv pip install --only-binary :all: numpy`

## 下载与加速工具（已全局就绪）

- `aria2c -x16 -s16 -c <URL>`：多线程 + 断点续传，模型/数据集/大 wheel 一律用它，别用 wget
- `HF_ENDPOINT=https://hf-mirror.com` 已在 `/root/.bashrc`，HF 下载零改动走镜像
- GitHub clone/pull：中央 clone `/workspace/egodata_process` 的 origin 已配成 `gh-proxy.com` 前缀（实测可用），`git pull` 直接可用（**只读拉取；push 一律从自己电脑做，勿走代理**）
- 完整加速手册（torch/flash-attn/deepspeed 预编译 wheel 选版方法等）：`.agent/skills/env-setup/SKILL.md`

## GPU

- **L40 46GB 一块**，多人错峰使用；跑训练前 `nvidia-smi` 看占用
- L40 计算能力为 sm_89；源码编译 CUDA 扩展时必须包含 `compute_89/sm_89`
- 显存起点与 OOM 处理见 `.agent/skills/mamba-training/SKILL.md`

## PR 流程

1. 自己目录实验跑通 → 整理代码进仓库对应 `src/` 模块
2. 分支命名 `<目录名>/<功能>`，如 `ash/hand-pose-extract`
3. PR 描述写清：输入输出是否符合 `03_data_pipeline.md` 契约、自测方式
4. 合并前更新 `.agent/memory/05_progress_log.md`
