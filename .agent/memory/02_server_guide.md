# 02 — 服务器使用规范

## 连接

```bash
ssh -p 31211 root@183.147.142.40
# 密码见群消息（不要写进任何会提交到 git 的文件）
```

- 机型：RTX 4090 24GB（gpufree 容器实例）
- 系统 Python 3.10.12，conda 装在 `/opt/miniconda3`（已配好清华镜像，初始化进 `/root/.bashrc`）

## 磁盘布局（重要：根盘只有 30GB）

| 挂载点 | 容量 | 用途 |
|--------|------|------|
| `/`（overlay） | 30G | 系统、代码、个人目录 —— **不要放大文件** |
| `/root/gpufree-data` | 49G | **持久数据盘**：数据集 + 所有 conda 环境 |
| `/data/tini` | 1T（宿主机共享，73% 已用） | 勿依赖，非本实例专属 |

```
/root/gpufree-data/
├── conda/{envs,pkgs}     # 所有人的 conda 环境都装这里（.condarc 已配好）
└── datasets/             # 数据集统一放这里
```

**空间纪律**：原始视频、模型权重、checkpoint、conda 环境一律放数据盘；每人完工后清理中间产物；`df -h` 定期自查。根盘写满会导致全实例不可用。

## 工作目录

- 每人只在 `/home/<自己的目录>` 下做实验（目录分配见 `01_team.md`）
- 每个个人目录下有软链：`data` → 数据盘 datasets，`shared` → `/home/shared`
- 共享仓库 clone 在 `/home/shared/egodata_process`，正式代码一律通过 PR 进仓库，不要在共享 clone 里直接改

## conda 环境规范（避免编译冲突）

1. **严禁**往 base 环境装任何包
2. 每人创建自己的环境，命名 `<目录名>_<用途>`，如：
   ```bash
   conda create -n ash_hands python=3.10 -y
   conda activate ash_hands
   ```
3. 需要 CUDA 编译的包（mamba-ssm、nvdiffrast 等）装在自己环境里，出问题删环境重来，不影响别人
4. pip 已全局配置清华镜像（`/etc/pip.conf`）

## GPU

- 4090 24GB 一块，多人错峰使用；跑训练前 `nvidia-smi` 看一眼占用
- 2026-10-15 搭建时 GPU 未挂载进容器（`/dev/nvidia*` 不存在）。若 `nvidia-smi -L` 无输出，到 gpufree 控制台确认 GPU 是否分配，再重启实例
- 显存预算按 24GB 规划，见 `skills/mamba-training/SKILL.md`

## PR 流程

1. 自己目录实验跑通 → 整理代码进仓库对应 `src/` 模块
2. 分支命名 `<目录名>/<功能>`，如 `ash/hand-pose-extract`
3. PR 描述写清：输入输出是否符合 `03_data_pipeline.md` 契约、自测方式
4. 合并前更新 `.agent/memory/05_progress_log.md`
