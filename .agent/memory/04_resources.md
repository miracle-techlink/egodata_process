# 04 — 资源汇总

## 项目资产

| 资源 | 地址 |
|------|------|
| 代码仓库 | https://github.com/miracle-techlink/egodata_process |
| 处理好的数据集 | https://www.modelscope.cn/datasets/Liuyue9698/ego_pose_dataset ⚠️ 2026-10-15 查：**仍是空壳**（只有 README），数据待上传 |
| 原始数据参考 | https://huggingface.co/datasets/LightwheelAI/EgoDemo ✅ 2026-10-22 **已获访问权**（SSH key + read token 已配在新服务器）。⚠️ 全量 1.44TB（四机位），只拉子集：`/root/gpufree-data/datasets/ego_hoi/raw/egodemo_subset/`；下载用 hf CLI + hf-mirror + token（直连 hf.co 极慢）。结构 `EgoProRaw/mp4/<场景>/<uuid>/<camera>/`，119 场景，单视频约 50MB；manifests/*.parquet 可能含标定信息（③ 待挖） |
| 云端环境配置指南 | https://github.com/miracle-techlink/gpu-rsi-plugin （精华版已收录：`skills/env-setup/SKILL.md`） |

## 服务器

- 现役：`ssh -p 30314 root@120.209.70.195`，**L40 46GB**（gpufree，Ubuntu 20.04）
- 旧 4090 机器因 GPU 无法挂载已弃用，不再作为开发环境
- 密码在群消息里，**不要**提交进仓库
- 规范详见 `02_server_guide.md`

## 关键开源项目 / 文献线索

| 用途 | 链接 |
|------|------|
| 手部姿态（入门首选） | [MediaPipe Hands](https://blog.csdn.net/g11d111/article/details/115539407) / [官方文档](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |
| 6D pose | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose)（需 RGB-D + mask + 物体 mesh） |
| 深度补齐（无深度数据时） | [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| 时序模型 | [state-spaces/mamba](https://github.com/state-spaces/mamba)（含 mamba-ssm、causal-conv1d） |
| 视线先验（增强探索） | 后续文献综述补充 |

## 镜像源与加速（服务器已配置）

- conda：清华镜像（`/opt/miniconda3/.condarc`）；pip/uv：阿里云镜像（`/etc/pip.conf` + 环境变量）
- uv、aria2c 已全局安装；`HF_ENDPOINT=https://hf-mirror.com` 等环境变量在 `/root/.bashrc`
- 装包用 `uv pip install`、大文件用 `aria2c -x16 -s16 -c`、GitHub clone 慢套 ghproxy 前缀（push 勿用）
- 完整手册：`skills/env-setup/SKILL.md`

## 文献综述（任务书第 1 项）

- 待补充：HOI 检测、意图预测、第一视角（egocentric）数据集（Ego4D / EPIC-KITCHENS）、SSM/Mamba 系列
- 文献笔记放仓库 `docs/literature/`（可后建），结论摘录进本文件
