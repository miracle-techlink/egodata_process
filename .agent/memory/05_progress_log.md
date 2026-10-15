# 05 — 进度日志

> 每人完工一个阶段就追加一条（最新在最上）。格式：
> `## YYYY-MM-DD 名字 — 做了什么｜产出路径｜遗留问题`

## 2026-10-15 Daniel — 七模块代码并行开发完成 + 契约首轮冻结

- 7 个 agent 并行完成各模块初版：① `src/preprocessing/split_clips.py`（fake-ffmpeg 编排级 e2e 过）② `src/hand_pose/extract_hands.py`（12 单测过）③ `src/pose6d/`（24 单测过，三段式可插拔：Detector/Depth/PoseBackend）④ `src/dataset/`（build_samples+label_rules+visualize+rot6d，合成数据 e2e 过）⑤ `src/encoding/`（10 单测过，CPU torch 实测）⑥ `src/mamba/train.py`（LSTM 基线 15 epoch 冒烟过，loss 正常下降）⑦ `tests/check_contracts.py` + `make_fixtures.py`（25 项自测全绿）
- **契约 2026-10-15 首轮冻结**：样本 label 分离键（label_subject/object_slot/object_id/interaction + name）、subject 0=left/1=right/2=none、interaction 7 类索引、obj_mask/obj_valid/hand_valid、dataset_index 顶层 {train,val,test}、pose6d bboxes 未检出填 -1、clips csv 固定 6 列+残段记 meta.json
- 新增 `docs/DELIVERABLES.md` 交付物对齐表（每阶段交付什么、谁验收、当前阻塞）
- 工位精简：删除每人工位冗余的 `shared` 软链与 `/home/shared`（工位只保留 clone + TASK.md + data 软链）
- 遗留：④ build_samples 有 4 处待修（恢复帧速度 NaN、index 结构、contract_version、窗口 off-by-one）；GPU 仍未挂载；EgoDemo 申请中

## 2026-10-15 Daniel — 全环境终验通过（无 GPU 条件下）

- daniel_test 补齐 torch 2.6.0 + causal_conv1d/mamba_ssm wheel（与 caron_mamba 同栈）
- 修复三处：xw_dataset 补 pyyaml；panda_6d 补系统库 libegl1 + libusb-1.0-0（open3d import 通过）；fish 的 ffmpeg 改走 **apt 系统级安装**（conda 装 ffmpeg 解依赖会被 2GB cgroup OOM kill——conda 大求解一律避免，优先 uv/apt）
- 终验结果：ash_hands(mediapipe 1.1.1) ✅ / xw_dataset ✅ / panda_6d(torch+trimesh+open3d) ✅ / arc_enc(torch 2.14.1+cu126) ✅ / caron_mamba+daniel_test(torch 2.6.0 + mamba wheels 在位) ✅ / ffmpeg 4.4.2 系统级 ✅
- ⚠️ 唯一阻塞：GPU 仍未挂载进容器（无 /dev/nvidia*），mamba import 验证与 ③⑥ 训练任务等 GPU 就位后进行；gpufree 控制台操作待定

## 2026-10-15 Daniel — 全环境预配完成

- 7 个 conda 环境建好（全落数据盘）：fish_split(ffmpeg→已改 apt 系统级)、ash_hands(mediapipe 1.1.1)、xw_dataset、arc_enc/panda_6d/daniel_test(torch)、caron_mamba
- 修复 mediapipe 缺 libGL（apt 装 libgl1/libglib2.0-0）
- mamba-ssm 路线定案：torch 2.14 无预编译 wheel 且源码编译不兼容 → caron_mamba/daniel_test 定格 **torch 2.6.0+cu126 + causal_conv1d 1.5.2 + mamba_ssm 2.2.6.post3 预编译 wheel**
- 参考仓库预拉：panda 工位 FoundationPose + Depth-Anything-V2；caron 工位 mamba
- 服务器自带 nvcc CUDA 12.6（编译能力有，但优先预编译 wheel 原则不变）

## 2026-10-15 Daniel — 工位弹药备齐 + 数据集情报

- 每个成员目录配好独立仓库 clone（`~/egodata_process`，origin 走 gh-proxy，全部同步到最新）
- 服务器系统 Python 装好 modelscope 1.40.2 + huggingface_hub[hf_xet] 下载工具
- 数据盘 `ego_hoi/README.md` 说明目录分工；`raw/download_egodemo.sh` 下载脚本就绪
- **重要情报**：① EgoDemo 是申请制（gated-manual），fish 需先去 HF 申请访问；② ModelScope 的 ego_pose_dataset 目前是空壳（数据未上传）——已在 04_resources.md 标注
- 遗留：fish 的 HF 申请待批；ego_pose_dataset 待上传方补数据

## 2026-10-15 Daniel — hdk 目录归属确认并迁移

- `hdk` = Caron（误用真名缩写建目录），其 repo clone + conda env 已迁移至 `/home/caron@Caron/`，附 MIGRATION_NOTE.md（prefix 环境重建提醒、`src/encoding/` 代码落点约定、remote 已改代理前缀）

## 2026-10-15 Daniel — 成员目录实名制

- `/home/` 成员目录改名 `<缩写>@<群昵称>`（fish@🐠、ash@Prick1yAsh、panda@空腹panda、xw@谢武、arc@Arc、caron@Caron），新增 `/home/README.md` 总览表
- GitHub 已打通（deploy key + ssh.github.com:443）：Mac 可 push；服务器仓库 origin 配 gh-proxy.com 前缀只读拉取
- 遗留：`hdk` 目录归属未登记；`daniel` 目录群昵称待确认

## 2026-10-15 Daniel — 收录环境配置指南 + 服务器加速落地

- 《云端GPU服务器环境配置指南》（[gpu-rsi-plugin](https://github.com/miracle-techlink/gpu-rsi-plugin)）精华版收录为 `skills/env-setup/SKILL.md`
- 服务器落地加速项：全局安装 aria2c + uv；`HF_ENDPOINT`/`UV_INDEX_URL`/`UV_CACHE_DIR`(数据盘) 写入 `/root/.bashrc`
- 定位网络问题：服务器直连 GitHub git 端口 TLS 不稳 → clone 用 ghproxy 前缀（push 不可走代理）
- 遗留：BBR/sysctl 容器内无权限跳过；GitHub push 仍需凭据

## 2026-10-15 Daniel — 项目初始化

- 服务器环境：安装 Miniconda（`/opt/miniconda3`，清华镜像），conda 环境与数据集规划到数据盘 `/root/gpufree-data`
- 建立 `/home/` 下 7 个成员目录（fish / ash / panda / xw / arc / caron / daniel）+ `/home/shared`
- 仓库初始化：README、AGENTS.md、`.agent/`（memory×6 + skills×7）、`src/` 模块骨架
- **遗留问题**：
  1. ⚠️ 服务器 GPU 当前未挂载进容器（`nvidia-smi -L` 无输出）——训练前需在 gpufree 控制台确认
  2. 数据集尚未下载到 `/root/gpufree-data/datasets/`
  3. 《训练环境配置优化》文档链接待补充进 `04_resources.md`
