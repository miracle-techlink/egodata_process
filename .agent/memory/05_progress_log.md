# 05 — 进度日志

> 每人完工一个阶段就追加一条（最新在最上）。格式：
> `## YYYY-MM-DD 名字 — 做了什么｜产出路径｜遗留问题`

## 2026-10-22 Codex — 锁定120个合格联合采集目标与持续计数口径

- 阶段目标120个通过验收的真人手部＋物体6D独立采集，候选目标约84/18/18；优先被试/采集整组隔离，不用Ego辅助、机位、滑窗、重复或插值凑数。旧冻结不改写。
- 当前已下载联合候选24（16/4/4）、正式合格0（0/0/0），距120合格缺口120；辅助24另列。目标状态在 `runs/human_dataset_v1_20261022/target_120_progress.json`。
- 交叉核查固定HO-Cap镜像目录与官方split：64个采集档案中train50/valid7/test7。当前只下载官方train策略下最多另补26候选，即使50全合格仍至少需其他来源70个合格采集；不把镜像目录数当整个官方数据规模。逐采集证据 `source_capacity_120.json`。
- 此次只锁定目标、审计来源容量，没有启动下载/训练；需推进既有24条时间/几何/参考质量验收与新来源验证，数量质量都达标再训练。
- GitHub同步限定为本阶段三份文档；只提交这两条数据阶段日志、交付表的数据阶段条目及数据规范文档，不夹带历史未提交阶段、代码、数据、权重或凭据。1,081tests＋84subtests对应本地/服务器工作区，不表示本次文档commit包含新增工具和测试源码。
- GitHub同步未完成：本地main已生成仅文档提交；HTTPS没有可用登录凭据，现有SSH认证成功但被目标仓库拒绝写入权限。未force-push、未通过只读代理推送；须取得 `miracle-techlink/egodata_process` 写权限后再同步，不能将本地提交当作已推送。

## 2026-10-22 Codex 多Agent — 数据先行，扩充48个真人采集并冻结候选划分

- 按用户要求停止新训练，三路并行做HO-Cap采集、EgoDemo长序列和发布治理；主agent独立审图、全量结构检查与整合，不复用旧短片冒充新长数据。
- 实际HO-Cap主机位24个采集/6人/22,729帧，RGB-D/K/mesh/公开参考已落地；56个物体ID、3个任务类型。EgoDemo24个新UUID/23,812帧/798.9307秒，单段20.13–59.63秒、9任务12场景；身份未知，仅辅助池。主视角共48独立采集/46,541帧，另6低视角是重复机位，不额外计数。
- 首/中/末预览确认48个主/辅助采集为真人手；HO-Cap低视角037排除，主视角105322251564全部22,729帧解码、SHA和内部几何结构检查通过。内部重投影一致性不是物理精度，预览不是全序列验收。HO-Cap实际时基未证实，不把默认30FPS算成实测长序列。
- 新增下载器、合并台账、参考结构审计、不可变freeze与测试；候选按subject整体16/4/4（主视角14,032/4,993/3,704帧），24条身份未知Ego辅助隔离。真实审计抓到pending样本权重为0导致全train，修复并新增回归；旧frozen_v1保留失败历史，当前本地frozen_v2，服务器frozen_server_v2_complete。
- 本地/服务器均1,081tests+84subtests通过；数据目录 `runs/human_dataset_v1_20261022/` 与 `/root/gpufree-data/datasets/ego_hoi/human_dataset_v1_20261022/`，详细协议/路径见 `docs/HUMAN_DATASET_V1_20261022.md`，汇总/分布/排除/SHA在数据根目录。未修改base/现有环境、未提交git、未启动新训练或预测评分。
- **未通过正式训练准入**：联合就绪0、training_allowed=false；120合格采集/4动作及各split数量不足，HO-Cap时间、场景、坐标/参考质量未全部核验，因果观测和有效掩码未生成，完整/≥5%帧复核待做。Ego无depth/K/mesh/物体GT和相机补偿，不冒充联合6D基准；固定第三人称HO-Cap不冒充第一人称。

## 2026-10-22 Daniel — 真实①② + Mamba GPU 验证 + 工位/清理收尾

- **EgoDemo 子集下载完成**：15 个机电类场景、328 个 mp4、19.47 GiB、0 个 incomplete；路径 `/root/gpufree-data/datasets/ego_hoi/raw/egodemo_subset/`。使用 hf CLI + hf-mirror + read token，未把 token 写入 git
- **真实①②复核通过**：2 个 head_left 视频重编码为 6 clips，clips 契约 0 error（2 个 10–18s 残段 warning）；2 个 hand_pose / 1200 帧 0 error，左/右/双手检出率 **75.2% / 83.8% / 63.3%**。产出在 `datasets/ego_hoi/{clips_real,hand_pose_real}/`
- **Mamba 在 L40 全验证通过**：causal_conv1d 1.7.0 + mamba_ssm 2.2.6.post3 本机 sm_89 wheel；FP32/BF16 前向反向、cuobjdump sm_89 cubin、3 epoch 训练与 checkpoint 重载均通过。train loss **4.3740→1.4405**、val loss **2.9758→0.9296**、峰值显存 **699 MiB**；产物 `/home/daniel/pipeline_run/runs/mamba_smoke/`
- **依赖兼容定格**：caron_mamba/daniel_test 均为 torch 2.6.0+cu124；transformers 5.20.2 与 mamba_ssm 不兼容，已降为 **transformers 4.56.2 + tokenizers 0.22.2 + huggingface-hub 0.36.2**。已独立复跑 Mamba GPU 前后向与 Arc 10 项 standalone 单测
- **工位最终结构**：7 个 `/home/<缩写>@<群昵称>/` 均无整仓 clone，只保留 README、实体 TASK、data 软链及本人任务实验/测试代码、refs 和产物；公用主代码仅 `/workspace/egodata_process`
- **清理完成**：删除 6 份工位冗余 clone、旧不兼容 wheel 与 `/tmp/miniconda.sh`（约 842 MiB）；Mamba 验证后删除临时 `src_build`（约 926 MiB），保留 `wheels/ubuntu2004/`。uv-cache 约 16G 保留用于加速重建；数据盘当前约 450G 可用
- **边界/遗留**：真实③ FoundationPose/深度/mesh 后端与真实④样本尚未完成；当前合成样本标签分布单一，冒烟准确率 1.0 不能作为模型效果结论。ModelScope ego_pose_dataset 仍为空壳

## 2026-10-22 Daniel — 迁移新服务器（L40）+ 全链路 GPU 冒烟

- **换机**：旧 4090 机器 GPU 挂不上弃用 → 新机器 `ssh -p 30314 root@120.209.70.195`（L40 46G，Ubuntu 20.04 glibc 2.31，数据盘 246G）
- 7 个 conda 环境在新机器全部重建（uv 阿里云；弃 1-2MB/s 的 rsync 方案）；wheels/工位/pipeline_run 已迁移
- **①→⑥ 全链路已端到端跑通**：切分（发现 -c copy 漂移，真实数据须 --reencode）→ 手部（mediapipe 定格 0.10.21）→ mock pose → 313 样本（check 全绿）→ **LSTM 训练 L40 上 loss 2.49→0.70**
- **EgoDemo 已获访问权**：元数据 clone 完成，全量 1.44TB 只拉子集；hf-mirror 7.3MB/s vs 直连 40KB/s，gated 走镜像必须 read token（已配）
- **mamba 走源码编译**（glibc 2.31 + sm_89 补丁 + FORCE_BUILD + --no-deps），详见 mamba-training skill；wheel 备份 `wheels/ubuntu2004/`
- 7 张任务卡已追加"下一步开发方向"（各模块实测发现 + 下一步）
- 遗留：mamba 编译验证与真实数据 ①② 由两个 agent 并行执行中；ego_pose_dataset（ModelScope）仍空壳

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
