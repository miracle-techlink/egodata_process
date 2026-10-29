# 04 — 资源汇总

## 2026-10-24 自动化复核入口

- 使用说明：[docs/AUTOMATIC_REVIEW_PIPELINE.md](../../docs/AUTOMATIC_REVIEW_PIPELINE.md)；入口`src/annotation/review_pipeline.py`，服务器配置`configs/review_pipeline_300.json`。
- 运行输出：`/root/gpufree-data/datasets/ego_hoi/automatic_review_300_20261024`；本地同名`runs/`镜像。中文`report.html`、`复核汇总.csv`、按原生帧段归并的`review_queue.jsonl`及可续跑`jobs.json`。
- 实测覆盖300段868255帧、V5全部1238窗口，80项针对性回归通过。自动检查通过与全标注完成独立统计，866260帧仍待连续语义复核；不更改V5准入或冻结包。

## 2026-10-24 V5可训练交互子集

- 数据：`/root/gpufree-data/datasets/ego_hoi/egodemo_training_v5_300_20261024`；本地`runs/egodemo_training_v5_300_20261024`。
- 接手：[docs/TRAINING_V5_HANDOFF.md](../../docs/TRAINING_V5_HANDOFF.md)；源接口`src/mamba/dataset_v5.py`、图模型`src/mamba/graph_v5.py`、真实GPU验收`src/mamba/graph_v5_smoke.py`。
- 数量：5采集、1995连续已看帧、1238实际窗口592/366/280；21实测监督类型，训练仅12。全部300采集仍保留；剩866260帧未连续复核，整物体6D/完整联合仍未放行。
- 冻结index SHA `4b162d6595f7b489a078d86b096732a2060300cc3ea5d92296463fccfcb179e6`；release SHA `58b8da4620b027eecb92e35fddde98af07525506b68d66f2a34f085b2cec5983`。
- 数字可移植包`egodemo_training_v5_300_20261024_interaction_subset.tar.gz`及SHA文件；完整范围以`PACKAGE_SCOPE.json`为准，源视频/PNG为私有外部审核依赖。
- 私有中文只读Web：`127.0.0.1:18791/review_web/index.html`，SSH端口转发18791；旧18790/V4保留。


## 项目资产

### 2026-10-24当前私有数据与标注入口

- 当前连续标注V4：服务器`/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v4_dense_300_20261024`，本地镜像`runs/egodemo_annotation_v4_dense_300_20261024/`。300段868255帧数值索引齐；3段1054帧实际连续复核，其中2段整段看完，完整标注仍未完成。入口`dense_delivery_report.json`、`逐段连续标注缺口.csv`、`数据格式与使用说明.json`、`mamba_dense_index.json`。866个新增诊断窗口全部属于原train；旧单客体头新增可用监督0。
- V4中文连续审核：私有`127.0.0.1:18790/review_web/index.html`，通过`ssh -p30314 -L 18790:127.0.0.1:18790 root@120.209.70.195`转发访问；只读，未渲染帧明确显示缺图。服务根为上述V4；旧18788保留。原生逐帧裁剪和页面证据保存在私有服务器`dense_evidence/`，不公开发布。
- 冻结300采集源池：服务器`/root/gpufree-data/datasets/ego_hoi/egodemo_expansion_300_20261023`；300独立UUID、75原始任务、868255帧，源SHA和固定选择保持不变。
- 第二轮几何与中文提案：`egodemo_refine_300_20261023`，162段含过滤后刚体/部件6D候选，299段有未被标定拦截的表面点。私有中文审核服务127.0.0.1:18788，经SSH转发访问，不公开原视频或逐条元信息。
- 实际Mamba v2：同数据父目录下`egodemo_mamba_v2_300_20261023`；数据、冻结split、中文说明、真实窗口样本、源码快照、全量审计、真实L40前向均在。正式训练未准入。
- 当前已交付代理抽样标注：`egodemo_annotation_v3_300_20261023`；不同于旧human批准，实际视觉复核记录单列assistant银标与不确定性。300段1666源区间均已抽样复核；精确边界和几何完整性仍未完成。加载器为`src/mamba/annotation_v3.py`，实际可用窗口见`annotation_target_coverage.json`，最新验收见05日志；不覆盖原始数据或v2。

| 资源 | 地址 |
|------|------|
| 代码仓库 | https://github.com/miracle-techlink/egodata_process |
| 处理好的数据集 | https://www.modelscope.cn/datasets/Liuyue9698/ego_pose_dataset ⚠️ 2026-10-15 查：**仍是空壳**（只有 README），数据待上传 |
| 原始数据参考 | https://huggingface.co/datasets/LightwheelAI/EgoDemo ✅ 2026-10-22 **已获访问权**（SSH key + read token 已配在新服务器）。⚠️ 全量 1.44TB（四机位），只拉子集：`/root/gpufree-data/datasets/ego_hoi/raw/egodemo_subset/`；下载用 hf CLI + hf-mirror + token（直连 hf.co 极慢）。结构 `EgoProRaw/mp4/<场景>/<uuid>/<camera>/`，119 场景，单视频约 50MB；manifests/*.parquet 可能含标定信息（③ 待挖） |
| 云端环境配置指南 | https://github.com/miracle-techlink/gpu-rsi-plugin （精华版已收录：`skills/env-setup/SKILL.md`） |

## 标注版试点（2026-10-22 实查）

- 已下载 Raw 子集的 82 个采集 UUID 全部可匹配标注版：EgoProStandard 61 个、EgoProStandard-body 21 个；不能把 328 个机位视频当成 328 个独立采集。
- 标注版包含双手世界坐标、头部/相机位姿与 `annotation.json` 原文子任务时间区间。原文事件不是现有七类 HOI，也没有补齐物体 6D 真值。
- 原始版与标注版时长/帧数不同，必须使用标注版配套视频及时间戳。manifest FPS 与 `meta/info.json` 标称 FPS 也可能因统计定义不同而不等，保留两者，以原始时间戳为时间基准。
- 工具入口：`src/dataset/egodemo_manifest.py` 与 `src/dataset/import_egodemo.py`；首批 10 段资产及校验清单见 `docs/DELIVERABLES.md`。

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
