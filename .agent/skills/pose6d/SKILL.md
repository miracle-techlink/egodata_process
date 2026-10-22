---
name: pose6d
description: 任务③ 用 FoundationPose 提取目标物体 6D 位姿（位置+旋转），产出 pose6d/*.npz。负责人 空腹panda。
---

# 物体 6D pose 提取（pose6d）

## 契约（详见 .agent/memory/03_data_pipeline.md ③）

- 输入：`clips/<clip_id>.mp4` + 深度（若可得）+ 物体 mask/检测框
- 输出：`pose6d/<clip_id>.npz`：`object_ids (N,)`、`poses (N,T,4,4)`（物体→相机）、`bboxes (N,T,4)`；poses 未检出填 NaN，bboxes 未检出填 -1

## 方案：FoundationPose（NVlabs）

仓库：https://github.com/NVlabs/FoundationPose

前置依赖检查清单（**开工第一天先逐项确认**）：

1. **深度**：FoundationPose 需要 RGB-D。若 EgoDemo/ego_pose_dataset 无深度通道 → 走降级路径：DepthAnything V2 单目深度估计补齐，meta 记 `depth_source: depth_anything_v2`
2. **mask**：需要首帧物体 mask → 用 Grounded-SAM / SAM2 自动出 mask，视频内由 FoundationPose 跟踪
3. **物体 mesh**：model-free 模式可用 RGB-D 重建（NeRF/bundlesdf），无需提前 CAD
4. **显存**：现役 L40 46GB 够单物体跟踪；多物体逐物体串行跑，批量前先实测显存

## 安装要点（CUDA 编译，独立环境！）

```bash
conda activate panda_6d   # 已预装 torch 2.14 + trimesh + open3d + opencv
# 按 FoundationPose README 补装 kaolin / nvdiffrast / bundlesdf 等 CUDA 扩展
```

- ⚠️ FoundationPose 生态对 torch 版本敏感（官方年代为 torch 2.0–2.5）；若扩展编译与 torch 2.14 冲突，把 panda_6d 降到 `torch==2.6.0` + cu126 再试
- 预编译 wheel 优先原则见 `.agent/skills/env-setup`；编译卡住先看官方 issues；不要用 docker（我们已在容器内）

## 要点

- 输出统一**物体→相机** 4×4 矩阵（FoundationPose 原生输出即是），不要混用相机→物体
- 同一 clip 多物体：`object_ids` 顺序固定（按首帧 mask 面积排序），跨帧 id 一致
- 先 1 个 clip 单物体跑通 → 交 ⑦daniel 校验 → 再多物体批量
- 跟踪漂移：可视化（④xw 的叠加脚本）抽检，漂移严重的 clip 记录到 meta `quality_flags`

## 协作

- mask/检测若工作量太大，找 ④xw 分工（他也在做数据侧工具）
