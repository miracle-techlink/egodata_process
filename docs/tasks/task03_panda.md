# TASK ③ — 物体 6D pose 提取（负责人：空腹panda / 工位 `/home/panda@空腹panda`）

## 你的使命

给每个目标物体输出**逐帧 6D 位姿**（位置+旋转）。交互是三维空间关系，2D 框表达不了"手在逼近哪个物体"——你的输出是交互建模的一半。

## 输入 → 输出

| | 内容 | 位置 |
|---|---|---|
| 输入 | ① 的 clips（+深度/mask，见前置检查） | `data/ego_hoi/clips/` |
| 输出 | `pose6d/<clip_id>.npz` | `data/ego_hoi/pose6d/` |

npz 字段：`object_ids (N,)`、`poses (N,T,4,4)`（**物体→相机**）、`bboxes (N,T,4)`；poses 未检出填 NaN，bboxes 未检出填 -1。契约细节：`.agent/memory/03_data_pipeline.md` ③节

## 当前状态

- ✅ 环境基础已备：`conda activate panda_6d`（torch + trimesh + open3d + opencv）
- ✅ 参考代码在工位：`refs/FoundationPose`、`refs/Depth-Anything-V2`；L40 GPU 与真实 clips 均已就位
- ✅ mock 后端已通过合成链路契约验证；⚠️ **真实 FoundationPose/深度/mesh 后端尚未跑通，是当前全真实链路的主要缺口**
- 编译注意本机是 Ubuntu 20.04 + sm_89，nvcc 在 `/usr/local/cuda`

## 前置检查清单（开工第一天做完）

1. 数据有无深度通道？没有 → Depth-Anything-V2 补齐，meta 记 `depth_source`
2. 首帧 mask 来源 → Grounded-SAM / SAM2 自动出（工作量大找 xw 分工）
3. 物体 mesh → model-free 模式用 RGB-D 重建，无需 CAD

## 实施步骤

1. **D1**：FoundationPose 环境补装（nvdiffrast/kaolin/bundlesdf，全装自己环境，优先预编译 wheel——见 `.agent/skills/env-setup`）
2. **D2 跑通**：1 个 clip 单物体跟踪，输出 npz → Daniel 契约校验
3. **D3**：多物体串行（id 按首帧 mask 面积排序，跨帧一致），漂移 clip 记 `quality_flags`

## 验收标准

- [ ] `tests/check_contracts.py pose6d <npz>` 通过
- [ ] 可视化（xw）抽检：物体坐标轴贴合、无明显漂移
- [ ] 坐标系确为物体→相机
- [ ] 更新进度日志，提 PR（分支 `panda/pose6d`）

## 求助与参考

- 手册：`.agent/skills/pose6d/SKILL.md`
- https://github.com/NVlabs/FoundationPose ｜ https://github.com/DepthAnything/Depth-Anything-V2

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：extract_pose6d.py 三段式管线已跑通（bbox-from-file + mock pose，8/8 clips 契约全过）。**新服务器 L40 GPU 已可用**，真实后端可以开工了。

1. **相机内参从数据集 manifest 挖**：EgoDemo 有 `EgoProRaw/manifests/*.parquet` 和 `studio.parquet`，先解析看有没有标定/内参/深度——有就接入 `--intrinsics-json`，消灭启发式内参（这直接决定 ④⑤ 的坐标系质量）
2. **编译依赖注意**：本机 Ubuntu 20.04 glibc 2.31，nvdiffrast/kaolin 等一律源码编译（参考 mamba-training skill 的 FORCE_BUILD/sm_89 补丁经验，L40 是 sm_89）
3. **分步打开后端**：先 DepthAnything 深度 → 再 FoundationPose（需 mesh）；GroundedSAM 检测器 API 可能要适配 Grounded-SAM-2
4. **model-free 重建（bundlesdf）是当前最大缺口**：没有 mesh 的物体走不了 FoundationPose register，评估是否作为主线方案
5. 每步都在真实 clip 上跑 check_contracts + 可视化抽检（物体轴不飘、id 跨帧一致）
