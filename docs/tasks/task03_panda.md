# TASK ③ — 物体 6D pose 提取（负责人：空腹panda / 工位 `/home/panda@空腹panda`）

## 你的使命

给每个目标物体输出**逐帧 6D 位姿**（位置+旋转）。交互是三维空间关系，2D 框表达不了"手在逼近哪个物体"——你的输出是交互建模的一半。

## 输入 → 输出

| | 内容 | 位置 |
|---|---|---|
| 输入 | ① 的 clips（+深度/mask，见前置检查） | `data/ego_hoi/clips/` |
| 输出 | `pose6d/<clip_id>.npz` | `data/ego_hoi/pose6d/` |

npz 字段：`object_ids (N,)`、`poses (N,T,4,4)`（**物体→相机**）、`bboxes (N,T,4)`；未检出帧 NaN。契约细节：`.agent/memory/03_data_pipeline.md` ③节

## 当前状态

- ✅ 环境基础已备：`conda activate panda_6d`（torch + trimesh + open3d + opencv）
- ✅ 参考代码已克隆到工位：`refs/FoundationPose`、`refs/Depth-Anything-V2`（深度缺失时的降级方案）
- ⚠️ **GPU 当前未挂载**，FoundationPose 的 CUDA 扩展编译与推理需要 GPU 就位（nvcc 12.6 已有，编译本身可以先试）
- ⏳ 等 ①fish 出 clips

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
