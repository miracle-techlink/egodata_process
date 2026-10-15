# TASK ② — 手部姿态提取（负责人：Prick1yAsh / 工位 `/home/ash@Prick1yAsh`）

## 你的使命

从每个 clip 中提取**双手 21 关键点**时序——意图前兆最先出现在手上，你的数据质量直接决定预测上限。

## 输入 → 输出

| | 内容 | 位置 |
|---|---|---|
| 输入 | ① 的 clips | `data/ego_hoi/clips/<clip_id>.mp4` |
| 输出 | `hand_pose/<clip_id>.npz` | `data/ego_hoi/hand_pose/` |

npz 字段：`left_hand (T,21,3)`、`right_hand (T,21,3)`、`conf (T,2)`，未检出帧填 NaN；附 meta.json。契约细节：`.agent/memory/03_data_pipeline.md` ②节

## 当前状态

- ✅ 环境已预装：`conda activate ash_hands`（mediapipe + opencv + numpy 已装好）
- ⏳ 等 ①fish 出 clips（等料期间：用手机拍一段 30s 操作视频当输入，先把提取管线跑通）

## 实施步骤

1. **D1 跑通**：`src/hand_pose/extract.py`，MediaPipe Hands 视频流模式（`static_image_mode=False`, `max_num_hands=2`, 置信度 0.5），处理 1 个 clip
2. **D2 验证**：关键点叠图/叠视频自检 + 交 Daniel 契约校验 + 交 xw 可视化抽检
3. **D3 全量**：批量 + 断点续跑（已存在的 npz 跳过），产出覆盖率统计（各 clip 检出率入 meta）

## 高发坑（必读）

- **左右手标签**：MediaPipe 的 Left/Right 是自拍镜像视角的标签，第一视角素材要验证并统一规则，meta 里记录。左右反了 = 下游全错
- 宁可整帧 NaN 不要补假值；低置信帧同样置 NaN

## 验收标准

- [ ] `tests/check_contracts.py hand_pose <npz>` 通过
- [ ] 3 个 clip 可视化抽检（xw 配合）左右手无误、时间对齐
- [ ] 检出率统计入 meta.json
- [ ] 更新进度日志，提 PR（分支 `ash/hand-pose-extract`）

## 求助与参考

- 手册：`.agent/skills/hand-pose/SKILL.md`
- 入门教程：https://blog.csdn.net/g11d111/article/details/115539407
- 官方文档：https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
