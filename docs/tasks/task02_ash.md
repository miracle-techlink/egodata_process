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

- ✅ `conda activate ash_hands`：**mediapipe 定格 0.10.21**，不要升级
- ✅ 2 个真实 clips / 1200 帧已提取并通过契约校验（0 错误）；左手检出 75.2%、右手 83.8%、双手 63.3%
- ⏳ 待做真实视频左右手可视化抽检、meta 推荐键修正和子集批处理

## 实施步骤

1. **D1 跑通**：`src/hand_pose/extract_hands.py`，MediaPipe Hands 视频流模式（`static_image_mode=False`, `max_num_hands=2`, 置信度 0.5），真实 clips 已跑通 ✅
2. **D2 验证**：关键点叠图/叠视频自检 + 交 Daniel 契约校验 + 交 xw 可视化抽检
3. **D3 全量**：批量 + 断点续跑（已存在的 npz 跳过），产出覆盖率统计（各 clip 检出率入 meta）

## 高发坑（必读）

- **左右手标签**：MediaPipe 的 Left/Right 是自拍镜像视角的标签，第一视角素材要验证并统一规则，meta 里记录。左右反了 = 下游全错
- 宁可整帧 NaN 不要补假值；低置信帧同样置 NaN

## 验收标准

- [x] `tests/check_contracts.py hand_pose <目录>` 首批真实数据通过（0 错误）
- [ ] 3 个 clip 可视化抽检（xw 配合）左右手无误、时间对齐
- [x] 检出率统计入 meta.json
- [ ] 更新进度日志，提 PR（分支 `ash/hand-pose-extract`）

## 求助与参考

- 手册：`.agent/skills/hand-pose/SKILL.md`
- 入门教程：https://blog.csdn.net/g11d111/article/details/115539407
- 官方文档：https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：extract_hands.py 在服务器真实跑通（8/8 clips）。**环境已定 mediapipe==0.10.21**（1.1.1 移除了 `mp.solutions` legacy API，不要升级！）。

1. **真实视频检出率评估**：合成测试视频全 NaN 是预期，真实 EgoDemo 画面上的左右手检出率才是指标（meta.json stats）。若检出率低：调 `min_hand_detection_confidence`、试用 wrist 机位视频、或评估 HaMeR 等更强模型（先出报告再换）
2. **左右手标签抽检**：用 ④ 的 visualize 叠图人工确认 ego 视角下 handedness 不反（验收硬指标）
3. **meta.json 键名对齐**：check_contracts 目前对 meta 推荐键 `source_clip/created_at/env` 给 warning——与 Daniel 统一命名后改掉（小改）
4. 吞吐优化：`--num_workers` 多进程跑大批量，记录 clips/小时
