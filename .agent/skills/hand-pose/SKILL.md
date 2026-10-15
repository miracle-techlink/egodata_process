---
name: hand-pose
description: 任务② 对每个 clip 提取双手 21 关键点手部姿态，产出 hand_pose/*.npz。负责人 Prick1yAsh。
---

# 手部姿态提取（hand-pose）

## 契约（详见 .agent/memory/03_data_pipeline.md ②）

- 输入：`data/ego_hoi/clips/<clip_id>.mp4`
- 输出：`hand_pose/<clip_id>.npz`：`left_hand (T,21,3)`、`right_hand (T,21,3)`、`conf (T,2)`，未检出帧填 NaN；附 meta.json
- 关键点顺序：MediaPipe Hands 21 点（0=腕，4=拇指尖，8=食指尖…）

## 基线方案：MediaPipe Hands

入门参考：https://blog.csdn.net/g11d111/article/details/115539407

```python
import mediapipe as mp
hands = mp.solutions.hands.Hands(
    static_image_mode=False,   # 视频流模式，帧间跟踪更稳
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
# results.multi_hand_landmarks + results.multi_handedness（注意 Left/Right 是镜像视角下的标签）
```

## 要点

- **handedness 坑**：MediaPipe 输出的 Left/Right 是相对自拍镜像的，处理第三人称/ego 视频时统一按 `multi_handedness` 标签写入 `left_hand/right_hand`，并在 meta 记录是否翻转过
- 第一视角视频手部遮挡多、检出率低：宁可整帧 NaN 也不要补假值；低置信帧同样置 NaN
- 批处理脚本 `src/hand_pose/extract.py`：支持 `--clips_dir --out_dir --num_workers`，断点续跑（已存在 npz 跳过）
- 性能：MediaPipe CPU 即可跑（约实时 2–5 倍速）；量大时用 GPU 版或后面升级 HaMeR/wilor（升级时保持 npz 契约不变）
- 首批先跑 3 个 clip，交 ⑦daniel 校验契约 + ④xw 可视化抽检

## 环境

```bash
conda create -n ash_hands python=3.10 -y && conda activate ash_hands
pip install mediapipe opencv-python numpy
```
