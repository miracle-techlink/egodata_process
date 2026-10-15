---
name: build-training-data
description: 任务④ 将手部姿态+6D pose 组装成时序训练样本（滑窗 npz + dataset_index.json）并做可视化抽检。负责人 谢武。
---

# 训练样本构建 + 可视化（build-training-data）

## 契约（详见 .agent/memory/03_data_pipeline.md ④）

- 输入：`hand_pose/<clip_id>.npz` + `pose6d/<clip_id>.npz`
- 输出：
  - `samples/{train,val,test}/<sample_id>.npz`：`hand_seq (W,2,21,3)`、`obj_seq (W,N_max,13)`、`label`
  - `samples/dataset_index.json`：样本清单 + split 划分
  - `vis/<clip_id>.mp4`：叠加可视化

## 要点

### 样本构建

- 滑窗：窗口长 W【草案默认 32 帧】、stride 16；label 取窗口**下一帧**的交互标注（主体/客体/类型）
- **split 按 source_video 隔离**（同一原视频的 clip 不跨 train/val/test），防数据泄漏
- 每帧物体数不固定 → 对齐到 `N_max`（数据集统计的 P95），不足补零 + mask 数组
- 缺失值策略：上游 NaN 保留，另存 `valid_mask`，让模型侧决定掩码方式
- 物体特征 13 维：位置 xyz(3) + 旋转 6D 表示(6) + bbox 面积(1) + 速度(3)（旋转用 6D 连续表示，避免四元数不连续）

### 可视化（与 ②③ 协作）

- 脚本 `src/dataset/visualize.py`：把手部 21 点骨架 + 物体 bbox + 6D pose 坐标轴叠加回原视频，输出 mp4
- 抽检比例：每批至少 5%，重点看：手部左右是否反、物体轴是否漂移、时间是否对齐
- 抽检结论记录到对应 meta.json 的 `quality_flags`

## 与上下游的对齐点

- 开工前和 ②ash、③panda 确认 npz 字段名逐字一致（跑 `tests/check_contracts.py`）
- label 的 `interaction_type` 标签体系与 ⑤arc 共同敲定后回写契约文件

## 环境

```bash
conda create -n xw_dataset python=3.10 -y && conda activate xw_dataset
pip install numpy opencv-python matplotlib tqdm
```
