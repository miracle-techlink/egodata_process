# TASK ④ — 训练样本构建 + 可视化（负责人：谢武 / 工位 `/home/xw@谢武`）

## 你的使命

把 ②③ 的逐帧特征组装成**模型能直接吃的时序样本**，并做可视化让全组"看见"数据质量。你是数据侧的总装车间 + 质检员。

## 输入 → 输出

| | 内容 | 位置 |
|---|---|---|
| 输入 | `hand_pose/*.npz` + `pose6d/*.npz` | `data/ego_hoi/` |
| 输出 | `samples/{train,val,test}/*.npz` + `dataset_index.json` + `vis/*.mp4` | `data/ego_hoi/samples/`、`vis/` |

样本字段：`hand_seq (W,2,21,3)`、`obj_seq (W,N_max,13)`、`label`（下一帧主体/客体/类型）。契约细节：`.agent/memory/03_data_pipeline.md` ④节

## 当前状态

- ✅ 环境已预装：`conda activate xw_dataset`（numpy + opencv + matplotlib + tqdm）
- ⏳ 等 ②ash、③panda 的首批 npz（等料期间：先写好可视化脚本，用合成 npz 自测）
- 🔑 **开工第一件事**：和 ash、panda 逐字对齐 npz 字段名（跑 `tests/check_contracts.py`）

## 实施步骤

1. **D1 可视化先行**：`src/dataset/visualize.py`——手部骨架+物体 bbox+6D 坐标轴叠回原视频出 mp4（②③ 的质检都靠它）
2. **D2 样本构建**：滑窗（W=32【草案】、stride 16），物体特征 13 维（xyz+rot6d+面积+速度），物体数补零到 N_max + mask
3. **D3 切分**：train/val/test **按 source_video 隔离**（防泄漏），索引落 `dataset_index.json`
4. **D4 标签**：与 arc 敲定交互类型体系【草案 {none,reach,grasp,hold,move,place,release}】，标注规则回写契约

## 验收标准

- [ ] `tests/check_contracts.py samples <dir>` 通过
- [ ] 抽检 ≥5% clip 可视化：左右手不反、物体轴不漂、时间对齐，结论记 `quality_flags`
- [ ] split 无视频级泄漏（脚本断言检查）
- [ ] 更新进度日志，提 PR（分支 `xw/dataset-builder`）

## 求助与参考

- 手册：`.agent/skills/build-training-data/SKILL.md`
- 上游字段争议 → 回 `03_data_pipeline.md` 对齐，别单方面适配
