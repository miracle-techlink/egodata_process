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

- ✅ 环境可用：`conda activate xw_dataset`（numpy + opencv + matplotlib + tqdm）
- ✅ 契约字段已冻结；313 个合成样本通过校验（0 error / 0 warning），LSTM/Mamba 均能读取训练
- ✅ 首批真实 hand_pose 已产出；⏳ 真实 pose6d 仍等 ③，当前不能把 mock 结果当真实物体轨迹
- 🔑 下一步：接真实 ②③ npz，出标签分布与 ≥5% 可视化抽检，不再等字段对齐

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

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：build_samples 已修 3 处（恢复帧速度 NaN、index 结构、contract_version），313 个样本通过契约校验 0 error 0 warning；index 冻结为 `{train,val,test}`，富元信息在 meta.json。

1. **真实数据标签分布**：合成数据 label 全 none 无意义，真实 ②③ 产出后看 `meta.json` 的 interaction_counts——启发式阈值（contact/reach margin、vel_thresh=0.05m/s）大概率要按真实尺度调，先出分布报告再调参
2. **与 ⑤arc 对齐交互标签规则**：label_rules.py 是草案，重点是 grasp/hold/move/place/release 的状态转移定义，对齐后回写契约冻结
3. **坐标系不一致（已知研究问题）**：手部是归一化图像坐标、物体是相机系米制，目前 contact 在图像空间判定是权宜之计；③ 内参落实后评估手部关键点升维到相机系
4. 可视化在真实 clip 上抽检 ≥5%：左右手不反、物体轴不飘、时间对齐
