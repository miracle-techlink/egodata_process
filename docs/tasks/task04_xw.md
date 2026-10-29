# TASK ④ — 训练样本构建 + 可视化（负责人：谢武 / 工位 `/home/xw@谢武`）

## 你的使命

把 ②③ 的逐帧特征组装成**模型能直接吃的时序样本**，并做可视化让全组"看见"数据质量。你是数据侧的总装车间 + 质检员。

**职责已扩展（2026-10-29）**：数据侧新增 **VLM 语义预标注 → 自动质检 → 人工审核** 一段，由你和 ①🐠 共同维护——语义标签不再只靠启发式规则，VLM 候选经审核后才是标签来源。

## 输入 → 输出

| | 内容 | 位置 |
|---|---|---|
| 输入 | `hand_pose/*.npz` + `pose6d/*.npz` + VLM 标注产物（`events.jsonl`/`none_intervals.jsonl`） | `data/ego_hoi/`、`vlm_annotation_*/` |
| 输出 | `samples/{train,val,test}/*.npz` + `dataset_index.json` + `vis/*.mp4` | `data/ego_hoi/samples/`、`vis/` |

样本字段：`hand_seq (W,2,21,3)`、`obj_seq (W,N_max,13)`、`label`（下一帧主体/客体/类型）。契约细节：`.agent/memory/03_data_pipeline.md` ④节；标注契约：`hoi_annotation_v1`（同文件标注节）

## VLM 语义标注 + 质检流水线（新增）

```text
clips（① 切分）
  → VLM 预标注：Qwen2.5-VL-7B 抽证据帧出「主体/客体描述/六类交互/粗略边界」，全部 pending
  → 自动质检：review_pipeline.py 复核时间轴/手姿态/几何/标注结构（docs/AUTOMATIC_REVIEW_PIPELINE.md）
  → 人工审核：稠密审核页逐事件 accepted / rejected / needs_revision
  → accepted-only 投影成三头标签（pending/重叠/未链接 track 一律 label_valid=0，不进 loss）
```

**铁律**：
- VLM 只产 pending 候选；accepted 必须有 reviewer、reviewed_at、真实 `object_id`，不能由集成脚本擅自通过
- 覆盖不完整时报 `coverage_status=incomplete`，**不许把"没标到"写成"零交互"**
- VLM 离线标注看过全事件，`allowed_as_model_input=false`，只能当训练监督，不能当模型输入特征

**GPU 并行加速（VLM 推理）**：
- 现状：`src/annotation/vlm_server_v5.py`——单模型实例 + 手写批处理队列（MAX_BATCH=6、MAX_WAIT=0.3s），N 个标注 worker 共享一张卡
- 目标：**vLLM 连续批处理**（PagedAttention，吞吐显著高于手写 batcher）。启动：`vllm serve /root/gpufree-data/models/Qwen2.5-VL-7B-Instruct --port 8100`，标注客户端走 OpenAI 兼容接口；切换前注意 `auto_review_v5` 目前是自定义 `POST /ask` 契约，需要适配层
- 排产注意：L40 48GB，7B 模型 bf16 约 19GB 峰值；跑 VLM 前先 `nvidia-smi` 确认无其他任务占卡（训练与标注错峰）

## 当前状态

- ✅ 环境可用：`conda activate xw_dataset`（numpy + opencv + matplotlib + tqdm）
- ✅ 契约字段已冻结；313 个合成样本通过校验（0 error / 0 warning），LSTM/Mamba 均能读取训练
- ✅ 首批真实 hand_pose 已产出；⏳ 真实 pose6d 仍等 ③，当前不能把 mock 结果当真实物体轨迹
- ✅ VLM 预标注已在真实视频跑通（Qwen2.5-VL-7B，权重 SHA256 全验证）：单段 11 原始候选 → 9 有效 pending + 2 拒绝，尚无 accepted——**正式标签仍为空，0/120**
- ✅ 自动质检流水线可用：`review_pipeline.py` 支持并行、断点续跑、按内容变化重检（`configs/review_pipeline_300.json`）
- 🔑 下一步：接真实 ②③ npz，出标签分布与 ≥5% 可视化抽检；配合 VLM 标注把 accepted 标签真正产出来

## 实施步骤

1. **D1 可视化先行**：`src/dataset/visualize.py`——手部骨架+物体 bbox+6D 坐标轴叠回原视频出 mp4（②③ 的质检都靠它）
2. **D2 样本构建**：滑窗（W=32【草案】、stride 16），物体特征 13 维（xyz+rot6d+面积+速度），物体数补零到 N_max + mask
3. **D3 切分**：train/val/test **按 source_video 隔离**（防泄漏），索引落 `dataset_index.json`
4. **D4 标签**：六类交互体系已冻结为 {none,reach,grasp,hold,move,place,release}；标签来源优先级：人工 accepted 的 VLM 候选 > 启发式规则（`label_rules.py`，仅草案兜底），最终规则与 ⑤arc 对齐后回写契约
5. **D5 VLM 标注批量跑**：① 的 clips 全量过 VLM 预标注 → `review_pipeline.py` 自动质检 → 人工审核页分派；每天更新 accepted 计数与覆盖状态

## 验收标准

- [ ] `tests/check_contracts.py samples <dir>` 通过
- [ ] 抽检 ≥5% clip 可视化：左右手不反、物体轴不漂、时间对齐，结论记 `quality_flags`
- [ ] split 无视频级泄漏（脚本断言检查）
- [ ] 标签全部来自 accepted 事件或显式 accepted none；`evaluate_scale` 报告的 coverage 为 `provided_complete` 才允许结算 accepted 数
- [ ] 更新进度日志，提 PR（分支 `xw/dataset-builder`）

## 求助与参考

- 手册：`.agent/skills/build-training-data/SKILL.md`、`.agent/skills/vlm-annotation/SKILL.md`
- 自动质检：`docs/AUTOMATIC_REVIEW_PIPELINE.md`；规模口径：`src/dataset/evaluate_scale.py`
- 上游字段争议 → 回 `03_data_pipeline.md` 对齐，别单方面适配

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：build_samples 已修 3 处（恢复帧速度 NaN、index 结构、contract_version），313 个样本通过契约校验 0 error 0 warning；index 冻结为 `{train,val,test}`，富元信息在 meta.json。

1. **真实数据标签分布**：合成数据 label 全 none 无意义，真实 ②③ 产出后看 `meta.json` 的 interaction_counts——启发式阈值（contact/reach margin、vel_thresh=0.05m/s）大概率要按真实尺度调，先出分布报告再调参
2. **与 ⑤arc 对齐交互标签规则**：label_rules.py 是草案，重点是 grasp/hold/move/place/release 的状态转移定义，对齐后回写契约冻结
3. **坐标系不一致（已知研究问题）**：手部是归一化图像坐标、物体是相机系米制，目前 contact 在图像空间判定是权宜之计；③ 内参落实后评估手部关键点升维到相机系
4. 可视化在真实 clip 上抽检 ≥5%：左右手不反、物体轴不飘、时间对齐
