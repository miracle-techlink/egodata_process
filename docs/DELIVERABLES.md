# 交付物对齐表（Deliverables Alignment）

> 用途：开工前全员对齐——每个阶段**交给下游什么东西、什么格式、谁验收**。
> 格式细节的唯一事实源是 `.agent/memory/03_data_pipeline.md`，本表只做交付关系对齐，不重复格式定义。
> 2026-10-15 Daniel 整理；2026-10-22 按 L40 实测更新。

## 交付链路总览

```
① fish 切分        clips/*.mp4 + clips_index.csv
    ↓
② ash 手部姿态     hand_pose/<clip_id>.npz ──────────┐
    ↓                                                ↓
③ panda 6D pose   pose6d/<clip_id>.npz ────────→ ④ xw 样本构建+可视化
                                                     samples/<split>/*.npz + dataset_index.json + vis/*.mp4
                                                        ↓
                                              ⑤ arc encoder/decoder（import 接口）
                                                        ↓
                                              ⑥ caron Mamba 训练 → runs/<run>/best.pt
                                                        ↓
                                              ⑦ daniel 全链路校验 + 冒烟（贯穿始终）
```

## 每阶段交付卡片

### ① fish → 给 ②③
- **交付物**：`clips/<clip_id>.mp4`（~20s/段）+ `clips/clips_index.csv`
- **验收人**：daniel（`tests/check_contracts.py clips`）+ ash/panda 各抽 1 段试跑
- **验收标准**：csv 每行可被 ffprobe 复核（时长/帧数一致）；clip_id 全局唯一；<10s 残段已丢弃并备注
- **前置依赖**：EgoDemo 已获访问权；15 场景子集已下载（328 视频，19.47 GiB）
- **当前状态**：2 个真实视频已 `--reencode` 成 6 clips，0 错误；两段 10–18s 残段为预期 warning。下一步批量切分与多机位整理

### ② ash → 给 ④
- **交付物**：`hand_pose/<clip_id>.npz`（left/right (T,21,3) + conf (T,2)，NaN 规则）+ meta.json
- **验收人**：daniel 契约校验 + xw 试读与可视化抽检
- **验收标准**：未检出帧整帧 NaN 而非 0；左右手判定经人工抽检不互换
- **前置依赖**：① clips 已就位；mediapipe 定格 0.10.21
- **当前状态**：2 个真实 clips / 1200 帧通过校验（0 错误），左/右/双手检出率 75.2% / 83.8% / 63.3%；meta 推荐键与可视化抽检待完成

### ③ panda → 给 ④
- **交付物**：`pose6d/<clip_id>.npz`（object_ids / poses (N,T,4,4) / bboxes (N,T,4)）+ meta.json（注明 depth_source）
- **验收人**：daniel 契约校验 + xw 试读 + 可视化抽检（轴向不飘）
- **验收标准**：同一物体跨帧 id 一致；poses 未检出为 NaN、bboxes 未检出为 -1（以契约为准）
- **前置依赖**：① clips 与 L40 已就位；还需相机内参、深度、mask 与物体 mesh/重建路线
- **当前状态**：mock 后端只用于合成链路验证；**真实 FoundationPose/深度/mesh 后端尚未跑通，是当前主要缺口**

### ④ xw → 给 ⑤⑥
- **交付物**：`samples/{train,val,test}/*.npz` + `dataset_index.json` + `vis/<clip_id>.mp4`
- **验收人**：daniel 契约校验 + arc/caron 试读一个 batch
- **验收标准**：split 按 source_video 隔离；subject/object_slot/object_id/interaction 标签键齐全；obj 特征 13 维顺序正确（pos3+rot6d+area1+vel3）
- **前置依赖**：真实 ② 已有首批输出；真实 ③ 待交付；interaction_type 启发式规则需按真实分布校准
- **当前状态**：313 个合成样本 0 error / 0 warning，可供⑤⑥联调；真实样本、标签分布与 ≥5% 可视化是下一步

### ⑤ arc → 给 ⑥
- **交付物**：`src/encoding/` 的 `build_encoder(cfg)` / `build_decoder(cfg)`，import 即用
- **验收人**：caron（训练脚本 forward/backward）+ daniel（`python tests/test_encoding.py`）
- **验收标准**：支持可变物体数（padding mask）；输出三路 logits；NaN/mask/gaze 接口单测通过
- **前置依赖**：④ 样本格式与接口契约已冻结
- **当前状态**：10 项单测及 LSTM/Mamba 集成冒烟通过；下一步真实样本 overfit 与手/物坐标单位联调

### ⑥ caron → 给全组（训练结果）
- **交付物**：`src/mamba/` 训练脚本 + `configs/*.yaml` + `runs/<run>/best.pt` + `metrics.jsonl`（或 tensorboard 日志）
- **验收人**：daniel（冒烟跑通）+ 全组（实验报告）
- **验收标准**：配置驱动；bf16；checkpoint 含契约版本；附 LSTM 基线开关
- **前置依赖**：④⑤ 与 L40 已满足合成冒烟；真实训练等④的真实 samples
- **当前状态**：sm_89 wheel、FP32/BF16 前后向、3 epoch 训练与 checkpoint 重载通过；train loss 4.3740→1.4405，峰值 699 MiB。合成标签单一，准确率不作效果结论

### ⑦ daniel → 给全组（质量闸门）
- **交付物**：`tests/check_contracts.py`（①–④ 产物校验 CLI）+ `tests/test_encoding.py` + 冒烟产物
- **验收人**：全组使用
- **前置依赖**：契约已冻结
- **当前状态**：25 项 checker 自测、合成①→⑥、真实①②与 Mamba GPU 冒烟已通过；③真实后端接入后重跑全真实链路

## 全员对齐要点

1. **只认契约**：上下游对格式有分歧时以 `03_data_pipeline.md` 为准；要改格式先改契约并 @ 上下游（AGENTS.md 铁律 1）
2. **合成与真实分开表述**：合成/mock 只证明代码链路可运行；真实实验必须来自真实③④产物，并检查标签分布和可视化
3. **外部阻塞已解除**：EgoDemo 访问与 GPU 均已就绪；当前瓶颈是③真实后端、真实标签质量，以及 ModelScope 数据集仍为空壳
4. **完工即更新** `05_progress_log.md`
