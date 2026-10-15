# 交付物对齐表（Deliverables Alignment）

> 用途：开工前全员对齐——每个阶段**交给下游什么东西、什么格式、谁验收**。
> 格式细节的唯一事实源是 `.agent/memory/03_data_pipeline.md`，本表只做交付关系对齐，不重复格式定义。
> 2026-10-15 Daniel 整理。

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
- **验收人**：daniel（`tests/check_contracts.py` ① 模式）+ ash/panda 各抽 1 段试跑
- **验收标准**：csv 每行可被 ffprobe 复核（时长/帧数一致）；clip_id 全局唯一；<10s 残段已丢弃并备注
- **前置依赖**：EgoDemo 访问权限（gated，⚠️ 待申请）
- **当前状态**：代码就绪即可，等数据集

### ② ash → 给 ④
- **交付物**：`hand_pose/<clip_id>.npz`（left/right (T,21,3) + conf (T,2)，NaN 规则）+ meta.json
- **验收人**：daniel 契约校验 + xw 试读
- **验收标准**：未检出帧整帧 NaN 而非 0；左右手判定经人工抽检（vis 图）不互换
- **前置依赖**：① 的 clips；环境 ash_hands ✅
- **当前状态**：环境就绪，等 clips

### ③ panda → 给 ④
- **交付物**：`pose6d/<clip_id>.npz`（object_ids / poses (N,T,4,4) / bboxes (N,T,4)）+ meta.json（注明 depth_source）
- **验收人**：daniel 契约校验 + xw 试读 + 可视化抽检（轴向不飘）
- **验收标准**：同一物体跨帧 id 一致；未检出帧 NaN
- **前置依赖**：① clips；⚠️ GPU（FoundationPose/DepthAnything 推理）；环境 panda_6d ✅
- **当前状态**：环境就绪，等 GPU 挂载 + clips

### ④ xw → 给 ⑤⑥
- **交付物**：`samples/{train,val,test}/*.npz`（hand_seq/obj_seq/label）+ `dataset_index.json` + `vis/<clip_id>.mp4`
- **验收人**：daniel 契约校验 + arc/caron 试读一个 batch
- **验收标准**：split 按 source_video 隔离（防泄漏）；label 三路齐全；obj 特征 13 维顺序正确（pos3+rot6d+area1+vel3）
- **前置依赖**：②③ 输出；interaction_type 标注规则先与 ⑤ 对齐（当前为启发式草案）
- **当前状态**：可用合成 npz 先行开发

### ⑤ arc → 给 ⑥
- **交付物**：`src/encoding/` 的 `build_encoder(cfg)` / `build_decoder(cfg)`，import 即用
- **验收人**：caron（真实训练脚本里跑通 forward/backward）+ daniel（test_encoding.py）
- **验收标准**：支持可变物体数（padding mask）；输出三路 logits；单测前向形状正确
- **前置依赖**：④ 的样本格式（可读契约开发，不等真实数据）
- **当前状态**：可立即开工

### ⑥ caron → 给全组（训练结果）
- **交付物**：`src/mamba/` 训练脚本 + `configs/*.yaml` + `runs/<run>/best.pt` + tensorboard 日志
- **验收人**：daniel（冒烟跑通）+ 全组（实验报告）
- **验收标准**：配置驱动；bf16；checkpoint 含契约版本；附 LSTM 基线开关
- **前置依赖**：④⑤；⚠️ GPU；环境 caron_mamba ✅（GPU 未挂载时 import mamba_ssm 报错属预期）
- **当前状态**：代码可立即写，训练等 GPU

### ⑦ daniel → 给全组（质量闸门）
- **交付物**：`tests/check_contracts.py`（①–④ 产物校验 CLI）+ `tests/test_encoding.py` + 冒烟脚本
- **验收人**：全组使用
- **前置依赖**：契约冻结（已冻结）
- **当前状态**：可立即开工

## 全员对齐要点

1. **只认契约**：上下游对格式有分歧时以 `03_data_pipeline.md` 为准；要改格式先改契约并 @ 上下游（AGENTS.md 铁律 1）
2. **不要等真实数据才写代码**：④⑤⑥⑦ 全部可基于契约 + 合成数据先行开发
3. **两个外部阻塞**：EgoDemo 申请（fish）、GPU 挂载（Daniel 联系 gpufree）——卡住 ①③⑥ 的运行，但不卡代码开发
4. **完工即更新** `05_progress_log.md`
