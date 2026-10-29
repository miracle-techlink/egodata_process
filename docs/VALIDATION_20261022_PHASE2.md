# 2026-10-22 第二轮验证与下一阶段计划

## 本轮交付和验收口径

本轮不把全 none 合成准确率当模型效果，也不从官方语义事件猜测三头 HOI 标签。正式 HOI 仍缺物体轨迹、坐标投影核验和人工复核标签。

| 工作流 | 负责人 | 本轮验收 | 结果载体 |
|---|---|---|---|
| 模型边界回归 | Agent A | mask/NaN、样本隔离、因果性、重载、忽略标签的回归与根因修复 | `tests/test_model_robustness.py` |
| 真实资产质检 | Agent B | 10 段视频完整解码；时间、掩码、事件和几何统计可溯源 | `src/dataset/audit_egodemo.py`、`real_data_audit.json` |
| 真实辅助预测 | Agent C | UUID 6/2/2 隔离；持位/恒速/LSTM/Mamba；3 seed；val 选模型 | `src/mamba/trajectory_benchmark.py`、`trajectory/` |
| 集成和坐标证据 | 主 Agent | 全量 CPU 回归、L40 四组合训练/重载、双目刚体约定验证 | `tests/run_gpu_acceptance.py`、`camera_rig.json` |

服务器产物根目录：`/root/gpufree-data/datasets/ego_hoi/validation_20261022_phase2/`。所有实验均使用新输出目录，源数据不改写；代码不自动 commit/push。

## 辅助预测协议

- 输入官方世界系双手腕历史（各 16 帧），预测 1 秒后双手位移；以 timestamp 定位目标，不能把 manifest 平均 fps 当原始时间。
- 对每只手减去最后观察手腕，观察/目标全部有效才保留。三个 seed 固定为 17、29、43；归一化仅拟合 train。
- UUID 排序后固定 6/2/2，验证集挑选 checkpoint，测试集不调参。该 pilot 仅有 10 段且任务不同，属于小样本跨任务诊断。
- 指标为原始坐标单位的欧氏终点误差；分别报告样本加权和 episode 等权、左右手和三 seed。单位未经独立标定，不转换成厘米。
- 只复用项目的 LSTM/Mamba 时序骨干和相同回归头；不是三头 HOI 模型，不输出 HOI accuracy。

## 下一阶段：按门槛推进，而非直接扩大训练

1. **先清质量疑点并做坐标校准（②③⑤⑦）**：复核本轮 `quality_review.csv` 的 42 条手腕跳变记录，记录保留/无效的证据而非按阈值直接删除。在至少 2 个有头部转动的 episode 上核验内参、畸变、左右眼、光轴以及手部重投影；报告误差分布和带帧号的失败样例。仅刚体约定成立不足以开启 `coordinate_mode=camera`。
2. **物体小闭环（③④）**：先选 2 段（整理电缆、调整货架标签），各完成稳定物体 ID、可见性/遮挡记录、人工复核框/轨迹；没有可信深度/mesh/6D 时保留缺失，不用零矩阵或 mock 冒充。
3. **标签核验（④⑦）**：对 46 条原文事件做区间拆分，记录主体、物体 ID、7 类动作或 unknown、证据帧与复核人；允许双手共同操作，不能强行把原文一条事件映射成一个单手标签。先冻结冲突/unknown 处理契约，再产正式样本。
4. **正式训练门槛（④⑤⑥⑦）**：采集 UUID 无泄漏、有效目标与物体轨迹齐全、类别/缺失分布可审计、至少 5% 样本投影视检、非全 none；满足后跑 LSTM/Mamba 相同划分与预算的 0.5/1/2 秒三 seed 对照，报告每头 macro-F1、联合 accuracy 和按 episode 的不确定性。
5. **扩展数据（①④）**：先通过上述两段小闭环，再扩展已匹配的 82 个标注 UUID；不为填满 GPU 提前全量下载/训练。

## 实测结果

### 正确性与 GPU 验收

- 本地 CPU（Torch 2.14）及服务器 `daniel_test`（Torch 2.6）均 **412 passed + 26 subtests passed**；本地耗时 3.97 秒、服务器 8.30 秒。
- 新增模型边界测试首次复现 **13 failed / 10 passed**；修复后该文件 **26 passed**。修复包括：运算前清除 invalid 手/物数据、decoder 在池化前清除 padding、物体 padding logits 使用 `-inf`、交叉熵只计算非 ignored 行。避免 `NaN × 0`、极大 padding、低于 `-10000` 的 none logit 和 ignored NaN logits 造成错误输出或梯度。
- L40 上 LSTM/Mamba × FP32/BF16 四组各训练 **100 step**，使用含全部 7 类、每类 2 个样本的合成压力 batch；四组梯度有限、loss 下降超过 99%，checkpoint 重载及因果前缀最大差异均 **0**。
- 最终 loss：LSTM FP32 **0.003747**、BF16 **0.002953**；Mamba FP32 **0.003317**、BF16 **0.003296**。合成 batch 记忆准确率不是泛化效果。验收产物以 `gpu_acceptance_fixed/` 为准，较早 `gpu_acceptance/` 是本轮边界修复前运行，留作溯源而不混用。

### 真实轨迹实验

固定 10 UUID 分组得到 train **1,715** / val **383** / test **1,404** 个样本。过滤 61 个无未来目标、18 个观察无效、8 个目标无效窗口，无时间间隙过滤；未依据测试表现更换划分或调参。所有六组训练完成后才加载 test，每组使用验证集选择的 checkpoint。

测试集误差（越小越好，双手欧氏终点误差平均，**原始坐标单位，不是厘米，也不是 HOI accuracy**）：

| 方法 | Episode 等权误差 | 样本加权误差 | 相对持位改进（episode 等权） |
|---|---:|---:|---:|
| 保持最后位置 | 0.059997 | 0.052526 | — |
| 末两帧恒速 | 0.096332 | 0.095917 | -60.56% |
| LSTM，3 seed 均值 | 0.058974 ± 0.000491 | 0.052118 | +1.71% |
| Mamba，3 seed 均值 | 0.059843 ± 0.000153 | 0.052516 | +0.26% |

`±` 为三个 seed 的样本标准差，不是数据集置信区间。两种模型参数量分别 8,934 / 10,470，共用 32 维、单层、16 帧输入和相同训练预算（20 epochs）；没有声称参数量完全相等。六组训练与评估共约 **11.51 秒**，六个 checkpoint 的验证集重载差异均为 **0**。

- 最优 epoch：LSTM seeds 17/29/43 为 **1/2/1**；Mamba 为 **1/1/1**。后续 epoch 未改善验证选择结果。
- 测试只有整理电缆（1,181 样本）和货架标签（223 样本）两个采集，滑窗高度相关；应优先看 episode 等权，不能把 1,404 个窗口当独立试验样本。
- **结论**：实现和真实时序链路可复现，但学习模型相对持位基线的收益很小，本 pilot 不支持 Mamba 优势或正式 HOI 有效性结论。下一步优先补齐观测、标签与数据覆盖，不扩大模型或用测试集继续调参。

### 相机刚体证据

10 段、18,079 帧均支持 `q` 表示 camera→world：`R(q)^T (p_right - p_left)` 的刚体残差 RMS 为 **7.89e-9～4.48e-7** 原始单位，优于反方向假设约 **2.12 万～66.07 万倍**。双目基线均值为 **0.063831～0.064018** 原始单位，各段有明显转头运动，静止退化案例已由单测覆盖。

这只是双目刚体约定证据；像素投影、内参、畸变、光轴和物理尺度仍未独立验证。**没有转换源数据，也没有开启正式模型的 camera 坐标模式。**

### 完整真实资产审计

- **10/10 段、18,079/18,079 帧完整解码通过，0 errors / 12 warnings**。不再只是容器头部帧数检查；逐帧 PTS 相对原生 timestamp 的最大误差约 **7.45 微秒**，事件时间表与原文半开区间一致。
- 原文共 **46 个事件**，覆盖 18,015 帧；9 段合计 **64 帧无语义事件覆盖**，没有事件重叠。不把未覆盖帧当 `none`，更不能把原文语义自动映射成七类 HOI。
- 另 3 项警告对应 **42 条手腕速度异常记录，涉及 40 个唯一帧、2 个 episode**：货架标签左手 2 条，电缆左手 7 条、右手 33 条。阈值是按段中位数/MAD 的诊断，不是物理错误结论。
- `quality_review.csv` 已列出 UUID、手别、帧号、时间、速度、阈值和配套视频路径，全部 `pending`；源数据未删除/修补。两段恰好是本诊断的测试采集，**未据此事后过滤测试集或重报更优指标**。
- 审计还核验来源 SHA-256、坏帧 mask、原始/暂存手坐标一致性、四元数有限性与模长。当前暂存层仍不是正式 HOI 训练集。

### 复现命令

在服务器中央仓库 `/workspace/egodata_process` 执行；输出路径必须是新的：

```bash
ROOT=/root/gpufree-data/datasets/ego_hoi
PY=/root/gpufree-data/conda/envs/daniel_test/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -m pytest -q \
  tests src/pose6d/tests src/hand_pose/test_extract_hands.py
"$PY" -m src.dataset.audit_egodemo \
  --manifest "$ROOT/annotated_pilot_20261022/selection.json" \
  --dataset-root "$ROOT/raw/annotated_pilot_20261022" \
  --staging-root "$ROOT/annotated_pilot_20261022/staging" \
  --output "$ROOT/NEW_real_data_audit.json"
"$PY" -m src.dataset.check_camera_rig \
  --staging-root "$ROOT/annotated_pilot_20261022/staging" \
  --output "$ROOT/NEW_camera_rig.json"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -m src.mamba.trajectory_benchmark \
  --staging-root "$ROOT/annotated_pilot_20261022/staging" \
  --output-dir "$ROOT/NEW_trajectory"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" tests/run_gpu_acceptance.py \
  --output-dir "$ROOT/NEW_gpu_acceptance"
```

`trajectory/` 保留 config、固定 UUID 划分、train-only normalization、逐样本 provenance、过滤计数、六份 history/checkpoint/test predictions/metrics。`real_data_audit.json` 和 `camera_rig.json` 为完整逐段证据；`quality_review.csv` 是下一轮人工复核入口。
