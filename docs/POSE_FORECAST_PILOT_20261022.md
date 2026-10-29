# 真人手与物体未来姿态预测 — 2026-10-22

## 本轮结论

**工程链路跑通，预测效果尚未通过验收，稳定30 FPS也尚未通过。**

本轮不是只做计划：已经实现并运行“真实MediaPipe + FoundationPose → 历史窗口 → Encoder → LSTM/Mamba → 手部/物体Pose Decoder → 未来0.5/1秒骨架和物体框”，完成6组真实观测伪标签训练、独立采集测试、视频导出以及直接接入逐帧感知循环。

输入全部是真人手DexYCB采集，没有机械臂。仍是第三人称固定相机录制RGB-D的回放，不是EgoDemo第一人称、不是实际摄像头直播，也没有把手部相对Z当相机米制坐标。未来帧的估计结果只作为伪标签/离线对照，不是独立GT。

## 直接查看（本地绝对路径）

- 预测与未来对照：`/home/nvidia/algotithm/egodata_process/runs/pose_forecast_20261022/future_pose_comparison.mp4`
- 实际逐帧感知＋预测叠加：`/home/nvidia/algotithm/egodata_process/runs/pose_forecast_20261022/future_pose_online.mp4`
- 三帧对照图：`/home/nvidia/algotithm/egodata_process/runs/pose_forecast_20261022/future_pose_preview.jpg`
- 恒速对照视频：`runs/pose_forecast_20261022/videos/constant_velocity/forecast_comparison.mp4`
- Mamba视频：`runs/pose_forecast_20261022/videos/mamba_seed17/forecast_comparison.mp4`
- 六组模型、全部数值：`runs/pose_forecast_20261022/benchmark/report.json`
- 预先指定展示权重：`runs/pose_forecast_20261022/benchmark/lstm_seed17/best.pt`
- 独立源码QA：`runs/pose_forecast_20261022/qa_review.json`；独立产物重算：`artifact_validation.json`。

主视频都是74帧、源30 FPS、约2.47秒。前31帧收集历史，随后43帧输出未来姿态；源片段不足以支持最后部分预测的未来对照，因此右侧明确显示 unavailable。编码成30 FPS视频不等于实际处理速度达到30 FPS。

对照视频左侧只能访问当前图像和截至当前时刻的姿态：青色实线为当前估计，橙色/紫色虚线为+0.5/+1秒预测。右侧是明确标记的离线未来画面，青色为未来估计，不是GT；紫色仍为之前生成的预测，不因看到未来重新修正。预先指定LSTM seed17展示，没有根据测试误差挑好看的模型。无需纹理贴图，也没有生成未来RGB画面。

## 数据与隔离

| 采集ID | 划分 | 帧数 | 目标 |
|---|---|---:|---|
| 20200709_150906 | train | 74 | 碗 |
| 20200709_145132 | train | 74 | 肉罐头 |
| 20200709_150654 | train | 72 | 碗 |
| 20200709_150326 | train | 74 | 清洁剂瓶 |
| 20200709_153322 | val | 68 | 泡沫砖 |
| 20200709_152240 | test | 74 | 剪刀 |

共436帧，6个独立采集，4/1/1划分在推理前固定，同一被试，不声称跨被试泛化。原本希望取得更长片段；实际已检查的公开片段只有68–74帧，未拼接、重复或重采样制造长序列。实际训练样本48、验证7、1秒测试13；0.5秒测试可评估28个观察时刻。滑窗高度相关，不能把这些窗口当独立试验。

测试采集原机位把人手和剪刀裁在画面边缘，因此在训练/测试评分前，仅按输入画面可见性改选同一采集的 `932122060857`；采集ID和test划分不变，不增加一个“独立样本”。其余机位均 `932122060861`，K与RGB/depth/seg逐路匹配；机位变化同时意味着测试存在视角分布变化。候选截图和时间记录在 `assets_evidence/test_camera_decision.json`。

只读取官方标签中的首帧 `seg.npy` 初始化FoundationPose；没有将手/物体GT位姿读入感知模型。原始标签隔离。资产920个场景文件及30个来源文件本地/服务器SHA全部匹配。手部检出缺失保持NaN，不插值补“正确骨架”。

## 模型与训练

- 历史32帧，预测间隔按时间戳0.5/1.0秒；只使用观察窗口，速度由每一对相邻时间戳计算。训练与在线均在窗口首帧将速度置0，禁止窗口外信息混入。
- 输入：`hand_seq (B,32,2,21,3)`、`obj_seq (B,32,1,13)`、`obj_mask (B,1)`。手为图像归一化XY/相对Z；物体为原始mesh→camera的米制位置和旋转。
- HOIEncoder生成实体token，维度32、单层关系attention；单层LSTM或Mamba按实体建模时间；独立PoseDecoder输出手关节残差、物体平移和旋转残差。现有三头意图decoder不改动。
- 输出：`hand_future (B,2,2,21,3)`、`object_future (B,2,1,4,4)`。无当前观测不输出虚构实体。旋转6D解码到SO(3)，不直接回归任意4×4矩阵。
- 固定单位缩放，不用val/test拟合归一化；手/物体位置SmoothL1、旋转矩阵差损失。mixed模式不计算跨坐标系距离/contact。
- LSTM/Mamba各seeds17/29/43、各20 epoch；48训练窗口、batch16；只有val损失选择best.pt，测试评分全部推迟到6组训练结束之后。manifest完整读取不等于训练使用test，报告对这两者分别记录。
- 零初始化残差相当于持位epoch0，也参与val选优；若被选中必须标INIT而不是已训练。本轮6组最佳epoch分别LSTM 3/4/2，Mamba 3/4/4，均有真实优化步。
- GPU运行环境为既有 `daniel_test`（Torch2.6.0+cu124、真实Mamba CUDA），6组训练及测试共约12.63秒。没有安装/改动base或现有环境。

## 真实测试结果（误差越小越好）

数值是相对未来MediaPipe/FoundationPose估计的**伪标签一致性误差**，不是独立物理准确率。旋转是未消除物体对称性的原始SO(3)误差，参考估计本身也可能有误。剪刀使用单刚体mesh，不预测铰接角。

| 方法 | 0.5秒手像素误差 | 0.5秒物体位置mm | 0.5秒旋转° | 1秒手像素误差 | 1秒物体位置mm | 1秒旋转° |
|---|---:|---:|---:|---:|---:|---:|
| 持位 | 67.35 | 116.52 | 26.68 | 129.01 | 224.63 | 45.10 |
| 恒速 | 52.07 | 87.99 | 40.28 | 104.20 | 207.08 | 88.26 |
| LSTM，3 seed均值 | 75.72 | 105.45 | 32.21 | 136.76 | 214.94 | 46.66 |
| Mamba，3 seed均值 | 80.27 | 101.88 | 31.68 | 141.05 | 213.65 | 47.91 |

两种学习模型物体平移比持位略好，但手部误差更大，旋转也没有联合优势；恒速平移较好但旋转更差。**因此不能说学习模型已经可靠预测，不能说Mamba优于LSTM/基线，也不放行生产效果。** 未根据这些test结果继续调参。全部方法在可评估观察时刻有联合输出：0.5秒28/28、1秒13/13，只表示覆盖，不是准确率。

## 在线性能与限制

真实循环在每帧先完成手部和物体估计，再预测，再画入这一帧。数据预取只解码下一帧，不执行未来帧模型推理。固定展示LSTM seed17，CPU预测、GPU FoundationPose。

| 运行 | 预测激活帧 | 激活期间处理FPS | 环路P95 ms | 超过33.33ms帧 |
|---|---:|---:|---:|---:|
| 初版 | 43 | 28.05 | 45.84 | 34 |
| 批量虚线绘图、减少PNG保存 | 43 | 29.60 | 38.55 | 22 |

优化只改绘图和产物保存，不改预测权重/数据/预测间隔。优化版如果把尚未开始预测的历史积累帧也算入，会得到32.42 FPS，但不能用它冒充预测激活时速度。首次注册约1.68秒，另需32帧历史，模型加载另计。**尚不满足稳定30 FPS，更不是摄像头capture-to-display或硬实时验收。** 两次运行均保留，不删除较慢结果。

## 测试、审计与代码

- 新模块：`src/forecasting/{data,baselines,model,train,online,evaluate,render,run,benchmark}.py`。
- 在线接入口：`src/pose6d/run_human_rgbd.py --forecast-method learned --forecast-checkpoint ...`。
- 本地和服务器全仓均984 tests + 26 subtests通过（分别7.57/15.23秒）；独立18项风险回归通过，覆盖未来泄漏、时间错位、分组别名泄漏、NaN、SO(3)、val选优、随机权重拒绝、epoch0标记和视频因果性。训练用daniel_test缺trimesh，不修改该环境，完整回归改用已有panda_human_live，首次环境缺依赖日志保留。
- 独立产物检查104/104通过：6个权重的训练/验证记录、哈希与有限性，8种方法两时域误差重算，5段视频均完整解码74帧，在线43帧激活耗时重新计算。
- 初轮失败日志保留。训练结束后修正了源码来源路径中的 `tests/../` 归一问题；不改变模型计算或训练权重，训练当时源码保存在 `source_snapshot/training_forecasting/`。

## 复现

服务器根目录 `ROOT=/root/gpufree-data/datasets/ego_hoi/pose_forecast_20261022`，中央仓库 `/workspace/egodata_process`。

训练/对照（输出目录必须是新的）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /root/gpufree-data/conda/envs/daniel_test/bin/python -m src.forecasting.benchmark \
 --manifest "$ROOT/forecast_manifest.json" --out-dir "$ROOT/benchmark_rerun" \
 --models lstm mamba --seeds 17 29 43 --epochs 20 --device cuda
```

预测对照视频可用 `python -m src.forecasting.run --help`；实时循环在上一轮RGB-D命令上追加：

```bash
--forecast-method learned --forecast-window 32 --forecast-horizons 0.5 1.0 \
--forecast-checkpoint "$ROOT/benchmark/lstm_seed17/best.pt" --forecast-device cpu
```

只看基线可以使用 `--forecast-method constant_velocity`，不需要权重。新相机/第一人称不能直接忽略运动与标定，必须另做适配。

## 下一阶段门槛

先取得更长、更多独立真人采集及可靠参考，并覆盖不同手/物体/视角；重新预注册划分，不能反复调整本轮test。先核验感知参考噪声与手部坐标，再比较多时域损失和模型。只有手/物体指标稳定优于基线、误差在预先约定阈值内且完整循环时延达标后，才称预测验收成功。本轮不通过贴图或视频美化掩盖误差，不启动正式HOI分类训练。
