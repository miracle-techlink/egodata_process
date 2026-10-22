# 真人手 + FoundationPose 同帧在线提取 — 2026-10-22

## 先纠正场景

上一轮 `mustard0` 是 FoundationPose 的机械臂演示，**不是 EgoDemo，也不符合用户本轮“只要真人手”的要求**。旧结果只保留为后端环境测试证据，不作为真人手联合交付。

本轮视频已换为真实的人手持物：DexYCB 的真人右手拿红碗，**第三人称固定相机，不冒称 EgoDemo 第一人称**。使用原序列 `20200709_150906`、相机 `932122060861`、连续源帧 `000030–000073`，共 44 帧，源速率 30 FPS，视频长度约 1.47 秒。这是短序列验证，不是长时间生产验收。

本轮已经完成：真实 MediaPipe 手部推理 + 真实 FoundationPose 物体注册/跟踪，在同一逐帧循环内完成后立即叠加、写入视频。不是复制数据集手部标注，也不是读完视频再拼接历史预测。

## 直接查看

本地根目录：`/home/nvidia/algotithm/egodata_process/runs/human_pose_live_20261022/`。

- **主视频**：`human_hands_foundationpose_30fps.mp4`。
- **单帧效果**：`human_hands_foundationpose_preview.png`。
- **六帧拼图**：`human_hands_foundationpose_contact_sheet.jpg`。
- **主视频原始输出**：`dexycb_human_lite_stream_3/`，包含 `hand_poses.npz`、`object_poses.npz`、`frames.csv`、`metadata.json`、`run_report.json`。
- **输入与来源**：`assets_manifest.json`、`assets_evidence/`、`egodemo_input_audit.json`。
- **验收与测试**：`independent_validation.json`、`setup_logs/pytest_full.log`。
- **入口**：`src/pose6d/run_human_rgbd.py`；手部实时封装：`src/hand_pose/online_hands.py`。

视频中蓝色是右手 21 点骨架，黄色是目标碗的三维模型包围框投影，红/绿/蓝轴是物体 XYZ 坐标轴投影。画面实时打印左右手检出、手部/物体耗时与输入来源。

服务器运行根为 `/root/gpufree-data/datasets/ego_hoi/human_pose_live_20261022/`；资产根为 `/root/gpufree-data/datasets/ego_hoi/human_rgbd_assets_20261022/`。

## 实测结果与“实时”的边界

所有最终复测中，44 帧均有右手预测和有效物体矩阵，左手保持 NaN；视频完整解码 44 帧。**44/44 是联合输出覆盖率，不是手部或物体准确率。** 独立精度未评估。

最终配置：MediaPipe 0.10.21 的 lite 模型、FoundationPose 注册迭代 5 次/跟踪迭代 1 次、Torch/OpenCV 各 1 线程、手部 CPU 与物体 GPU 同帧并行、有界预取下一帧的 RGB-D 解码。只预取输入，不提前推理后续帧；模型输出按顺序等待齐全后再绘制。

| 最终试验 | 稳态生产 FPS | 单帧环路 P95 | 文件读入开始→写出 P95 | 超过 33.33 ms 的环路帧 | 本轮预算判定 |
|---|---:|---:|---:|---:|---|
| `dexycb_human_lite_stream_1` | 41.409 | 27.056 ms | 55.490 ms | 2/43 | 通过 |
| `dexycb_human_lite_stream_2` | 41.471 | 27.967 ms | 61.821 ms | 2/43 | 通过 |
| `dexycb_human_lite_stream_3`，按 30 FPS 节奏回放 | 41.524 | 27.265 ms | 66.899 ms | 2/43 | 通过 |

解释：

- 三次是**同一 44 帧复测**，不是 132 个独立新样本。每轮稳态统计为除注册帧之外的 43 帧。
- 生产 FPS 是实际逐帧处理耗时计算的处理能力，不是 MP4 文件头的 30 FPS。第三轮另有回放等待，41.524 FPS 不表示实际展示 41.524 帧/秒。
- 预算规则为平均处理能力 ≥30 FPS 且环路 P95 ≤33.33 ms；**不要求每一帧都在预算内**。每轮仍有 2 个慢帧，最大环路耗时约 67.4–70.2 ms，因此不宣称无抖动或硬实时。
- 文件读入→输出包含预取队列等待，不能用较小的环路 P95 冒充端到端延迟；它也不是摄像头曝光→屏幕显示延迟。
- 首帧注册约 1.63–1.66 秒，模型加载/预检还有额外开销，不计入稳态 FPS。
- 首个预取跟踪帧会跨注册阶段排队，审计发现读源→输出最大约 1.72 秒；不能用稳态 P95 冒充整段最大延迟。
- **本次输入是录制 RGB-D 的在线回放，没有连接真实摄像头，也没有在服务器上验证桌面窗口。** `--display` 支持有桌面的环境显示每帧；本次服务器无桌面，交付的是同帧生成的 MP4 和数值结果。

性能调试不是只挑最快一次：未预取的 full-hand/两次物体跟踪迭代约 17.08 FPS；一次迭代约 25.51 FPS；full-hand 加预取后平均 33.55–34.54 FPS，但其中两轮 P95 超过预算。最终 lite+预取的三轮均达到上述短序列预算。所有日志保留。
跟踪迭代从 2 减为 1、手部模型从 full 改为 lite 属于性能取舍，**尚未用独立真值量化精度变化**。

## 输入、坐标与真实性

- 输入为官方公开来源的 RGB-D、真实相机内参、真实首帧目标分割及匹配的 YCB `024_bowl` 原始模型。不是把芥末瓶模型换个名字，也不是用矩形框冒充 mask。
- 深度为 uint16 毫米，乘 `0.001` 得米；mesh 未任意缩放或重心化。首帧目标 mask 8,414 像素均有有效深度。
- 首帧 mask 来自配套分割，后续 FoundationPose 只读取同帧 RGB-D/K 跟踪；原始位姿/手部 GT 文件与输入 scene 隔离，不用于初始化或渲染骨架。
- 物体输出是原始 mesh→camera 的 4×4 变换，保存为 `(1,44,4,4)`。
- 手部输出是模型预测的图像归一化 XY 与相对 Z，保存为左右手各 `(44,21,3)`、`conf (44,2)`；非镜像输入按官方自拍约定交换左右标签。`conf` 是左右手分类分数，不是关节精度。
- **同帧视频叠加不等于米制三维手—物融合**。没有将 MediaPipe hand-centered/world landmarks 冒充相机坐标关节。
- 碗近轴对称，绕对称轴的朝向有固有歧义；不能拿这个例子证明所有旋转角已经精确测出。位置、旋转、手部、联合准确率均保持 null。
- 源图、深度、mask、K、mesh、手部 TFLite、FoundationPose 权重及源码均有哈希记录。CPU 注入测试显式标为 `test_only`，不混入真实模型产物。

## 代码与环境验证

- 新增 `OnlineHands` 的线程归属、时间戳、左右手、缺失值、错误传播及绘图测试，以及联合 runner 的同帧因果顺序、预取、失败报告、清理异常、节奏和视频一致性测试。
- 本地新增模块 79 项测试通过；服务器完整回归 **876 passed + 26 subtests passed**，耗时 12.54 秒。
- 独立产物审计 **182/182 项检查通过**：最终三组和三组 baseline 共264帧完整解码，各组44帧NPZ/CSV/视频一致，107份记录文件的702次哈希引用匹配，48份产物在审计期间未变。独立重算保留baseline超时与启动排队，不将一致性检查当精度验收。
- 新建 `/root/gpufree-data/conda/envs/panda_human_live`，没有修改 base、原 `panda_6d` 或上一轮 `panda_fp_real`。使用 Torch 2.6.0+cu124、NumPy 1.26.4、MediaPipe 0.10.21；原生光栅化/KNN/mycpp 功能再次验证通过。

## 为什么还不能直接宣布 EgoDemo 完成

已经逐帧抽看现有 10 段 EgoDemo，确实是人手；机械臂来自上轮另一个公开样例，不是这些视频。
对一个已获访问授权的刚性目标候选另做了只读审计；受上游许可限制，不公开其记录标识、路径或可定位记录的清单：

1. 有本地左目视频和双目标定，不能说完全没有相机标定。
2. 右目视频及 MCAP 仍是 Git/LFS 指针，没有本地实际 payload；MCAP 内是否包含深度尚未读取确认。
3. metadata 声明深度设备，但未找到实际深度帧；已有处理视频 938 帧和 raw 标定 1107 帧尚未建立精确映射。
4. 相机外参推得的基线数值缺明确单位证据，raw metadata 内部 UUID 与目录 UUID 也不一致，不能直接猜测“米”并混配。
5. 未取得吹风机真实 mesh/尺寸和首帧目标 mask。

因此本轮明确交付的是 **DexYCB 真人手联合视频与可运行在线管线**，不是假装已经为 EgoDemo 补出缺失深度和物体位姿。接实际第一人称任务的下一步是补齐上述真实输入和时间/单位证据；不恢复正式 HOI 训练。

## 复现（服务器）

```bash
ROOT=/root/gpufree-data
export PATH="$ROOT/conda/envs/panda_human_live/bin:$PATH"
export MPLBACKEND=Agg PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WARP_CACHE_PATH="$ROOT/build_cache/foundationpose_real_20261022/warp"
SCENE="$ROOT/datasets/ego_hoi/human_rgbd_assets_20261022/scenes/dexycb_bowl_20200709_150906_932122060861_f030_073"
cd /workspace/egodata_process
python -m src.pose6d.run_human_rgbd \
  --foundationpose-repo "$ROOT/third_party/FoundationPose_real_20261022" \
  --scene-dir "$SCENE" --mesh-file "$SCENE/mesh/textured_simple.obj" \
  --out-dir "$ROOT/datasets/ego_hoi/human_pose_live_20261022/human_rerun_01" \
  --dataset-name DexYCB --viewpoint third_person_fixed_camera \
  --human-evidence "$SCENE/meta.json" \
  --source-fps 30 --fps-source "$SCENE/meta.json; official paper section 2.1" \
  --depth-scale 0.001 --est-refine-iter 5 --track-refine-iter 1 \
  --torch-threads 1 --opencv-threads 1 --hand-model-complexity 0 \
  --save-every 1000 --prefetch-input --pace-replay
```

输出目录必须是新的。桌面环境可以另加 `--display`，按 Esc 会记录未完成而不是报成功；当前无桌面服务器不要加。要接真实相机，需要提供实际 RGB-D 输入设备/流及相机标定、目标 mesh 和首帧 mask，不能把普通 RGB 摄像头自动等价为这些输入。
