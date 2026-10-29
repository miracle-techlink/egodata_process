# 6D 位置 / 姿态专项验收 — 2026-10-22

## 最终判定

**现有交付不通过真实 6D 验收：0/8 份合格。** 不是目录为空：确实有 8 对 NPZ/JSON，合计 5,214 个视频帧、每帧 2 个物体，共 10,428 个物体帧。但全部是明确标记为 `pose_estimator=mock`、`depth_source=none` 的模拟输出。

| 验收层级 | 实测结果 | 结论 |
|---|---|---|
| 旧版 shape/dtype/字段契约 | 8/8 文件通过 | 只说明文件接口合规 |
| 有限刚体矩阵 | 10,428/10,428 物体帧 | 只说明代数结构合规 |
| 真实 6D 来源与必要证据 | 0/8 文件通过 | 全部拒绝 |
| 实测位置准确率 | 无独立真值 | 未评估，null |
| 实测旋转准确率 | 无独立真值 | 未评估，null |
| 位置与旋转联合成功率 | 无真实预测/独立真值 | 未评估，null |

**0/8 是本次交付验收通过率，不是模型准确率。** 准确率的分母尚未建立，不能写成 0% 或 100%。

## 位置与旋转查到了什么

1. **Z 全部固定为 1**：10,428 个物体帧一致，来源是 mock 默认值，不是传感器或米制深度估计。不把此数值宣传为测得距离 1 米。
2. **旋转全部是单位阵**：10,428 个 `R = I`，相对单位阵的角度和跨帧旋转变化均为 0 度；不是估计出了真实物体转动。
3. **X/Y 来自检测框中心反投影**：重新从 bbox 与文件内参计算，和现有平移的最大绝对差为 `1.4702479067540963e-08` 原始数值单位，符合 float32 舍入。这印证了当前输出是占位构造。
4. **代码来源一致**：`src/pose6d/extract_pose6d.py` 的 `_MockObjectTracker.estimate` 明确使用单位旋转、框中心和固定/输入深度；这些文件的后端来源本身也写明 mock。不能仅因真实物体静止、R=I 或 Z 不变就判假，本次拒绝依据是明确来源与实现证据。

全量文件范围：

| clip | 视频帧 | 物体帧 | 严格门禁 |
|---|---:|---:|---|
| demo01_000 | 602 | 1,204 | not_accepted |
| demo01_001 | 702 | 1,404 | not_accepted |
| demo01_002 | 802 | 1,604 | not_accepted |
| demo02_000 | 602 | 1,204 | not_accepted |
| demo02_001 | 702 | 1,404 | not_accepted |
| demo03_000 | 602 | 1,204 | not_accepted |
| demo03_001 | 702 | 1,404 | not_accepted |
| demo03_002 | 500 | 1,000 | not_accepted |

此表来自 `/home/daniel/pipeline_run/data/pose6d/`，不是前轮货架/电缆两个真实标注 episode。两段真实 episode 的预期 `pose6d_real/<UUID>.npz/.meta.json` 再次检查仍不存在，涉及 1,194 + 5,970 = 7,164 帧；未将两类分母混在一起。

## 运行条件与代码缺口

服务器 L40 与 `panda_6d` 的 CUDA 张量实测可用，GPU 本身不是本次首要阻塞。

- 实际导入项目 `FoundationPoseBackend` 失败，首个异常为 `No module named 'imageio'`。独立依赖探针另发现 `kornia`、`pytorch3d`、`nvdiffrast` 等缺失；不是只根据环境配置猜测。
- `mycpp` 只能导入命名空间，没有所需 `cluster_poses` 或编译扩展；不能把 `import mycpp` 不报错等同于可执行后端。
- FoundationPose 的 scorer/refiner 预期 weights/config 文件缺失；DepthAnything 预期 checkpoint 目录缺失。盘上找到的其他训练 checkpoint 不等于 6D 模型权重。
- 去掉探针自身临时文件限制后，`torchvision`、DepthAnything metric/relative 源码均成功导入；DepthAnything 仍未加载权重或进行推理。其首轮临时目录异常是审计器造成的，不算服务器缺陷。
- 已检范围未找到目标物体 mesh 或真实 mask；名称中包含 depth 的参考说明图片/历史 JSON 不是逐帧米制深度。范围、跳过路径与完整异常保存在运行报告，不能据此声称整个服务器所有位置都不存在资产。
- 当前项目封装尚未接入 model-free mesh 重建；注册时 `_FoundationPoseObjectTracker._rect_mask` 使用 bbox 矩形，不是已经验证的物体分割 mask。即使环境装好，仍需补齐可验收的目标几何和 mask 来源。

运行探针只进行导入、GPU 小张量与文件盘点；没有安装包、下载权重、重建 mesh 或执行真实 FoundationPose 推理。受限只读探针若因临时文件限制失败，不计作环境依赖缺失，需看补充复核结果。

官方运行要求的参考来源为 https://github.com/NVlabs/FoundationPose ：scorer/refiner 权重、测试数据及本机扩展需分别准备；本轮以服务器实测为准，不把仓库已 clone 视为后端已可运行。

## 验收中修复的问题

发现旧 `--strict` 仅在部分后端初始化时生效，逐物体创建失败、逐帧估计异常、必需深度缺失仍能继续写出全 NaN 的产物，造成“命令成功”误导。

- strict 模式现在遇到上述情况立即抛错，不生成新的 NPZ/JSON；非 strict 保留原降级行为，但 tracker 创建失败增加 `pose_backend_create_failed` 标记。
- 新 mock 输出增加 `mock_pose` 标记和明确的模拟结果警告，不改写历史 8 份证据。
- 视频迭代器提前关闭时释放解码句柄，覆盖严格失败路径的资源清理。
- 用服务器修复前源码复现：选中的 8 项验收回归 **6 failed / 2 passed**。修复后该文件全部 **13 passed**，完整本地/服务器回归均 **730 passed + 26 subtests passed**，分别耗时 7.08 / 10.16 秒，结果另存日志。

这修复的是失败报告与来源标记，**没有把 mock 变成真实 6D**。

## 独立数值复核

另一个 agent 用 SciPy 随机旋转及独立标量公式复核 SO(3)/平移门禁：137 个场景、1,632 项断言通过，26,778 次矩阵分类无参考分歧；包含 4,096 个随机旋转及 float32/float64、反射、非正交、错误最后行、非正深度、NaN/Inf 与缺失情形。

这些都是合成数值验证，不是模型真实准确率。门禁允许有限正深度的数值矩阵，不等于物理距离合理；有坏物体时即使其他物体覆盖率为 100%，也必须读取 `status` 与 `issues`，不能仅看 coverage。

## 交付物

本地：`runs/pose6d_acceptance_20261022/`。

服务器：`/root/gpufree-data/datasets/ego_hoi/pose6d_acceptance_20261022/`。

- `pose_inventory.json`：8 份现有输出的来源、位置/旋转统计、严格门禁与摘要。
- `object_frames.csv`：10,428 行物体帧；真实精度通过字段保持空值。
- `mock_pose_diagnostic.png`：示例 clip 的 X/Y/Z 和旋转曲线，显著标注 MOCK/REJECTED；不是实物 6D 可视化。
- `source_inventory.json` 与 `inputs/`：原始证据副本，16 个输入文件 SHA-256 与服务器一致。
- `runtime_readiness.json`：命令、导入异常、CUDA 实测和资产盘点。
- `numerical_gate_review.json`：独立数值交叉验证，准确率字段为 null。
- `contract_*.log`、`strict_regression_before_fix.log` 与 `pytest_*.log`：格式检查、红灯复现和修复后测试证据。

## 放行前必须完成

1. ③负责人先完成真实 FoundationPose 后端及权重准备，用已知刚性物体、真实米制深度/分割/mesh 跑一个短序列，启用 strict。
2. ④负责人确认该序列的相机标定、目标尺寸/单位、物体坐标定义和时间对齐，输出真实物体轴与手部同系图，而不是矩形框中心或相机轴。
3. ⑦负责人先冻结独立参考、位置/角度误差阈值、对称物体处理与可见性分母，再报告位置/旋转误差及联合成功率。

以上未完成前，保持 6D 验收拒绝，不恢复正式手—物 HOI 训练。
