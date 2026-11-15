# 真实 FoundationPose 链路跑通 — 2026-10-22

## 本轮结论

**真实模型的注册 → 连续跟踪 → NPZ/逐帧表 → 物体轴/三维场景图 → 视频导出已跑通。**
不是 mock，也不是只导入库：在 L40 上使用官方 scorer/refiner 权重，对官方实拍 `mustard0` RGB-D 样例完整推理 737 帧。

| 项目 | 实测结果 | 含义 |
|---|---|---|
| 官方脚本烟测 | 首帧注册 + 后续 2 帧跟踪成功 | 原生模型确实执行 |
| 项目新 runner 烟测 | 3/3 帧，NPZ/图/视频导出成功 | 新入口真实联调 |
| 完整序列 | 737/737 帧返回有效刚体位姿，0 中途失败 | **运行覆盖率 100%，不是准确率** |
| 推理时间 | 34.005 秒 | 包含逐帧读取与 GPU 同步，不含模型初始化/预检/导出 |
| runner 总时间 | 82.113 秒 | 包含预检、模型初始化、推理、PNG/MP4 导出及 MP4 完整解码；不含安装下载及额外三维图 |
| 导出视频 | 640×480，完整解码 737 帧 | 10 FPS 仅是预览回放速率，不宣称原始采样率 |
| 全仓回归 | 797 passed + 26 subtests passed，13.04 秒 | 代码行为验证，不是模型精度 |
| 独立产物复核 | 41 项检查通过，1,489 份来源文件哈希一致 | 不复用 runner 验证函数，不重新推理或读取 GT |
| 位置 / 旋转 / 联合准确率 | 均为 `null` | 当前独立 6D 参考不可用，未计算 |
| EgoDemo 手—物融合 | 仍未通过 | 本轮是公开实物样例，不是项目两段视频 |

运行报告的 `status=success` 只表示这条推理/导出链路成功，不能替代业务数据的严格验收。
旧 8 份 mock 产物仍然不通过，未被改写；历史审计结论见 `POSE6D_ACCEPTANCE_20261022.md`。

## 这次补齐了什么

- 新增独立环境 `/root/gpufree-data/conda/envs/panda_fp_real`，从 `daniel_test` 克隆后仅在新环境安装依赖；没有降级原 `panda_6d` 或修改 base。
- Torch 2.6.0+cu124、torchvision 0.21.0；PyTorch3D 0.7.9、nvdiffrast 0.4.0 和 `mycpp` 原生扩展实际构建成功，目标 GPU 架构 sm89。
- CUDA 光栅化实测覆盖 338 个像素，PyTorch3D CUDA KNN 与 `mycpp.cluster_poses` 功能验证通过；不是把空命名空间导入成功当作运行成功。
- 官方 refiner `2023-10-28-18-33-37`、scorer `2024-01-11-20-02-45` 权重与配置已下载；两权重 `weights_only=True` 加载成功。
- 官方 `mustard0` 包安全解压，737 张 RGB 全有同名深度；深度目录另有 595 张非 RGB 配对帧，不按两个列表位置错配。
- 首帧使用真实分割 mask（3,252 像素，其中 2,647 有有效深度），不是检测框矩形；后续只用 RGB-D 跟踪，不输入后续 mask 或 GT。
- 深度原始 uint16 毫米，乘 0.001 转米；保留原始 mesh 坐标、实际相机标定和原始 object→camera 返回值，不将内部中心化 `pose_last` 当作输出。

代码入口：`src/pose6d/run_rgbd_sequence.py`。输出目录必须是新目录；输入尺寸、帧匹配、有效深度、刚体矩阵与视频完整性失败即报错，无 mock/identity 回退。输入文件、模型权重、mesh、标定、runner 和上游实现均记录 SHA-256。

上游工作副本：`/root/gpufree-data/third_party/FoundationPose_real_20261022`，commit `a1b694b83e633c2cb6115b9063d940a687759392`，跟踪文件无改动；权重通过软链指向资产目录。
PyTorch3D commit `204fe5831b3a5bbe079f09d72afb4515178f8892`，nvdiffrast commit `253ac4fcea7de5f396371124af597e6cc957bfae`。
Eigen/Boost program-options 的 Ubuntu 包只解压到数据盘隔离 sysroot；对 Eigen CMake 导出路径做了可重定位修正，没有系统级覆盖安装。

## 可直接查看的交付

本地根：`runs/foundationpose_real_20261022/`。

服务器运行根：`/root/gpufree-data/datasets/ego_hoi/foundationpose_real_20261022/`。

服务器资产根：`/root/gpufree-data/datasets/ego_hoi/foundationpose_assets_20261022/`。

| 相对运行根路径 | 内容 |
|---|---|
| `mustard0_full/preview.mp4` | 完整 737 帧物体框/XYZ 轴叠加视频 |
| `mustard0_full/scene_3d_0000.png` | 首帧 RGB + 米制点云/物体 mesh + 目标局部三维图 |
| `mustard0_full/scene_3d_0368.png` | 中间帧：机器人持物转动，估计坐标轴与点云同系 |
| `mustard0_full/scene_3d_0736.png` | 最后一帧同系三维图 |
| `mustard0_full/pose_trajectory.png` | 平移及相对首帧旋转变化，不是误差曲线 |
| `mustard0_full/poses.npz` | `object_ids (1,)`、`poses (1,737,4,4)`、`bboxes (1,737,4)` |
| `mustard0_full/poses.meta.json` | 真实后端/权重/帧/深度/标定/mesh 来源及哈希 |
| `mustard0_full/trajectory.csv` | 737 行物体矩阵与 bbox 来源，精度字段留空 |
| `mustard0_full/run_report.json` | 完整运行状态、帧数、耗时、视频完整性与未评估精度 |
| `mustard0_full/overlays/` | 服务器保留全部 737 张原始分辨率叠加 PNG，本地用完整视频和三张图查看 |
| `runtime_probe.json`、`environment_freeze.txt`、`build_provenance.txt` | 功能探针、完整环境、源码和构建溯源 |
| `setup_logs/` | 下载外的安装、编译、烟测、正式运行及回归测试日志 |
| `assets_manifest.json`、`assets_evidence/` | 官方下载 URL、大小、哈希、帧配对、单位与参考调查 |
| `independent_validation.json` | 另一个 agent 对产物矩阵/CSV/视频/来源的交叉检查 |
| `evaluation_plan.json` | 首次推理前冻结的评估范围、5 cm / 5° 工程阈值与禁止 GT 泄漏规则 |

三张三维图使用真实深度反投影；紫色是预测位姿变换后的原始 mesh，场景/相机轴/物体轴处于同一个相机坐标系。**没有添加不存在的手部关键点，也不是 EgoDemo 手—物融合图。**
人工抽看了首帧叠加、中间及末尾三维图，目标框/轴随瓶子运动；这只是抽检，不替代逐帧独立位姿真值。

737 帧的估计平移范围为 X `[-0.45707, 0.00016]` 米、Y `[-0.15575, 0.12016]` 米、Z `[0.51900, 0.80402]` 米；相对首帧最大旋转约 `149.27°`。
这些量说明输出不是此前的 `R=I, Z=1` 占位序列，**不说明其误差已经达标**。
独立检查中，737 帧均为有限、正 Z、合法齐次刚体矩阵，旋转正交性最大元素误差约 `3.47e-6`，没有单位旋转帧；CSV、NPZ、737 张 PNG、完整 MP4 帧序一致。

## 为什么本轮仍不报准确率

官方演示 ZIP 中没有 `annotated_poses`。还对官方原始 `mustard0.tar.gz` 做了有界流式调查：前 69,091,054 字节可读到相同 SHA 的 `cam_K.txt`、`gt_mask` 和后续巨大 rosbag，但尚未取得独立 6D 参考。没有把 GT mask、预测自身或与首帧的差异冒充姿态真值。

推理前声明的 5 cm / 5° 只是本轮工程阈值，不是官方评测指标或双方已确认的业务标准。由于缺独立参考，本轮命令没有启用参考评估，两类误差与联合准确率均保持 `null`，不存在“100% 精度验收通过”的结论。

## 复现命令（现有服务器环境）

```bash
ROOT=/root/gpufree-data
ENV="$ROOT/conda/envs/panda_fp_real"
ASSETS="$ROOT/datasets/ego_hoi/foundationpose_assets_20261022"
export PATH="$ENV/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
export PYOPENGL_PLATFORM=egl MPLBACKEND=Agg
export XDG_CACHE_HOME="$ROOT/build_cache/foundationpose_real_20261022"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
cd /workspace/egodata_process
python -m src.pose6d.run_rgbd_sequence \
  --foundationpose-repo "$ROOT/third_party/FoundationPose_real_20261022" \
  --scene-dir "$ASSETS/demo_data/mustard0" \
  --mesh-file "$ASSETS/demo_data/mustard0/mesh/textured_simple.obj" \
  --out-dir "$ROOT/datasets/ego_hoi/foundationpose_real_20261022/mustard0_rerun_01" \
  --depth-scale 0.001 --est-refine-iter 5 --track-refine-iter 2
```

加 `--limit-frames 3` 可做烟测。不要复用已有输出目录。
取得同帧同原始 mesh 坐标、单位为米的独立参考后，才可显式启用 `--reference-dir`、`--reference-frame-confirmed` 和两个误差阈值；禁止根据 GT 重设初始化、对齐姿态或挑选帧。CPU 测试的注入估计器输出明确标记 `test_only`，不得用作真实模型证据。

## 下一关

真实后端不可运行这一阻塞已解除。下一关不是继续修环境，而是为实际 EgoDemo 刚性目标补齐同采集的米制深度、真实分割、几何模型/尺寸和独立参考，再对齐手部与相机坐标。
当前电缆包含非刚性形变，货架标签近似平面，不能把它们不加论证地当作刚性 CAD 目标，也不能直接把 mustard 瓶子的结果混入项目两段数据。正式 HOI 训练保持暂停。
