# 真人手＋物体6D基线：复现SOP与模型接手入口

## 自动化复核入口

2026-10-24新增[300段自动化复核流水线](AUTOMATIC_REVIEW_PIPELINE.md)。在中央仓库按文档运行`python -m src.annotation.review_pipeline --config configs/review_pipeline_300.json --output <独立运行根> --resume`，自动检查原300段与V5全部1238窗口，输出中文报告及可追溯的区间待办。按内容SHA断点续跑；追加`--require-complete`可阻止未完成全标注的数据以退出码0通过交付门槛。读取`summary.json`时区分执行完成、自动检查通过及全标注完成三个状态。

## 当前接手入口：V5交互训练子集

用户已于2026-10-24确认本次V5交互子集复核验收完成，详见最新交付说明中的验收记录。

2026-10-24：先读 [TRAINING_V5_HANDOFF.md](TRAINING_V5_HANDOFF.md)。V5已有5段1238个实际有监督窗口（592/366/280），新`EgoDemoGraphV5Dataset`默认准入加载，真实Mamba有向图前反向通过。该子集仅用于已标注二维物体观测和官方手姿态条件下的交互预测；300段完整几何＋交互联合数据仍未交付。

服务器根`/root/gpufree-data/datasets/ego_hoi/egodemo_training_v5_300_20261024`。检查`training_release.json`、`交付状态_V5.json`与`逐段连续标注缺口_V5.csv`；编解码、样本形状、运行示例和中文网页18791在新接手文档中。不要把V5接口传给旧七类训练入口。V4、V3与下列2026-10-22记录均为历史状态，保持冻结供溯源。

## V4阶段历史接手状态

**2026-10-24前版交付：V4已补300段868255帧数值索引、1054帧实际连续复核和866个Mamba诊断窗口；连续逐帧全标注仍未完成。** 先读`docs/DELIVERABLES.md`当前条目和`.agent/memory/03_data_pipeline.md`末尾契约。历史44帧和120合格目标不替代本次300段任务。

当前V4根`/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v4_dense_300_20261024`。先读`dense_delivery_report.json`、`逐段连续标注缺口.csv`及中文`数据格式与使用说明.json`。连续视觉复核剩867201帧；当前图标签有正负值，旧单客体预测头新增可监督值仍为0。全部新增窗口属于原train，不能用于声称有验证/测试集效果。加载与格式检查：

```python
from src.mamba.annotation_v4 import EgoDemoAnnotationV4Dataset, collate_annotation_v4

base = "/root/gpufree-data/datasets/ego_hoi/egodemo_mamba_v2_300_20261023"
v3 = "/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v3_300_20261023"
v4 = "/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v4_dense_300_20261024"
ds = EgoDemoAnnotationV4Dataset(base, v3, v4, split="train", diagnostic=True,
    windows_path=v4 + "/dense_review_windows/train.jsonl")
batch = collate_annotation_v4([ds[0]])
```

`graph_targets`是V4实际连续复核目标；`v3_graph_targets`另存旧抽样目标。`dense_event_values`只在`dense_event_label_valid=true`处有正负含义，未知不自动当负例。没有whole物体槽位的视觉实体和双客体关系只保留图结构；当前模型尚无图预测头。

在现有`caron_mamba`环境可运行`python -m src.mamba.annotation_v4_smoke --base-root <V2> --v3-root <V3> --dense-root <V4>`复现实际加载与前向；0次优化，不训练。下一批必须新建产物版本并重做连续证据/身份/窗口审计，不覆盖当前冻结review或index。中文只读网页通过SSH转发18790访问`/review_web/index.html`；浏览器测试不会修改审核状态。

以下保留V3复现入口，供冻结来源核对：

服务器标注根`/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v3_300_20261023`，基础数值根`/root/gpufree-data/datasets/ego_hoi/egodemo_mamba_v2_300_20261023`。先核对`delivery_manifest.json`，再读`数据格式与使用说明.json`、`annotation_audit.json`、`annotation_target_coverage.json`；不要把有JSON文件当作有连续标签。

```python
from src.mamba.annotation_v3 import EgoDemoAnnotationV3Dataset, collate_annotation_v3

base = "/root/gpufree-data/datasets/ego_hoi/egodemo_mamba_v2_300_20261023"
annotation = "/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v3_300_20261023"
ds = EgoDemoAnnotationV3Dataset(base, annotation, split="train", diagnostic=True)
batch = collate_annotation_v3([ds[0]])
```

`diagnostic=True`用于复现加载/前向，不放行正式训练。当前默认旧窗口的`silver_event_label_valid`可能全false，这是时间没有命中复核点。要加载补充窗口，请额外传`windows_path=annotation+"/review_windows/train.jsonl"`，划分必须与split参数一致；具体有标签例子见smoke报告的`windows_path`与`window_indices`。实例ID与几何槽位分离，双客体边在`graph_targets`中；未知位置不能设为none或负例。

以现有`/root/gpufree-data/conda/envs/caron_mamba/bin/python`运行`python -m src.mamba.annotation_v3_smoke --base-root <基础根> --annotation-root <标注根> --diagnostic --review-windows`可复现补充窗口上的真实GPU前向，命令不训练。v3默认禁用旧v2弱监督，保留独立助手银标；旧七类`src.mamba.train`不能直接消费本接口。后续修订review应新建版本目录，依次重建entity_annotations、interaction_annotations、annotations/index、review_windows并重新审计；新窗口导出器拒绝覆盖已冻结index，不手工只改状态。

下文保留2026-10-22交接历史：当时发布的是**已有实测证据的44帧真人手＋单物体基线**，不是300段标注、120段数据集完成或完整HO-Cap长序列验收。

- 已验证：DexYCB第三人称固定相机、真人右手拿碗；源帧000030–000073，44帧。源速率30FPS，约1.47秒，明确是短截取，不能冒充整个原始采集。
- 未验证：独立手部/物体精度、米制手物三维融合、真实摄像头采集、长期稳定跟踪、正式未来预测精度。44/44是联合输出覆盖率，不是准确率。
- 已发布入口：`src/pose6d/run_human_rgbd.py`、`src/pose6d/run_rgbd_sequence.py`、`src/hand_pose/online_hands.py`、`configs/human_rgbd_baseline.args`。
- 本次`run_human_rgbd.py`及其测试从44帧实测源码快照发布，不包含之后加入但尚未发布的forecast扩展。不要把本机的未提交变体当成GitHub版本；从干净检出运行下面的测试。
- 历史阶段曾暂停继续扩充120或训练；该暂停不覆盖用户2026-10-23授权的300段标注工作。旧联合候选24、正式合格0/120，候选16/4/4，不能用现有EgoDemo辅助池抵扣。数据治理入口：`docs/HUMAN_DATASET_V1_20261022.md`。

## 1. 结果与精确路径

本地根：`/home/nvidia/algotithm/egodata_process/runs/human_pose_live_20261022/`。
服务器结果根：`/root/gpufree-data/datasets/ego_hoi/human_pose_live_20261022/`。

| 内容 | 相对结果根的路径 |
|---|---|
| 可直接观看的视频 | `human_hands_foundationpose_30fps.mp4` |
| 预览/拼图 | `human_hands_foundationpose_preview.png`、`human_hands_foundationpose_contact_sheet.jpg` |
| 逐帧数值与时间表 | `dexycb_human_lite_stream_3/hand_poses.npz`、`object_poses.npz`、`frames.csv` |
| 完整配置、来源SHA、报告 | `dexycb_human_lite_stream_3/metadata.json`、`run_report.json` |
| 历史独立产物核验 | `independent_validation.json` |
| 历史源码、测试快照 | `source_snapshot/` |
| 环境与原生功能证据 | `environment_freeze.txt`、`runtime_probe.json`、`setup_logs/` |

服务器已准备输入：`/root/gpufree-data/datasets/ego_hoi/human_rgbd_assets_20261022/scenes/dexycb_bowl_20200709_150906_932122060861_f030_073/`。这些原始数据、权重和输出**不在Git仓库**；必须已有授权与资产，不能认为clone即可凭空产生数据。

历史主试验总墙钟15.8278秒，逐帧流阶段3.1031秒；注册帧1,634.77ms。其后43帧平均处理能力41.5236FPS、环路P95 27.2647ms，文件开始读取到写出P95 66.8990ms。MP4的30FPS只是播放速率；这些数字不包含真实摄像头曝光/传输/显示延迟，不保证每帧都满足33.33ms预算。

## 2. 已验证环境与启动前检查

服务器中央源码目录：`/workspace/egodata_process`。生产Python：`/root/gpufree-data/conda/envs/panda_human_live/bin/python`。禁止向base装包或改动其他人的环境。

| 组件 | 已验证版本/约束 |
|---|---|
| 系统与GPU | Linux、NVIDIA L40 46GB、sm89 |
| Python | 3.10.22 |
| Torch / CUDA runtime | 2.6.0+cu124 / 12.4 |
| NumPy / MediaPipe | 1.26.4 / 0.10.21，手部wrapper严格检查MediaPipe版本 |
| OpenCV | opencv-contrib-python 4.11.0.86；不要仅凭缺少opencv-python发行包判断cv2不可用 |
| trimesh / PyYAML | 5.1.1 / 6.0.3 |
| 原生依赖 | nvdiffrast0.4.0、PyTorch3D0.7.9、warp1.18.0、已编译mycpp |
| FoundationPose | `/root/gpufree-data/third_party/FoundationPose_real_20261022`，commit `a1b694b83e633c2cb6115b9063d940a687759392`，实测时跟踪文件无修改 |

权重在 `/root/gpufree-data/datasets/ego_hoi/foundationpose_assets_20261022/weights/` 的 `2024-01-11-20-02-45/model_best.pth` 和 `2023-10-28-18-33-37/model_best.pth`，FoundationPose目录通过软链引用；不得提交或重新分发权重。MediaPipe模型随已验证wheel安装，wrapper记录其SHA。

```bash
ROOT=/root/gpufree-data
export PATH="$ROOT/conda/envs/panda_human_live/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9
export PYOPENGL_PLATFORM=egl MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$ROOT/build_cache/foundationpose_real_20261022"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
nvidia-smi
df -h "$ROOT"
python -c 'import sys, torch, cv2, mediapipe, trimesh; print(sys.version); print(torch.__version__, torch.version.cuda, cv2.__version__, mediapipe.__version__, trimesh.__version__); assert torch.cuda.is_available()'
git -C "$ROOT/third_party/FoundationPose_real_20261022" rev-parse HEAD
```

在新机器上只能新建独立环境，按该FoundationPose版本及目标GPU重新验证CUDA光栅化、KNN和mycpp；不能复制不兼容的`.so`或用CPU/mock声称真实推理。已有服务器环境可直接复用，本SOP不要求重新编译或改base。

## 3. 输入契约与坐标

```text
SCENE/
  rgb/<stem>.png
  depth/<stem>.png
  masks/<first_stem>.png
  cam_K.txt
  mesh/textured_simple.obj
  mesh/<referenced mtl and texture files>
  meta.json
```

- RGB、depth同帧同尺寸、帧名可排序且逐一对应。后续mask可缺失；首帧必须是真实目标分割mask，不能用矩形bbox代替，也不能从参考位姿构造初始化。
- `cam_K.txt`为3×3真实彩色相机K；深度必须与RGB对齐。本例原始深度按0.001换算为米。其他数据集不能直接复制这个scale。
- mesh原点、朝向、尺度保持来源定义，单位为米；不得先中心化、旋转、缩放再把结果冒称原mesh坐标。FoundationPose输出原mesh→camera的4×4矩阵，camera为x右/y下/z前。
- 手部为MediaPipe21点：图像归一化XY和相对Z，不是米制相机坐标。0腕、4/8/12/16/20为五指尖；左右槽位依据是否镜像记录，当前非镜像输入做标签交换。
- 未检测手整手21点与对应置信度为NaN；不能补零/插值制造覆盖。置信度是左右手分类分数，不是关节精度。
- 必须提供有依据的源FPS与真人输入证据。本例CSV时间为输出帧序号/FPS推导的相对时间，**不是设备硬件时间戳**。未知时基的新数据不能通过传30假装满足这个入口。
- 公开参考可在推理完成后单独审计；不得把未来手/物体参考、后续mask或GT位姿接入当前帧推理、初始化、重定位或调参。

## 4. 完整复现命令

在含本次发布代码的干净工作区执行；现有中央工作区有其他未提交功能，交接时先核对`git status`与源码SHA，不要覆盖他人改动。

```bash
cd /workspace/egodata_process
SCENE="$ROOT/datasets/ego_hoi/human_rgbd_assets_20261022/scenes/dexycb_bowl_20200709_150906_932122060861_f030_073"
RUN="$ROOT/datasets/ego_hoi/human_pose_live_20261022/handoff_$(date +%Y%m%d_%H%M%S)"
mapfile -t BASELINE_ARGS < configs/human_rgbd_baseline.args
python -u -m src.pose6d.run_human_rgbd "${BASELINE_ARGS[@]}" \
  --foundationpose-repo "$ROOT/third_party/FoundationPose_real_20261022" \
  --scene-dir "$SCENE" --mesh-file "$SCENE/mesh/textured_simple.obj" \
  --out-dir "$RUN" --human-evidence "$SCENE/meta.json" \
  --fps-source "$SCENE/meta.json; official DexYCB recording rate"
```

命令不使用`--limit-frames`，处理该输入目录全部44帧，但目录本身是明确的短截取。不要将它当作完整原始采集。无桌面的服务器不加`--display`；真实相机接入是另一个尚未验证任务。

默认配置为手lite、首帧注册5次、跟踪1次、Torch/OpenCV各1线程、最多预解码下一帧、按30FPS节奏回放。手部与物体模型仍消费同一个当前帧；禁止提前推理未来帧。

## 5. 输出与可执行质检

输出目录必须不存在。成功后包含 `human_object_overlay.mp4`、`hand_poses.npz`、`object_poses.npz`、`frames.csv`、`metadata.json`、`run_report.json` 和抽样PNG。结果根目录的 `human_hands_foundationpose_30fps.mp4` 是历史交付别名，不是runner直接输出名。

```bash
python - "$RUN" <<'PY'
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np
root = Path(sys.argv[1])
report = json.loads((root / 'run_report.json').read_text())
assert report['status'] == 'success' and report['real_models']
assert report['frames_completed'] == report['frames_requested']
assert report['coverage']['human_hand_frames'] > 0
assert all(value is None for value in report['accuracy'].values())
with np.load(root / 'hand_poses.npz', allow_pickle=False) as hands:
    assert hands['left_hand'].shape == hands['right_hand'].shape == (report['frames_completed'],21,3)
    assert hands['conf'].shape == (report['frames_completed'],2)
with np.load(root / 'object_poses.npz', allow_pickle=False) as objects:
    poses = objects['poses']
    assert poses.shape == (1,report['frames_completed'],4,4)
    assert np.isfinite(poses).all() and (poses[:,:,2,3] > 0).all()
    assert np.allclose(poses[:,:,3], [0,0,0,1], atol=1e-6)
    rotation = poses[:,:,:3,:3]
    assert np.allclose(rotation.swapaxes(-1,-2) @ rotation, np.eye(3), atol=1e-3)
    assert np.allclose(np.linalg.det(rotation), 1, atol=1e-3)
with (root / 'frames.csv').open() as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == report['frames_completed']
assert [int(row['frame_index']) for row in rows] == list(range(len(rows)))
assert np.all(np.diff([float(row['timestamp_ms']) for row in rows]) > 0)
video = cv2.VideoCapture(str(root / 'human_object_overlay.mp4'))
decoded = 0
while video.read()[0]:
    decoded += 1
video.release()
assert decoded == report['frames_completed']
print({'artifact_checks_passed': True, 'frames': decoded, 'accuracy': None})
PY
```

以上检查是结构、完整性和数学合法性，不验证目标跟踪正确。正式质检还要核对所有输入/模型/代码哈希，查看首中末、至少5%有效帧及全部异常帧，记录手左右、遮挡、漂移、框/轴方向、帧序和时间。独立GT及同坐标证据未齐备时不报位置/旋转/手部精度。

代码回归（无需下载权重；模型注入仅用于CPU测试，不能作为真实模型验收）：

```bash
python -m pytest -q tests/test_online_hands.py tests/test_human_rgbd.py \
  tests/test_rgbd_sequence.py tests/test_human_baseline_config.py
```

以干净HEAD导出并只覆盖拟发布的44帧源码/配置/测试，以上四组共147项通过；SOP内的产物检查脚本另对历史44帧真实产物执行。与历史876项全仓测试分开，不把未提交工作区的1,081项测试当作本commit能力。

## 6. 已处理问题与故障恢复

- 真实模型缺依赖、权重缺失或CUDA失败：抛错并写失败报告，无mock回退。先修独立环境/资源路径，不向base安装。
- MediaPipe图创建、process与close必须同线程；runner统一由单worker管理。非镜像左右标签显式交换，不把视角当作镜像依据。
- GPU同步用于准确计时；同时保留输入读取到写出延迟，不能仅报最小GPU耗时。性能优化含lite模型、一次跟踪迭代和有界I/O预取，独立精度取舍仍未量化。
- 输出目录存在即拒绝，不覆盖旧产物；推理异常与清理异常都会阻止success状态。先保留失败目录/日志，再用新run_id从首帧重跑。
- 不能从中间帧假装恢复FoundationPose跟踪状态；没有序列化、验证状态恢复机制。不得靠读该帧参考位姿“恢复”。
- 输入SHA变化、首帧空mask、K/尺寸/深度错误先修输入；不将错误帧偷偷删除或重编号。无人手的序列不能报真人联合交付通过。
- 视频编码失败或无显示器：检查编码器/目录写权限；headless不要`--display`，保存视频不是连接相机的证据。

## 7. 批量生产参数与接手顺序

每个采集至少登记 `dataset,subject_id,acquisition_id,camera_id,scene_dir,mesh_file,source_fps,fps_evidence,human_evidence,expected_frames,split,license`。禁止用相机/切片/滑窗充当独立采集。单L40先串行运行，一个采集/一个目标/一个新输出目录；多目标和多GPU资源分配另验收，不能将单目标FPS外推为多目标FPS。

批量时复用第4节参数数组，但必须按来源覆盖 `--dataset-name`、`--viewpoint`、`--source-fps`、`--depth-scale`及各路径。每次run后独立质检，失败隔离，不加入正式训练manifest。源时间未知的HO-Cap暂不能直接走要求已知FPS的已发布基线。

接手优先级：

1. 从干净GitHub检出跑上述CPU回归，再在现有服务器复现44帧，确认环境与资产路径；不必重复下载。
2. 审查未发布的完整HO-Cap实验，补齐验收器与视觉/时间/单位证据，再决定是否发布完整序列入口。
3. 完整真人序列的工程与质量门禁通过后，才恢复120段规范生产；数量和质量均通过后才训练。不要当前就扩大下载/训练。

### 未发布实验：必须作为WIP处理

本地WIP：`src/pose6d/full_hocap.py`、`configs/full_hocap_v1.json`、`tests/test_full_hocap.py`，均不随本次基线发布；纯CPU辅助函数测试83项通过，不是完整运行验收。

服务器实验根：`/root/gpufree-data/datasets/ego_hoi/full_hocap_20261022/`。`production_01/`已由真实模型处理HO-Cap单采集全部446帧、4物体，`report.json`为complete；**随后验收器因numpy.bool_不能JSON序列化失败，未产生成功的acceptance.json，未完成视觉/几何/独立精度验收**。因此不作为本次“已验证基线”或正式合格数据，不能只看report的complete。

原始报告/日志为 `production_01/report.json`、`logs/production_01.log`、`logs/acceptance_01.log`；视频/数值为 `production_01/overlay.mp4`、`sequence.npz`、`timing.csv`。失败位置在 `audit_output → write_json`，下一模型先为NumPy标量JSON转换补回归测试；不要仅修序列化就宣布质量通过。物理时戳仍未知，hand_camera_surface_m也仅是深度表面代理，不是已验证的解剖关节真值。本次不继续修该实验，不让它阻塞GitHub交接。

2026-10-23接续记录（本地工作区，尚未同步服务器）：已在时间检查生成处将NumPy布尔标量转为Python bool；新增8种合成产物验收/CLI回归，修复前复现相同序列化异常，修复后 `python -m pytest -q tests/test_full_hocap.py` 为91 passed。覆盖报告落盘、完整视频解码、失败返回码、原产物不变与验收拒绝覆盖。首次SSH被网络沙箱拦截；权限恢复后服务器可达，但本机默认密钥认证被拒，待确认有效凭据，真实446帧产物尚未复验。恢复连接后先检查是否已有后续验收，不覆盖历史 `acceptance.json`。本地修复与测试不构成正式数据或独立精度放行。

2026-10-23后续：认证已解决，修复与测试已同步，`panda_human_live` 中91项测试通过；原446帧产物完成17项工程检查，`production_01/acceptance.json` 已生成，原视频/NPZ/CSV及报告不变。每帧至少一只手、四物体均有数值有效位姿；手别输出的5帧变化待核查，完整视觉、源物理时戳与独立精度仍未验收，正式训练门禁保持关闭。本地复验副本与部署备份记录见 `runs/full_hocap_resume_20261023/`；当前用户要求优先研究交互信息标注方案。

### 2026-10-23：手部、物体身份和协作关系可视标注

按用户要求新增 `src/annotation/pose_relations.py`，契约见 `.agent/memory/03_data_pipeline.md` 中 `pose_relation_annotation_v1`。在原始图像上叠加解剖左右手 H-L/H-R、物体稳定 ID/官方名称中文翻译及已有 6D 估计；关系按有向边和原生帧区间保存，支持接触、抓持控制、共同运动、支撑、稳定固定和作用于。协作模式区分单手、双手共同操作、双手分工、交接、工具介导。原估计和已有 VLM 候选不被改写。

本地示例 `runs/pose_relation_annotation_20261023/render_03/` 包含完整446帧 `overlay.mp4`、区间 `annotations.json`、逐帧 `frame_graphs.jsonl`、审核清单 `review.csv`、预览与哈希清单。6段持续关系、5个转换候选、1段单手操作、1段任务语义全部为 `pending`，由助手查看原图提出，不是人工金标准；左右手异常帧363/382/383/396/398需复核。物体名称来自本地HO-Cap官方元数据，桌面仅为环境参照，不声明其6D姿态。时间未知时只用原生帧，30FPS仅用于播放。

服务器资源根 `/root/gpufree-data/datasets/ego_hoi/pose_relation_annotation_20261023/`；`annotations_server.json` 绑定服务器原始manifest与NPZ哈希，`assets/`内字体无需更改环境。使用现有 `panda_human_live` 环境，可在 `/workspace/egodata_process` 运行以下命令；`render_next`必须是新目录：

```bash
ANNOTATION_ROOT=/root/gpufree-data/datasets/ego_hoi/pose_relation_annotation_20261023
POSE_ROOT=/root/gpufree-data/datasets/ego_hoi/full_hocap_20261022
/root/gpufree-data/conda/envs/panda_human_live/bin/python -m src.annotation.pose_relations \
  --manifest "$POSE_ROOT/scenes/subject_7_20231022_192832_105322251564/scene_manifest.json" \
  --sequence "$POSE_ROOT/production_01/sequence.npz" \
  --annotations "$ANNOTATION_ROOT/annotations_server.json" \
  --font "$ANNOTATION_ROOT/assets/NotoSansCJK-Regular.ttc" \
  --output-dir "$ANNOTATION_ROOT/render_next"
```

审核时在新的JSON副本中修改实体关系、帧区间、证据帧和状态，再生成新视频复核；`review.csv`是审核清单，当前没有CSV回写器。只有带审核人和审核时间的human来源才能设为accepted；协作审核通过还要求依赖关系已通过。24项契约测试在本地/服务器通过。此版本是离线标注，不可充当预测模型的历史输入；训练准入仍关闭，未来标签导出需独立处理时间和因果边界。当前可视布局最多4个物体与1个环境参照。

### 2026-10-23：中文网页审核入口

私有网页：https://hand-object-review-zh.justgiant1027.chatgpt.site 。本页基于上述446帧样例，使用原生帧图与原始画面播放视频，手部、物体与关系从当前草稿动态叠加；修改不会残留旧视频中烘焙的关系。界面、实体显示名、关系/事件/协作角色、审核状态和CSV表头采用中文。

填写审核人后，从左侧选择关系/事件/协作/任务或5帧手别异常。左右方向键逐帧、空格播放，起止帧可一键取当前帧，结束帧在界面中包含、导出JSON仍是半开区间。有效编辑自动保存为待审草稿，确认通过需显式点击；修改已通过关系会使依赖协作退回待审。草稿只存当前浏览器，跨设备或交给服务器处理前请导出JSON；CSV是中文审核表。导入会检查采集与源内容SHA，需要确认替换当前浏览器草稿。手别审核另存意见，不暗改NPZ。

构建入口 `src/annotation/review_web.py`；网页源码 `src/annotation/web_assets/`；可在空目录运行 `python -m src.annotation.review_web --manifest <源清单> --sequence <原NPZ> --annotations <候选JSON> --output <空目录>`。本地Site工作目录 `/home/nvidia/algotithm/sites/hoi-review-zh/`，服务器离线包 `/root/gpufree-data/datasets/ego_hoi/pose_relation_annotation_20261023/web_review_01/site.tar.gz`，解包网页在该目录 `unpacked/dist/`，可用静态HTTP服务访问。部署信息与浏览器证据在 `runs/pose_relation_annotation_20261023/web_qa/`，其中 `qa_export.*` 是隔离浏览器的自动化测试记录，不能作为真实审核结果。

10项Node数据审核测试及14项真实浏览器流程检查通过，覆盖保存刷新、播放/逐帧定位、通过条件、关系依赖失效、末帧445边界、异常手别、JSON/中文CSV、异源导入拒绝和手机布局；浏览器导出另通过Python原契约校验。生产初始13条候选均pending，无测试审核记录混入发布数据。源时间、独立姿态质量与正式训练门禁保持原状态。

### 用户整批确认与训练格式核对（2026-10-23）

用户看过网页后明确表示“我看了一遍，大致没问题，可以都通过”。据此13条区间记录与5条手别复核设为accepted，桌面实体候选一并确认；审核人记“用户（本会话确认）”，保留原话、原SHA、应用时间和原候选来源。新版本 `runs/pose_relation_annotation_20261023/approved_01/` 与服务器同名数据目录，含annotations、中文审核CSV、逐帧图和approval_record。原pending版本留存，源姿态不修改，正式训练未启动。私有网页更新为13/13通过；旧浏览器草稿迁移保留内容，确认之后的新修改仍需重新审核。

按服务器现有代码实测：合成输入B=2、W=32、N_max=8，经编码器得到 `(2,32,11,128)`，Mamba处理 `(22,32,128)`，解码输出主体 `(2,3)`、物体 `(2,9)`、旧7类动作 `(2,7)`。L40仅做一次随机初始化前向，未训练/载入checkpoint；证据 `approved_01/mamba_shape_inspection.json`，详细字段在 `training_format_inspection.json`。person token由两手腕及有效标记生成，不是单独的人体姿态；中文名称仅展示，object_ids用于槽位映射。

服务器train.py比本地多label_valid严格校验及loss/metric屏蔽，快照已保存；encoder/decoder/config哈希一致。现有build_samples仍调用启发式label_clip，未读取本次pose_relation_annotation_v1，也没有写出服务器要求的label_valid。正式接入需补标注适配，不能运行旧builder后认为人工标签已用于训练。

当前解码器对实体平均后预测固定物体槽位，交换两个有效物体token后物体分数变化仅1.49e-7，暴露槽位对应信息丢失。课题版建议逐物体打分，并增加有向实体对多标签关系、未来转换事件及协作监督；这是架构建议，尚未修改现有训练实现。离线标注仅作监督，不进入因果历史输入；源时间未知时仍不能把播放30FPS当真实速度或秒级预测依据。

## 8. 数据分发和提交边界

GitHub只发布本SOP、相关契约/进度、44帧基线所需源码/配置/测试；不发布原始数据、预览帧、权重、凭据或EgoDemo记录级标识。EgoDemo原始材料及可定位记录的清单不可再托管；HO-Cap再分发需CC BY 4.0署名和第三方资产白名单。ModelScope `Liuyue9698/ego_pose_dataset`只完成权限预检，尚未上传数据，不能把当前Apache模板当作所有来源数据的许可。详细门禁见 `docs/HUMAN_DATASET_V1_20261022.md`。
