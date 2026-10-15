# 03 — 数据流水线接口契约

> 这是全项目的**唯一接口事实源**。改格式 = 先改本文件 + 通知上下游（AGENTS.md 铁律 1）。
> 标注【草案】的条目由负责人在首次实现时敲定细节，然后回到本文件更新。
>
> **契约版本：2026-10-15（首轮冻结）** —— ⑥ 的 checkpoint 以此版本号记录。

## 统一约定

- **坐标系**：一律相机坐标系；右手系
- **ID 命名**：`clip_id = <原视频名>_<3位段号>`，如 `demo01_007`；**段号 0 基**；全局唯一
- **元信息**：每个产物目录带 `meta.json`（fps、帧数、来源 clip、生成时间、生成环境）
- **fps**：不强制重采样，原始 fps 记入 meta；时序窗口按帧数定义
- 存储根：`/root/gpufree-data/datasets/ego_hoi/`（即个人目录 `data/ego_hoi/`）

## ① 视频切分（fish）

- 输入：原始长视频（EgoDemo 等，`.mp4`/`.webm`）
- 输出：
  ```
  clips/<clip_id>.mp4          # 每段 ~20s（18–22s 可接受）
  clips/clips_index.csv        # 固定 6 列: clip_id,source_video,start_sec,end_sec,fps,frames
  clips/meta.json              # 含 dropped_residuals（被丢弃残段的记录，不进 csv）
  ```
- csv 各列以 ffprobe 实测值为准（`end_sec = start_sec + 实测时长`，`frames` = 实测帧数）

## ② 手部姿态（ash）

- 输入：`clips/<clip_id>.mp4`
- 输出：
  ```
  hand_pose/<clip_id>.npz
    left_hand  : (T, 21, 3) float32   # MediaPipe 21 关键点顺序，图像归一化坐标 + 相对深度
    right_hand : (T, 21, 3) float32   # 未检出的帧整帧填 NaN（低置信帧同样置 NaN，阈值默认 0.5）
    conf       : (T, 2) float32       # 左右手置信度；未检出帧同样 NaN（conf NaN ⟺ 该手未检出）
  hand_pose/<clip_id>.meta.json      # 含 width/height（④ 需要）+ 检出率统计
  ```
- 左右手判定：按 mediapipe handedness 标签原样写入（ego 视角与镜像自拍几何一致），人工抽检确认

## ③ 物体 6D pose（panda）

- 输入：`clips/<clip_id>.mp4` + 深度（若可得）+ 目标物体 mask/检测框
- 输出：
  ```
  pose6d/<clip_id>.npz
    object_ids : (N,) str             # obj_000... 按首帧出现先后→首帧面积降序
    poses      : (N, T, 4, 4) float32 # 物体→相机 变换矩阵；未检出帧为 NaN
    bboxes     : (N, T, 4) int32      # xyxy；未检出帧填 -1（int32 存不了 NaN）
  pose6d/<clip_id>.meta.json          # 含 depth_source 与实际生效的 detector/pose_estimator
  ```
- 【草案】深度来源：若数据无深度通道，先用单目深度估计（DepthAnything）补齐，在 meta 里注明 `depth_source`
- 【草案】相机内参 K 未在输入契约里：当前由 CLI 传入或启发式估计（meta 记 `quality_flags: heuristic_intrinsics`），待数据集标定信息明确后回写

## ④ 训练样本构建 + 可视化（xw）

- 输入：②③ 的输出
- 输出：
  ```
  samples/<split>/<sample_id>.npz    # split ∈ {train,val,test}
    hand_seq   : (W, 2, 21, 3) float32 # 窗口内双手关键点（W=窗口帧数，默认 32）
    obj_seq    : (W, N_max, 13) float32# 每物体: 位置3 + 旋转6D表示6 + bbox面积1(像素²) + 速度3(m/s, 帧差×fps, t=0为0)
    obj_mask   : (N_max,) float32      # 真实物体槽=1，padding=0
    obj_valid  : (W, N_max) float32    # 该帧该物体 pose 存在=1（pose 缺失帧 obj_seq 整行 NaN）
    hand_valid : (W, 2) float32        # 该帧该手检出=1（上游 NaN 原样保留进 hand_seq）
    object_ids : (N_max,) str          # 槽位→物体 id；padding 槽为 ""
    label_subject       : int64        # 下一帧: 0=left, 1=right, 2=none
    label_object_slot   : int64        # 槽位索引；无交互 = -1
    label_object_id     : str          # 物体 id；无交互 = ""
    label_interaction   : int64        # 见下 7 类索引
    label_interaction_name : str
    clip_id / source_video / start_frame / fps : 溯源字段
  samples/dataset_index.json         # {"train": [...], "val": [...], "test": [...]}，条目含 sample_id+source_video
  samples/meta.json
  vis/<clip_id>.mp4                  # 关键点 + 物体轴叠加可视化，用于人工抽检
  ```
- split 按 source_video 隔离（`source_video = clip_id.rsplit("_",1)[0]`），防泄漏
- 【冻结】类别索引：`subject` 0=left/1=right/2=none；`interaction_type` 0=none, 1=reach, 2=grasp, 3=hold, 4=move, 5=place, 6=release
- 【草案】`interaction_type` 启发式规则（`src/dataset/label_rules.py`，待与 ⑤ 对齐后冻结）：contact/reach 在**归一化图像空间**判定（腕点 vs 外扩 bbox），hold/move/place/release 用物体速度阈值 + 接触状态转移
- ⚠️ **已知问题（坐标系不一致）**：② 手部为图像归一化坐标 + 相对深度（非米制），③ 物体为相机系米制——两个空间不可直接比距离。当前 contact 判定与 ⑤ 的关系特征各自在局部空间进行；后续方案：手部关键点结合深度与内参升到相机系（依赖 ③ 内参落实）

## ⑤ 编码器 / 解码器（arc）

- 输入：④ 的 `samples/<split>/*.npz`
- 【冻结】实体 token 顺序：`[person, left_hand, right_hand, obj_0 .. obj_{N_max-1}]`，共 `3+N_max` token/帧
- 【冻结】接口（`src/encoding/`）：
  ```python
  enc = build_encoder(cfg)  # cfg: dict/dataclass，键见 src/encoding/encoder.py
  tokens = enc(hand_seq, obj_seq, obj_mask=None, gaze_seq=None)  # (B,W,3+N_max,d_model)
  out = dec(tokens, obj_mask)  # {"subject": (B,3), "object": (B,N_max+1), "interaction_type": (B,7)}
  ```
- NaN 处理在 encoder 入口（置 0 + observed 标志位）；padding 物体的 object logits 置 -inf；object 第 N_max 类 = none
- 时序维度保留（不做池化），时序建模是 ⑥ 的事；gaze 先验接口预留（`use_gaze` 开关）

## ⑥ Mamba 训练（caron）

- 输入：④ samples + ⑤ encoder/decoder 模块
- 输出：`<runs_root>/<run_name>/best.pt` + `last.pt` + 训练日志（runs_root 走配置，存数据盘 `datasets/ego_hoi/runs/`）
- checkpoint 必含：model / optimizer / epoch / config / **contract_version**（= 本文件顶部版本日期）
- 预训练：先在公共数据集（ego_pose_dataset / Ego4D 类）上预训练，再迁移机电场景
- 必须保留 `--baseline lstm` 对照开关（不依赖 mamba_ssm，可在无 GPU 环境跑数据链路冒烟）

## ⑦ 集成测试（daniel）

- `tests/check_contracts.py`：校验 ①–④ 任一产物是否符合本契约（shape、dtype、字段、NaN 规则、类别取值域、split 泄漏）
- `tests/make_fixtures.py`：合成合规/违规数据，供校验器自测与 ④⑤⑥ 联调
- 冒烟流程：任选 1 个 clip 走通 ①→⑥ 全链路
