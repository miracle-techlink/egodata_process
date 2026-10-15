# 03 — 数据流水线接口契约

> 这是全项目的**唯一接口事实源**。改格式 = 先改本文件 + 通知上下游（AGENTS.md 铁律 1）。
> 标注【草案】的条目由负责人在首次实现时敲定细节，然后回到本文件更新。

## 统一约定

- **坐标系**：一律相机坐标系；右手系
- **ID 命名**：`clip_id = <原视频名>_<3位段号>`，如 `demo01_007`；全局唯一
- **元信息**：每个产物目录带 `meta.json`（fps、帧数、来源 clip、生成时间、生成环境）
- **fps**：不强制重采样，原始 fps 记入 meta；时序窗口按帧数定义
- 存储根：`/root/gpufree-data/datasets/ego_hoi/`（即个人目录 `data/ego_hoi/`）

## ① 视频切分（fish）

- 输入：原始长视频（EgoDemo 等，`.mp4`/`.webm`）
- 输出：
  ```
  clips/<clip_id>.mp4          # 每段 ~20s（18–22s 可接受）
  clips/clips_index.csv        # clip_id,source_video,start_sec,end_sec,fps,frames
  ```

## ② 手部姿态（ash）

- 输入：`clips/<clip_id>.mp4`
- 输出：
  ```
  hand_pose/<clip_id>.npz
    left_hand  : (T, 21, 3) float32   # MediaPipe 21 关键点顺序，图像归一化坐标 + 相对深度
    right_hand : (T, 21, 3) float32   # 未检出的帧整帧填 NaN
    conf       : (T, 2) float32       # 左右手置信度
  hand_pose/<clip_id>.meta.json
  ```

## ③ 物体 6D pose（panda）

- 输入：`clips/<clip_id>.mp4` + 深度（若可得）+ 目标物体 mask/检测框
- 输出：
  ```
  pose6d/<clip_id>.npz
    object_ids : (N,) str             # 本 clip 跟踪到的物体列表
    poses      : (N, T, 4, 4) float32 # 物体→相机 变换矩阵；未检出帧为 NaN
    bboxes     : (N, T, 4) int32      # xyxy，供可视化与裁剪
  pose6d/<clip_id>.meta.json
  ```
- 【草案】深度来源：若数据无深度通道，先用单目深度估计（DepthAnything）补齐，在 meta 里注明 `depth_source`

## ④ 训练样本构建 + 可视化（xw）

- 输入：②③ 的输出
- 输出：
  ```
  samples/<split>/<sample_id>.npz    # split ∈ {train,val,test}
    hand_seq   : (W, 2, 21, 3)       # 窗口内双手关键点（W=窗口帧数，【草案】默认 32）
    obj_seq    : (W, N_max, 13)      # 每物体: 位置3 + 旋转6D表示6 + bbox面积1 + 速度3
    label      : dict                # 下一帧: subject(左/右手), object_id, interaction_type
  samples/dataset_index.json         # 样本清单 + split 划分（按 source_video 隔离，防泄漏）
  vis/<clip_id>.mp4                  # 关键点 + 物体轴叠加可视化，用于人工抽检
  ```
- 【草案】`interaction_type` 标签体系：{none, reach, grasp, hold, move, place, release}，标注规则与 ⑤ 共同敲定

## ⑤ 编码器 / 解码器（arc）

- 输入：④ 的 `samples/<split>/*.npz`
- 编码器约定【草案】：人/左手/右手/各物体 → token；每 token 特征 = 自身状态 + 相对空间关系（与其他实体的距离、相对速度、contact 标志）
- 解码器输出：下一时刻 `subject / object / interaction_type` 三路 logits
- 模块接口：`src/encoding/` 下提供 `build_encoder(cfg) -> nn.Module`、`build_decoder(cfg) -> nn.Module`，供 ⑥ 直接 import

## ⑥ Mamba 训练（caron）

- 输入：④ samples + ⑤ encoder/decoder 模块
- 输出：`checkpoints/<run_name>/best.pt` + 训练日志（存数据盘 `datasets/ego_hoi/runs/`）
- 预训练：先在公共数据集（ego_pose_dataset / Ego4D 类）上预训练，再迁移机电场景

## ⑦ 集成测试（daniel）

- `tests/check_contracts.py`：校验 ①–④ 任一产物是否符合本契约（shape、dtype、字段、NaN 规则）
- 冒烟流程：任选 1 个 clip 走通 ①→⑥ 全链路
