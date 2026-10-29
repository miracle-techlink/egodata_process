# 03 — 数据流水线接口契约

## 2026-10-23 增补：300段第二轮补全（egodemo_refine_v1）

- 用户明确要求继续补齐稳定物体ID、交互边界及剩余几何；独立输出到egodemo_refine_300_20261023，沿用同一300采集冻结选择和原生时间轴，保留第一轮全部产物与负面质量判定。
- 用经视觉核对的目标实例种子、多帧实例分割与稠密双目几何改进覆盖。实体ID只能在有实例对应证据的连续追踪/重识别范围稳定；跨遮挡或同类物体互换不按名称强行合并。局部参考系保持单物体一致，重新初始化必须记录参考系对齐变换或显式分段。
- 刚体6D必须来自实际3D对应、PnP或点云注册；非刚体记录三维表面/形变，铰接物分部件。纯手部位姿复制、bbox中心配单位旋转、无证据插值不计为补齐。非适用、遮挡、不可辨识旋转与方法失败分别编码，不把非刚体算成刚体6D失败。
- 交互层保留关系/转换/协作/操作，记录主体、稳定目标ID、半开原生帧区间、直接证据及边界不确定范围。几何距离、共同运动和原文语义各自保留来源，模型推断、助手视觉复核与用户人工确认分别记录；不得通过改审核状态冒充补齐。
- 验收同时统计目标实例覆盖、连续性、实际姿态/形变有效帧、身份冲突、边界缺项与可训练监督掩码。数值或文件结构合规不能替代语义和姿态质量；独立精度无参考时仍为null。
- 第二轮目录：`dense_seeds`为经双目校正、左右一致性筛选后的三维参考点；`native_tracks`为同一参考系双向追踪的原生帧轨迹；`joint_annotations/<uuid>/objects.npz`为关联后的`object_ids(N),frame_index(T),poses(N,T,4,4),pose_valid(N,T)`。参考系关联需同时刻表面重合及相对变换稳定证据，冲突帧置NaN。`proposal.json`仅为自动准备的中文审核提案，不能计入人工通过数。
- 审核实体可以保存`seed_polygons:[{frame_index,points_native,source}]`，使用原生像素坐标。此编辑只定义新的待审物体表面种子；不直接改写原始跟踪或变更训练有效性。改动物体轨迹集合后，旧参考系对齐信息必须失效并重算。

## 2026-10-23 增补：300段联合补全包（egodemo_joint_completion_v1）

- 用户授权补全300段的手部姿态、物体6D与交互事件。派生输出独立存放，保留原冻结选择、原视频、时间戳、原手部槽位和坏帧掩码；不继承先前单个HO-Cap样例的人工审核。
- `hands.npz` 保存 `timestamps(T),frame_index(T),hands_world(T,2,21,3),hands_camera(T,2,21,3),hand_uv(T,2,21,2),hand_valid(T,2),joint_in_image(T,2,21),joint_rotation_world_wxyz(T,2,21,4),joint_rotation_valid(T,2,21),camera_to_world(T,4,4),K(3,3)`；左右顺序固定left/right，位置单位沿用来源声明的米。缺测NaN，旋转有效性与位置有效性分开，不插值填造有效帧。
- 内参取同一UUID、同一标注版的episode parquet，核对length和head_left视频尺寸；畸变非零拒绝当前针孔分支。世界→相机依照已有fusion_geometry约定。保留源SHA、投影图内覆盖、四元数模长和手腕速度异常；这些不是精度评估。
- `interactions.json` 独立区分 `source_semantic_intervals,operation_candidates,relations,transitions,cooperations`。操作词表有中文名称、来源原文和命中词；一条原事件允许多种操作，不把整个区间每帧当成所有操作同时发生。操作起止尚未细化时 `temporal_precision=source_interval_only`，主体和物体ID未知保留null，`label_valid=false`；不能据文字推造接触、左右手或物体身份。六类旧标签及旧模型输出编号保持不动。
- 物体6D必须使用真实估计后端并记录所用深度、标定、目标mask、物体参考系和尺度来源。无mesh分支可研究同采集双目重建及刚性点对应的SE(3)估计，但需要独立验收，不能把bbox中心+单位旋转记为6D。非刚体显式记 `rigid_pose_applicable=false` 或局部刚性部件；没有6D输出时只登记缺项，不生成伪造的pose文件。
- `summary.json` 分列手部导出、语义候选、实跑物体6D、视觉审核和人工通过数量。离线标注不作为因果模型输入，质量未通过的部分保持训练禁用。视频/逐条元信息/派生记录只保存在私有研究工作区。
- 用户进一步确认保留300段并按刚体/非刚体分别标注。刚性参考点试验采用同采集左右目、真实K/相机外参三角化与PnP，原点为初始可见表面中位点、轴为初始相机轴，明确非CAD中心/语义规范轴。`pose_valid` 仅表示几何估计通过，独立精度保持null。非刚体输出跟踪点/形变观测而非整物体刚体矩阵。
- 文本模型仅生成中文翻译及有逐字原文依据的物体名称；不采用其自报接触/动作/材料判断。候选操作由可追踪原文词组支持，材料规则只给出待审类别。物体检测与SAM分割产生 `object_observation_candidates`，跨帧身份关联和语义类别均需保留候选状态；首次观测编号不自动等于已经确认的稳定物体ID。
- 中文草稿修复可剥离JSON数组之后的模型赘文、删除不属于原文连续子串的物体提及；仅用本批已验证原文词组词典做精确子串补提取，记录 `repair_audit`，保留原失败输出。此修复只恢复结构合法的候选，不改变审核状态或声称语义正确。
- 稠密跟踪候选按 `tracklet` 保存原生帧号、时间戳、左右目表面点UV、相机/世界系三维点及有效性掩码。连续点轨迹ID只在该tracklet内稳定；跨遮挡/重新初始化的物体ID合并待审。刚体候选需通过三维点刚性拟合与非退化检查；非刚体始终保留点轨迹，可另外报告局部表面参考系但不可称整物体6D。`nearest_hand_surface_m` 使用同帧米制手部与可见表面点，不将近距离自动接受为接触/抓持。源事件中点初始化前的帧标记未处理，不用后见初始化数据冒充因果模型输入。
- 批量几何输出使用 `dense_tracklets_v4`；v1–v3试验保留但不进入汇总。种子排除手部/腕部采集设备和整堆泛称；杯/罐/瓶等容器在表面归属未核实前只保留点与局部表面估计。整物体6D还要求连续至少5帧刚性拟合有效。此门槛是质量筛选，不是独立精度证明。
- `visual_quality_overrides.json`保存助手视觉抽查的负面质量判定及证据帧；被拦截轨迹不显示可用整物体轴，原始候选与NPZ保留。抽查通过不了解为人工批准；汇总分别列原始候选6D物体帧与剔除已知失败后的候选帧，不把tracklet帧数总和当作唯一视频帧覆盖率。
- 同一稳定物体ID下的不同tracklet有各自的局部参考系；身份关联不等于参考系已经对齐，不得直接拼接成连续物体6D或训练输入。正式训练前仍需跨tracklet刚体参考系对齐与姿态质量验证。
- 私有中文审核服务仅绑定127.0.0.1，经SSH转发访问。`review_drafts/<uuid>.json`保存人工编辑的实体（稳定ID、中文名、刚性类别、角色、对应tracklet列表）与交互（层、类型、手别、目标ID、原生半开帧区间、备注）。保存仅为pending草稿，不自动接受或放行训练；带来源selection SHA及revision防止错序列/并发覆盖。原视频、自动候选和旧草稿历史保留。网页上的结束帧为包含该帧，写文件时加1。

> 这是全项目的**唯一接口事实源**。改格式 = 先改本文件 + 通知上下游（AGENTS.md 铁律 1）。
> 标注【草案】的条目由负责人在首次实现时敲定细节，然后回到本文件更新。
>
> **契约版本：2026-10-22** —— 保留 2026-10-15 的张量形状与类别编号；新增预测间隔、采集级隔离和坐标安全规则。⑥ 的 checkpoint 以配置中的版本号记录。

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

## 2026-10-22 增补：正确性与真实标注接入

- **预测间隔**：④ 新增 `--horizon-seconds`（默认不指定，兼容旧版下一帧）。指定时以 `ceil(seconds * fps)`、至少一帧计算间隔；目标帧为 `start + window - 1 + horizon_frames`。样本与 index 增加 `target_frame`、`horizon_frames`、`horizon_seconds`（实际帧间隔 / fps）。观察窗口不得包含目标帧。
- **采集级隔离**：④ 新增 `split_group`，优先用上游元信息的 `source_episode_uuid`，其次提取 clip/source_video 中完整 UUID，没有 UUID 时回退旧 source_video。一个采集的所有机位、切片、数据格式必须落在同一 split；`source_video` 保留原溯源含义。index 与样本均记录 `split_group`；⑦ 在该字段存在时校验跨 split 泄漏。
- **关系坐标**：⑤ 增加配置 `coordinate_mode`，默认 `mixed`，保持旧版手部图像归一化 / 物体相机米制输入，但禁止计算跨这两个空间的距离、速度差和接触特征。同空间手部关系与物体关系保留。只有调用方已经确认手和物体均为同一米制相机系时，才可显式配置 `camera`。不可通过将未知 6D 位姿填零冒充真实结果。
- **时序轴**：⑥ 输入 `(B,W,E,D)` 必须先转为 `(B,E,W,D)` 再合并为 `(B*E,W,D)`；输出必须按逆变换恢复，禁止直接 reshape 混合实体与时间。
- **评估**：保留旧 `acc_macro` 作为三路 accuracy 的均值以兼容日志，但它不是类别级 macro-F1。新增各头 macro-F1、联合三路 accuracy；正式结果同时报告类别分布。单类别/全 none 冒烟不能作为真实模型效果。
- **EgoDemo 标注暂存层**：使用 `source_episode_uuid` 匹配 `EgoProStandard` / `EgoProStandard-body`，优先下载该版本配套视频、parquet 和元信息。Raw 与标注版不得直接按帧号对齐。保留原始世界系手部/相机位姿、时间戳、坏帧标记和带时间区间的原文事件描述；世界→相机变换必须明确旋转约定后才能接入 `camera` 模式。
- **标签边界**：官方 `subtask_annotations` 是语义事件，不是七类 HOI、左右主体、物体槽位或物体 6D 真值。导入工具只输出可追溯的标注暂存产物与待审核事件，不自动伪造这四类标签；正式三头 samples 需完成物体轨迹与人工核验标签后构建。

本轮接口涉及 ④ xw、⑤ Arc、⑥ Caron、⑦ Daniel；统一由集成负责人同步实现与回归测试。

## 2026-10-22 增补：第二轮验证协议（非正式 HOI 评估）

- **真实资产审计**：读取暂存 `sequence.npz` / `meta.json` / `events.json` 与 selection 指向的配套视频，独立输出 JSON 质检报告；视频必须逐帧解码，核对解码帧数与时间轴。检查时间递增、坏帧掩码、事件覆盖/重叠、四元数模长与手腕运动异常分布。几何统计仅作诊断，不自动认证相机旋转方向，不改写原数据或生成标签。
- **辅助轨迹预测**：仅用官方 world 手腕坐标（关节 0）构建 16 帧观察、1.0 秒未来目标的诊断样本；用原始 timestamp 搜索第一个不早于目标时刻的帧，拒绝过大时间间隙。观察和目标两手均须有效。输入为两手相对各自最后观测手腕的历史位移，预测两手未来位移，模型看不到目标帧或未来统计。
- **隔离与对照**：UUID 排序后固定 train/val/test=6/2/2（10 段 pilot），记录精确 UUID、过滤计数、实际时间间隔和样本清单。三个训练 seed；持位与末两帧恒速基线，加同一输入/回归头的现有 LSTM/Mamba backbone。只在 train 拟合归一化，只按 val 选择 checkpoint，test 不参与调参。
- **指标与产物**：报告每手及双手欧氏终点误差，使用原始坐标单位（未独立验证单位时不宣称厘米/米）；同时报告样本加权与 episode 等权均值、持位改进率、每 seed 结果、checkpoint 重载差异及配置/版本/耗时。输出独立目录，不进入正式三头 samples，不覆盖已有实验。
- **解释边界**：任务均衡 10 段且不同任务的 UUID 留出仅为小样本跨任务诊断；轨迹结果不能称为 HOI accuracy，也不能证明 Mamba 优于 LSTM。正式 HOI 门槛仍为物体稳定 ID/轨迹、经确认的坐标系、人工复核三头标签及采集级隔离。

新增诊断工具由⑦集成负责人统筹，④负责时间/分组，⑤⑥复用时序骨干并审计训练隔离；不变更现有正式训练接口。

## 2026-10-22 增补：真实手—物 3D 融合验收

- 本轮暂停训练，先验收同一采集的原图、世界系双手和物体→相机位姿。必须使用来源可追溯的相机标定；优先使用同一标注 episode 的 `meta/episodes` parquet 中 `camera_intrinsics/head_left` 与配套视频。Raw 与标注版的帧号不能直接对应；若使用 Raw 资产，必须另行证明时间与世界系映射，本轮不混用。
- 几何计算仅在明确 quaternion=wxyz、camera→world 后使用 `p_camera = R(q).T @ (p_world - camera_position)`；图像投影使用该相机/该去畸变视频的真实内参。本轮只接受四个畸变系数全为零的标注版内参，非零直接报错，不静默降级。无标定时不允许启发式内参冒充验收结果。手部布局按发布方 21 点定义；发布方文档声明位置单位为米，不等于独立尺度校准已通过。
- 逐帧表至少区分手部有效、正深度、落在图像内、真实物体 pose 可用、重投影误差是否有独立参考、联合可验收与联合通过。可用率/覆盖率不是姿态准确率；独立 2D/3D 真值缺失时误差和成功率留空，不填 0 或 100%。
- 诊断渲染器限定完整、零起始标注 episode：文件路径绑定实际 variant/任务/UUID 与 `head_left`，`frame_index` 必须连续从零起；全片解码 PTS 与暂存时间戳逐帧绝对误差不超过 0.1 毫秒。非零起始、子片段或其他分片布局需另做显式映射，不静默按数组位置假设对齐。
- 物体 pose 验收拒绝 mock、未知/缺失深度来源、启发式/来源未知内参、采集不匹配、时间不匹配与非刚体/非有限矩阵；仅数值矩阵合规不代表精度。没有独立参考时允许记录真实估计覆盖，但 accuracy 及 joint_success_rate 必须为 null。
- 独立门禁 `src.pose6d.acceptance.audit_object_pose` 要求经审核的 `meta.acceptance_evidence`：采集 UUID、camera_id、`object_to_camera` 约定、米制深度来源，以及有引用的标定/时间对齐声明；字段格式见函数文档。此字段是验收旁路要求，不改变正式写出器，不允许只填写声明代替实际验证。门禁只检查声明及文件的一致性，不访问引用、不证明估计器实际执行或准确性。
- 交付 PNG 多视图、连续预览 MP4、逐帧 CSV、JSON 证据与人工检查记录。缺物体时图上显著标注 HAND/CAMERA ONLY 与 OBJECT 6D MISSING，不画虚构物体轴，不称其为完整融合图；整体验收结论为 blocked/not_accepted，并列明可继续的最小前置条件。

涉及②③④⑦；本轮图像/几何产物独立存放，不覆盖源数据、正式 pose6d 或 samples。

## 2026-10-22 增补：6D 输出专项验收

- 位置/旋转数值检查与真实精度验收分开：报告物体帧数量、有限/刚体矩阵、平移与旋转变化，不把单位阵、常深度、轨迹平滑或格式通过当成真实 6D 成功。没有独立参考时，位置、旋转、联合准确率均为 null。
- 显式 mock 后端仍允许用于接口自测，但新产物必须记录 `quality_flags: mock_pose` 与模拟结果警告；旧产物不回写。真实门禁继续检查实际后端来源，不依赖质量标记是否为空。
- `--strict` 同时覆盖后端初始化、每物体 tracker 创建、必需深度缺失和逐帧估计异常：异常立即抛出，不将其吞掉后写出看似正常的结果。非 strict 保留缺失帧降级，但 tracker 创建失败需显式记录 `pose_backend_create_failed`。显式 none/mock 自测不因此获得真实精度认可。

## 2026-10-22 增补：真实 RGB-D 后端跑通协议

- 先在有配套 RGB、米制深度、真实首帧物体 mask、mesh 和相机内参的公开实拍短序列验证 FoundationPose 注册/跟踪，再接项目 EgoDemo；公开样例成功不等于 EgoDemo 真实融合成功。不得填造 mesh 尺度、深度或物体轴来跳过输入缺失。
- 独立 RGB-D runner 采用原始 mesh 坐标的 object→camera 变换；输入显式记录 RGB/depth/mask 路径、深度换算因子、mesh/标定摘要及代码/权重来源。真实分割 mask 送入 register，不以 bbox 矩形代替。输出保持 `poses (1,T,4,4)`、`bboxes (1,T,4)`、稳定单物体 ID 的正式 NPZ 结构，另有逐帧表、PNG/MP4、运行报告；严格失败不冒充成功。
- 参考位姿只允许用于推理后的评估，不可用于初始注册、重定位、相机/mesh 自动对齐或模型选择。评估前确认参考与原始 mesh 的坐标定义、单位及帧 ID；没有此证据就不计算精度。若使用工程烟测阈值，须在推理前显式给出位置/角度阈值，并标为本轮工程阈值而非数据集官方基准。
- 环境使用数据盘上的独立验收环境，不降级/覆盖现有 `panda_6d` 或其他同学环境；权重/样例/编译缓存置数据盘，不修改 base，不启动 HOI 训练。

## 2026-10-22 增补：仅真人手的在线手—物联合提取

- 用户明确排除机械臂样例。输入必须经过实帧检查，确有真人手与目标物体；优先实际 EgoDemo，替代公开人手 RGB-D 序列必须明确数据集和视角，不可冒称 EgoDemo。缺米制深度、真实首帧 mask 或目标 mesh 时不能用 mock、任意尺寸模型或 GT 位姿替代。
- 同一逐帧循环内执行真实手部模型和 FoundationPose，再立即画入该帧并输出，不用历史标注手或未来帧伪装实时提取。保存同帧手部 NPZ、原 mesh→camera 物体 NPZ、来源、时间戳、逐帧耗时及联合覆盖率；缺手填 NaN，不插值制造检出。
- 手部主输出保持 MediaPipe 图像归一化 x/y 与相对 z，不将 hand-centered world landmarks 直接当 camera/world 米制关节。与物体坐标轴同屏重投影只称“同帧联合叠加”，未经校准不称完整三维同系融合。
- 真人左右手按明确的输入是否镜像规则处理并记录，不把“第一人称”自动等同自拍镜像。检测分数/左右手分类分数不可冒充关节精度。
- 区分在线因果处理、文件回放和真实相机接入。视频回放 FPS、实际稳态处理 FPS、逐帧延迟分别报告；只有实测达到指定输入速率才称实时达标，保存 MP4 本身不构成实时证明。没有连接 RGB-D 相机时不得声称已经接通实时摄像头。
- 允许有界的下一帧 I/O 解码预取，但不提前执行未来帧模型推理；最多预取一帧，记录预取配置，并分别报告消费环路耗时与文件读取开始→输出延迟。P95预算达标不等于每帧无超时，也不等于硬实时。

## 2026-10-22 未来姿态预测原型（pose_forecast_v1）

涉及②③④⑤⑥⑦：新增独立 `src/forecasting/`，不替换现有三头 HOI decoder，不把姿态预测与交互分类混为一谈。

- 输入仅为截至 observation time 的真实估计结果。默认历史32帧、预测0.5/1.0秒；目标帧按时间戳定位，越界/断档/无效目标不制造标签。采集 sequence/group ID 整体划分 train/val/test，不以不同相机或相邻窗口伪装独立采集。
- 初版 `coordinate_mode=mixed`：手部保持图像归一化 XY/相对Z，物体为原 mesh→camera 的米制变换；禁用跨域距离/contact。固定相机先验必须记录，未经相机运动补偿不能声称第一人称世界系预测已验收。
- 模型 batch 沿用 `hand_seq (B,W,2,21,3)`、`obj_seq (B,W,N,13)`、`obj_mask (B,N)`，其中物体特征顺序保持位置3/旋转6/bbox面积1/速度3。
- 新增几何回归输出：`hand_future (B,H,2,21,3)`、`object_future (B,H,N,4,4)`，H按 `horizons_seconds` 排序；无当前观测的实体保持NaN，不把零填充值当预测。旋转须在SO(3)，矩阵末行固定[0,0,0,1]。训练监督同名目标、独立有效性掩码；只在有效位置计算损失。
- `PoseForecastModel` 复用 HOIEncoder 和 LSTM/Mamba 时序骨干，新增 PoseDecoder，初版不修改旧HOI标签体系。模型配置/checkpoint需记录窗口、预测间隔、坐标、来源、划分、权重和代码摘要。
- 推理持位/因果恒速基线及学习模型均不得访问未来标签；未来估计只能在评估和独立对照画面使用。未来帧的MediaPipe/FoundationPose输出属于伪标签，不是独立GT，相关误差只称相对于未来估计的一致性。
- 视频左侧仅当前图像+当前估计+虚线未来预测，右侧可显示明确标记的离线未来对照；预测时刻、目标时刻、方法、来源需可查。纹理贴图不是前置条件，不生成伪未来RGB。
- 单一或短序列只交付demo/工程测试，不能随机拆帧制造泛化结果。学习模型若未优于持位基线须如实保留失败结果；模型选择只用val，不依据test挑参数。报告按采集等权及样本加权、每预测间隔、有效覆盖率、手像素误差、物体米制位置误差和有对称性限制说明的旋转误差。

## 2026-10-22 数据先行：human_pose_dataset_v1

- 用户要求先扩充长且多样的真人序列，再规范划分，再训练/评测。本阶段不启动训练；旧6段436帧仅保留为工程烟测，不作为正式数据验收通过的依据。
- 新增独立候选台账与发布审计，不覆盖原始数据或旧划分。台账顶层为 `schema_version: human_pose_dataset_v1`、`records`；每条记录包含 `sequence_id, dataset, acquisition_id, subject_id, scene_id, object_ids, action_labels, camera_id, view, frame_count, duration_seconds, fps, timestamp_source, human_verified, assets, capabilities, quality, split`。未知身份/动作/坐标/时间依据用空值或pending记录，不推测补齐。
- `assets` 为带 `role,path,sha256` 的文件列表；相对路径按台账所在目录解析，校验实际内容，不把链接、下载计划、LFS指针当作数据。`capabilities` 显式区分 `rgb,depth,intrinsics,mesh,hand_metric_gt,object_pose_gt,hand_observations,object_observations,camera_motion_compensated`；GT与估计分目录，不将未来GT接入感知/初始化。`quality` 的 `decode,timestamps,calibration,coordinates,annotation_alignment,visual_review` 分别为pass/fail/pending。
- 正式联合姿态数据必须有连续真人RGB-D、真实K/目标mesh、可核验的米制手/物体参考、同帧估计与有效性掩码。只有手部标注/无物体GT的长EgoDemo进入辅助池，不冒充联合6D基准；移动相机没有运动补偿时不进入固定相机预测发布。
- 连续时长至少10秒，优先20–60秒；不通过拼接独立采集、重复帧、插值、改变fps制造时长。不以滑窗数代替独立采集数。每条时间证据必须来自源时间戳/视频PTS/官方帧率，而非未经核验的默认30fps。
- 第一版暂定数据门槛（项目门槛，不是官方benchmark）：至少120个独立合格采集、6名有依据身份的被试、8种物体、4种有依据动作；train/val/test目标70/15/15，实际比例因整组划分偏差须如实报告。按被试隔离为主划分，同采集全部机位/切片/转码/源内容别名必须同组；未知被试不伪称跨被试泛化。test至少15个独立采集，val至少15个；train至少60个。动作/物体分布与未见类别单列，不能为凑比例拆组。
- 划分在窗口生成、归一化拟合、训练之前冻结，记录种子、策略、内容SHA和清单SHA；一旦冻结不可原地覆盖。训练统计仅来自train，val用于超参和早停，test仅在训练流程冻结后最终评测。旧test已经看过，只能作为开发/回归数据，不进入新未见测试集。
- 候选分组权重使用标注/质量筛选之前的原始独立采集数，不能使用就绪数（否则全部pending时会全落入train）。同一批原始采集补标注不能改变被试组归属；跨主机路径适配后内容SHA和候选划分必须一致。追加新数据须新版本并审计既有holdout归属，不自动改写旧冻结。
- 审计输出必须分别统计候选/实际下载/真人确认/合格长序列/联合姿态就绪/每split独立采集、被试、物体与动作；不满足门槛或存在泄漏时 `training_allowed=false`，不得输出“正式数据集已完成”或自动启动训练。
- 本阶段规模目标锁定为120个通过验收、能用于真人手部＋物体6D联合训练/评测的独立采集，目标train/val/test约84/18/18，优先按被试和采集整组隔离，实际偏差单列。持续进度必须区分已下载联合候选数与已验收联合合格数，距120缺口按后者计算；辅助池、重复机位/窗口、插值重复片不能抵扣缺口。既有冻结保持不变，扩容形成新版本并保留已冻结被试归属。

## 2026-10-22 完整真人序列生产试跑：full_hocap_v1

- 本轮只跑现有采集的一整段，不扩容、不训练；先使用候选train组subject_7/20231022_192832、机位105322251564的全部446帧及4个物体。完整帧数取官方meta.yaml，0到num_frames-1不得裁剪/重复/插值或跳帧后报全段成功。
- 准备阶段从首帧公开标注仅提取K和物体分割mask，不读取obj_poses或hand_joints作推理/初始化。此为公开首帧分割辅助初始化，不声称自动检测或无标注初始化。推理输入独立manifest只含RGB-D、K、对应原mesh、首帧mask及来源证据，不接入后续帧参考或未来帧。
- 坐标为OpenCV右手相机系（x右/y下/z前）；物体4×4为原mesh到相机，mesh单位m，深度原始uint16按有依据scale转m。MediaPipe手部保留图像归一化XY/相对Z；另存RGB-D反投影的hand_camera_surface_m及逐关节有效mask，明确它是深度表面代理，不是经独立验证的解剖关节中心。
- 输出包含frame_ids、source_timestamp_s、source_timestamp_valid、display_timestamp_s、processing_elapsed_s、左右手(2,T,21,3)、物体(N,T,4,4)、各级有效标记和逐帧耗时。源物理时戳未取得时全NaN且valid=false；展示时间仅用于播放，不冒充采集时戳或可训练预测时域。
- 模型必须是真实MediaPipe/FoundationPose，无mock/参考位姿回退；缺失为NaN，不插值。逐帧同帧推理后渲染人手骨架、4个物体框/XYZ轴与帧号。报告完整处理、可解码视频、输入/输出哈希、覆盖率、实测耗时、模型版本和错误；工程验收与独立精度分开，精度未知保持null。
- 输出目录必须新建，失败记录已处理帧数和原因，不将partial视为complete。FoundationPose状态不做不安全的中间帧伪恢复；恢复需保留失败目录、修复原因后以新run_id从首帧重跑。批量先准备与验收每采集，再串行或明确资源分配运行，不绕过数据发布门禁。

## 2026-10-23 增补：姿态实体与协作关系可视标注（pose_relation_annotation_v1）

- 用户指定标注形态：真实手部骨架叠加解剖左右手/手ID，真实物体6D位姿叠加稳定物体ID/名称，再定义带时间区间的协作关系。新增独立离线标注层，不改变既有七类编号、三头训练接口或源姿态NPZ。
- 顶层 `schema_version,sequence_id,frame_count,source_bindings,entities,relations,transitions,cooperations,task_semantics,hand_review_frames`。绑定原scene manifest及sequence NPZ的SHA；所有区间为同一采集native帧的 `[start_frame,end_frame)`。没有源物理时戳时仅按帧标注，不把播放30FPS换算结果当实际时间。
- `entities` 包含 `entity_id,kind,display_name`；hand实体固定 `left_hand/right_hand`，side是被摄者解剖左右；object实体ID必须逐一对应源NPZ的object_ids，名称引用官方元信息或明确标为待审；context实体可表示桌面等环境，但没有6D位姿时不画伪造物体轴。实体角色可由协作记录按区间赋予。
- `relations` 为多条可同时存在的有向边：`relation_id,source,target,predicate,start_frame,end_frame,evidence_frames,label_source,review_status,notes`；首版词表 `approaching,contact,grasp_control,co_motion,supported_by,stabilizes,acts_on`。不得仅凭2D距离、bbox重叠或同屏出现认证接触/抓持/协作；geometry_status单独记录。未标注/缺测不是none，手部归一化坐标与物体米制坐标不直接作距离计算。
- `transitions` 包含 `event_id,source,target,event_type` 与同样的区间/证据/审核字段；区间表示转换发生位置的待审范围，不制造精确接触帧。词表 `contact_onset,grasp_onset,motion_onset,support_gained,support_lost,release`。
- `cooperations` 保存参与实体及角色 `participants:[{entity_id,role}]`、引用relation_ids和时间区间；mode取 `single_hand_manipulation,bimanual_joint,bimanual_complementary,handover,tool_mediated`。双手模式必须真的含左右两手；单手样例不冒充双手协作。工具、直接接触物体、被作用工件可以不同，分别保留。
- 所有语义关系/转换/协作候选默认为pending，图上用虚线与“待审”区分；accepted必须有真实人工reviewer/reviewed_at、证据帧及human来源。模型或助手视觉抽查不能伪装人工金标准。hand_review_frames显示手别异常提示，不暗改源关键点槽位。
- 交付同帧叠加MP4、预览图、原annotations.json副本、逐帧frame_graphs.jsonl、review.csv及来源/输出SHA清单。逐帧图保留并行关系与source/target实体观测有效性；非因果标注 `allowed_as_model_input=false`，本可视化模块不放行正式训练。涉及②③④⑤⑥⑦，由集成负责人后续同步。

## 2026-10-23 增补：中文网页人工审核（pose_relation_review_web_v1）

- 网页读取固定446帧采集的原生帧JPEG、播放视频、姿态投影与原始候选；显示中文名称/关系/角色/状态，内部稳定ID和既有schema字段不改名。画面关系从当前审核草稿动态绘制，不使用烘焙旧关系的视频冒充修改后结果；暂停时用原生帧图精确定位，播放30FPS仅为展示时间。
- 审核四层区间（持续关系、转换事件、协作方式、任务语义），编辑主体/客体/类别/区间/证据帧/备注；支持新增候选。界面结束帧为“最后包含帧”，导出转换回 `[start_frame,end_frame)`，最后帧445对应end_frame446。越界、非整数、空证据、证据不在区间及不存在的实体拒绝保存。
- 草稿保存在当前浏览器，以sequence内容SHA隔离；导出 `pose_relation_annotation_v1` JSON及中文表头CSV、可导回同一来源草稿。审核确认需用户填写审核人，并显式点击通过/需修改/否决；accepted记录human来源与ISO审核时间，保留original_label_source。任何已审核记录内容再次改变时回到pending。协作通过要求依赖关系均已通过；依赖关系修改/否决后已通过的协作自动回到pending并提示。
- 另存 `handedness_reviews`（帧号、中文意见、状态、审核人/时间），处理5帧手别异常；仅审核备注，不暗改原手部槽位或NPZ。实体中文显示名可改，不改稳定ID。既有source_bindings和原数据不改写。
- 导出附加 `web_review`（version、base_annotation_sha256、exported_at、修订信息），保留 training_allowed=false、allowed_as_model_input=false。标注审核通过不自动等于姿态质量或正式训练准入通过。浏览器自动保存不冒称云端数据库同步；导出文件是跨设备备份/交接方式。
- 用户在本会话明确确认整批通过时，可保存带原话、确认范围、审核应用时间和原始SHA的 `batch_review`，审核人写“用户（本会话确认）”，不虚构实名。派生新的已审核JSON与逐帧图，保留原候选版本；手别审核表示用户复核完成，不自动纠正姿态或放行训练。网页新版本以新annotation SHA隔离草稿；可从 `previous_annotation_sha256` 对应旧草稿迁移已确认ID的审核状态，保留原字段及旧草稿，不重复覆盖此后修改。确认时间之后的草稿修改仍待审，新增未包含ID不自动接受。

### 第二轮补全的人工表面种子与几何审计补充

- 对初次重建无参考的采集追加独立 `manual_observations`，保存实际看过的原生帧、多边形、预览SHA、assistant来源和pending状态；检测/SAM分数为null，不伪造模型置信度。原自动检测目录保留。
- 小目标手工种子使用1280宽双目重建和3原生像素特征间距；仍保持左右一致性1.5px、至少12个有效参考点、同一鲁棒拟合/连续帧门槛。手工多边形避开可见手和穿戴设备，替代过宽的手腕排除带；不降低几何验收阈值。
- 订书机壳体/底座等只给 `pose_scope=rigid_part`，不把铰接整物体伪装刚体。透明袋只保留可见表面估计与归属不确定性；面粉等颗粒内容不赋刚体6D。
- 全量工程审计逐采集检查来源SHA、native时轴、左右手投影、SE(3)、NaN/valid一致、物体身份引用、边界范围和训练门禁；通过只表示文件/坐标合同成立，不代表姿态准确或人工金标准。
- 增补像素级双目标定检查：每采集四分之一、二分之一、四分之三帧，双向SIFT比值匹配后用独立MAGSAC筛选，比较对应点与源标定F矩阵的原生像素极线误差。结果为抽查证据，不是完整标定或精度真值。两帧以上有效且中位误差中位数>4px标为inconsistent；其三维物体轴、距离交互、三维身份关联在派生层全部禁用，原始估计保留诊断，不凭空修正官方标定。
- 2026-10-23继续补救：对原标定与图像不一致的采集，可在独立`calibration_recovery_trials`拟合右相机相对左相机外参，基线长度严格沿用原标定（尺度不是图像独立测得）。前两个采样帧用于稳健拟合；第三帧和10%/45%/90%三个额外帧均不得参与拟合，验证极线误差、匹配覆盖和三角化有效率。通过只允许候选重建，不能声称独立位姿精度或改写官方标定。参考点与原生追踪必须都记录同一校正SHA后才能解除派生几何拦截，旧几何不得只改状态冒充重新计算。
- 原文操作补全单列为`semantic_drafts_refined`：修复onto/into、load、cover、lay等词形漏匹配，增加有实际原文证据的提起、舀取、分装、夹固、拆解、压平；保留原中文和原生区间，逐操作保存原文连续证据及span。旧语义目录不修改。规则命中仍是source_interval_only/pending，不替代接触边界、主体/客体ID或人工金标准，不改旧训练七类编号。

## 2026-10-23 用户授权并行交付：egodemo_mamba_v2

- 用户明确授权多个agent并行，并要求直接交付Mamba训练数据格式。本轮交付实际序列分片、窗口索引、加载器、编码适配器、配置、来源和质量审计；不启动训练，不覆盖旧7类三头接口。
- 新schema `egodemo_mamba_v2`；序列NPZ使用原生时间戳，保留官方手姿态、过滤后物体位姿、非刚体表面点和各级有效mask。JSON保存中文实体、来源、几何/交互审核状态及稀疏事件。未知监督不是none，旧7类不硬映射49种操作。
- 窗口默认32个原生观测帧、stride=16，目标为观测终点后1.0秒的首个源时戳帧，允许误差不超过1.5倍本段中位帧间隔；不插值。按独立采集/可识别采集分组先冻结train/val/test，再生成窗口，归一化不读取val/test。
- 加载后固定接口：`hand_seq[B,W,2,21,3]`、`hand_valid[B,W,2]`、`obj_pose_seq[B,W,N,4,4]`、`obj_valid[B,W,N]`、`surface_seq[B,W,N,P,3]`、`surface_valid[B,W,N,P]`、`object_kind[B,N]`、`object_mask[B,N]`；默认N=8、P=16。所有几何为世界系米制后以最后观测帧的左相机为共同锚点，非法几何保持NaN并附mask，编码器内部才安全替换。
- 物体槽位仅依据观测窗内可见性排序（首次出现帧、稳定ID提案），不得用未来事件、未来持续时长、未来置信度选择。表面点按观测窗内已出现的`tracklet_id/point_id`键选择，在未来目标中保持同一点键，不补未来才出现的点。
- 单步目标：`target_hand[2,21,3]`、`target_hand_valid[2]`、`target_obj_pose[N,4,4]`、`target_obj_valid[N]`、`target_surface[N,P,3]`、`target_surface_valid[N,P]`。物体/表面来自离线双向估计，保留`allowed_as_model_input=false`；当前仅诊断加载，不将其伪称已验证的因果观测。
- 源语义49种及物理关系/转换/协作候选使用独立词表与多标签候选张量，已审核监督mask仍为false；源区间标签另有weak候选掩码，但不能自动变为gold标签。正式联合训练门禁保持false，允许显式diagnostic模式完成真实Mamba前向检查；不得把全mask监督的零loss称为有效训练。
- 目录建议 `egodemo_mamba_v2_300_20261023/{sequences,windows,splits,vocab.json,dataset_index.json,export_report.json}`。每序列/窗口和划分必须绑定内容SHA与原选择SHA。未独立核实的被试不冒称跨被试划分；记录实际使用的scene/environment/UUID分组及残留泄漏风险。

### egodemo_mamba_v2 序列分片与索引字段冻结

- `sequence.npz`: `timestamps(T) float64, frame_index(T) int32, hands_world(T,2,21,3) float32, hand_valid(T,2) bool, camera_to_world(T,4,4), object_ids(N) string, object_kind(N) int8 [0unknown,1rigid,2deformable,3articulated], object_pose_world(T,N,4,4), object_pose_valid(T,N), object_observed(T,N)`。`object_observed`仅表明候选可见几何，不能用于未经审核的存在/不存在监督。
- 表面点稀疏COO：`surface_frame_index(S),surface_object_index(S),surface_point_index(S),surface_world(S,3),surface_frame_ptr(T+1)`；`point_keys(M)`编码`tracklet_id/point_id`，`point_object_index(M)`绑定实体槽位，加载器按原生帧指针读取。非法/被拒绝的3D点不进入COO，坐标中的缺失不填造。
- `metadata.json`保存中文实体与源内容/过滤SHA；`events.json.records`含`layer,type,start_frame,end_frame,object_index,subject,label_valid=false,weak_positive=true`；源语义的主体/客体仍未知（null或-1），不从近邻几何推断。
- `windows/<split>.jsonl`含`sequence_id,start_frame,end_frame`（排他）、`anchor_frame,target_frame,target_timestamp,actual_horizon_s,sequence_path,metadata_path,events_path`。路径以新dataset根为准。vocab按`operation/relation/transition/cooperation`独立存`codes/names_zh`，operation为当前49实测类型。

### egodemo_mamba_v2 加载与解码接口定稿

- 冻结划分为210/45/45采集，按scene/environment连通组隔离（53/12/12组）；此前已见34采集关联的129采集全部放train。被试均缺失，不能宣称跨被试评测；采集比例不代表帧数/窗口比例，任务可以跨split。
- 事件张量`event_candidates,event_weak_mask,event_label_valid`为`(B,N+1,5,K)`，第N槽是未知客体，主体槽顺序0左手/1右手/2双手/3物体/4未知。K为49操作+6关系+6转换+5协作=66；这只是词表容量，当前非零候选为49/3/2/1类。多标签候选未命中位置为unknown，不能作为负例；全部人工监督mask=false。
- 状态/操作以目标帧是否在半开区间内为准；转换候选区间与`(anchor,target]`相交，只表示不确定发生范围，不伪造精确边界。已知但未入观测槽位的客体事件丢弃并计数，不归入未知客体。
- 逐帧编码将左右手、物体位姿、非刚体表面点、相对手腕位置、材料和各级mask转为`(B,32,11,128)`实体token，帧内注意力后转置为`(B*11,32,128)`交给真实Mamba。输入编码只读观测字段，不读取target或事件标签。单步输出`hand(B,2,21,3),obj_pose(B,8,4,4),surface(B,8,16,3),event_logits(B,9,5,66)`及独立prediction masks；旋转6D经正交化转SO(3)，事件用sigmoid独立解码到中文。
- 物体位姿的物体系为被追踪可见表面的参考系，不是CAD标准轴。所有模型几何输出均在最后观测帧左相机坐标系；转回世界系使用随样本保存的`anchor_camera_to_world`。无物体有效位姿时，网络数值输出必须结合预测mask读取，不能将padding或非刚体head值当有效6D。
- 本版本只提供独立Dataset/模型/真实前向验收入口，不能直接交给旧`src.mamba.train`的七类训练配置。`diagnostic=True`只允许检查未审核、非因果估计的数据结构，不构成训练许可；真实前向随机初始化、0 optimizer steps，无效果指标。

## 2026-10-23 用户要求完整标注与文档：assistant_annotation_v3

- 保留冻结300采集、v2分片和split原样，在独立`egodemo_annotation_v3_300_20261023`产物目录补实体、角色、交互和证据。不改写原始姿态、来源标注或既有人工记录；代理标注不伪装human审核。
- 每采集JSON含`schema_version,sequence_id,source_bindings,entities,interval_annotations,relations,transitions,cooperations,coverage,review_records`。全部原文区间逐一建记录；未标注时间区间和每层可观测性单独保存，不能把未观察当none。实体沿用既有ID；新可见但无3D实例另有visual实体ID，geometry_status=unavailable，不伪造位姿。
- 记录`annotation_source`区分`assistant_visual_review/source_text/geometry_rule`，`review_status`区分`assistant_reviewed/estimated/unresolved/rejected`。助手只有实际查看绑定图像/帧后才能写assistant_reviewed，并记录agent、时间、图像SHA、native证据帧和判断范围；规则生成不能自动标为已视觉复核。不填写虚构人工审核人。
- `silver_label_valid`只对证据支持的助手标注成立；`human_label_valid`保持独立。语义操作来源区间、物理接触、抓持、支撑、工具作用不能互相冒充。未见到转换瞬间记录半开不确定范围及censored标记，不凭原文顺序合成精确帧。
- 物体角色可随区间变化，工具/工件/容器/支撑物分开；双客体关系保留source/target实体引用。非刚体保留表面轨迹与材料/可观测性；有形变不能用整物体6D补齐计数。跨遮挡身份仅在有视觉/几何证据时合并，否则保留片段身份。
- 完整性报告同时给出300采集处理覆盖、1666源区间处理覆盖、实际看过证据的区间数、实体/关系/转换各自确认与未知数、剩余几何缺口；不能把文件齐全或空值齐全称为完整标注完成。已有物体/表面几何非因果属性保留，训练准入不因新增标注自动改变。
- 统一视觉复核文件为`reviews/<shard>/<UUID>.json`、schema=`assistant_annotation_v3_review`；绑定原proposal SHA和该采集evidence manifest SHA，含reviewer_agent/reviewed_at和逐source_interval_id记录。每区间记录实际查看的gallery相对路径/SHA/native帧号、`evidence_scope=sampled_native_frames_only`、中文观察与operation/entity assessments。判断status取`supported/contradicted/uncertain`；`assistant_reviewed`只表示看过，不使其中unknown标签自动确认。4路视觉分工按冻结排序UUID为[0,100)/[100,200)/[200,280)/[280,300)。
- v3交付是v2数值分片上的独立标注层：`annotations/<UUID>.json`及`mamba_overlay_index.json`绑定原v2 index与split SHA。原数值几何和53,255窗口不改；独立adapter可读取新增银标和双客体图边，原human mask继续false。只在明确有证据的点或连续范围上赋监督，抽样看到动作不能自动扩展为整个原文区间的逐帧物理状态。暂未入观测槽的新视觉实体保留图引用和未映射原因，不挤入旧padding槽。
- 视觉确定的错误实体须给出显式`geometry_exclusion_scope=entity_until_reidentified`（误框/身份）或更窄帧范围；v3适配器仅在派生batch中清除此实体位姿/表面/目标和存在槽mask，保留NaN及拒绝原因，不改v2源数组。角色或可修正名称不自动触发几何拒绝。该全局质检过滤也不能被宣传为已验证因果感知。

### assistant_annotation_v3 监督与适配最终约定（2026-10-24）

- 操作记录的evidence_frames可以仅表示支持“源区间内曾发生”的观察范围。只有显式temporal_coverage=sampled_points并由复核者选择实际操作帧，才生成逐帧silver_event_records；sampled_evidence_interval_presence等不进入帧监督。状态关系也只在明确evidence_frames激活，不由start/end包围范围补全中间帧。
- 双客体图以source/target为有向边事实源；兼容字段object_id=source、secondary_object_id=target、subject=object。旧兼容字段矛盾时保存original_object_binding并显式规范化，不把工具→工件压成工件→工件。
- 协作仅给participants时，保留所有显式参与者；唯一对象可规范化为object_id，两手参与则subject=both_hands。多对象写participant_object_ids/participant_object_slots，只保留图目标，不伪造方向或压到unknown客体/单客体头。
- 根复核中relation:grasp与cooperation:bimanual分别是grasp_control、bimanual_joint的同义命名；最终银标层按冻结词表规范化并保留original_type/type_normalization，原review不改，统计不靠同义别名虚增类别。operation:grasp（拿取）不受该映射影响。
- EgoDemoAnnotationV3Dataset继承v2观测槽选择，增加silver_event_candidates和silver_event_label_valid，形状(B,N+1,5,K)。未知为mask=false；当前银标只含支持正例，不生成隐含负例。graph_targets保留完整双客体边及未映射视觉实体，不作为现有单客体事件头的已学习目标。
- v3派生batch将旧v2 event_weak_mask全部清零，防止holder/plug/opening及opened/folded等名词或形容词误匹配继续监督；event_candidates旧值仅留溯源。新原文质量判定仍在标注JSON独立保存。human mask保持false。
- 模型输入编码/几何输出沿用v2；银标和图目标不进入Encoder。现有模型头不新增双客体图预测，交付图结构供后续图头使用。正式train gate不因银标数量增长自动解除。
- 视觉分工在长序列收尾时接力为A[0,100)及[173,180)、B[100,173)、C[200,280)、root[180,200)和[280,300)，均按冻结UUID顺序；每条review另有实际agent和时间记录，接力分片独立保存避免覆盖。
- `annotation_target_coverage.json`枚举冻结53255窗口，并对时间命中明确证据的窗口实际调用v3适配器，分别统计graph_targets与单客体头可用标签；不能用1666区间复核完成数替代窗口监督覆盖。当前所有split的画面都参与了助手标注，因此不再是对本标注流程的盲测；split不变也不保证被试隔离。
- 补充`review_windows/{train,val,test}.jsonl`，从明确sampled_points操作/关系/协作证据帧构建每采集每target唯一的32帧历史窗：anchor是实际horizon>=1秒的最大历史帧，超时不超过1.5倍原生median_dt，anchor>=31；不从删失转换合成帧、不补造早于历史窗的观测。新行沿用v2数值路径/源选择SHA/固定split，另绑定annotation及证据记录；index绑定base/overlay/split和各JSONL SHA。旧53255窗口与索引均不改，默认Dataset仍加载旧窗口，显式windows_path绝对路径才使用补充窗。
- 补充窗是按已有正例选择的诊断数据，不增加独立采集数、不产生负例，也不能作为无偏评测集。`review_window_target_coverage.json`实际加载统计与原窗口覆盖分开；`annotation_v3_smoke --review-windows`可在这组索引上做随机权重真实前向，仍不执行训练或解除正式门禁。

## 2026-10-24 继续补连续标注：assistant_annotation_v4_dense

- 用户要求继续推进到连续逐帧交付。V4独立根`egodemo_annotation_v4_dense_300_20261024`绑定冻结V3 overlay、V2 index/split；不得覆盖V3或把旧抽样银标扩成稠密真值。
- `dense_reviews/<UUID>.json`记录真正连续查看的原生帧：schema_version、sequence_id、source_bindings、reviewer_agent、reviewed_at、evidence_manifests（路径/SHA）、reviewed_ranges半开区间、entities、state_segments、transition_events、unresolved_ranges。连续证据manifest须列出每个原帧/时间戳/原图decoded SHA、裁剪xyxy、联系表SHA及该页实际显示的全部帧；声明已看范围中每一帧必须有实际查看证据，不能以固定跳帧代表连续审核。
- state_segments用`layer,type,subject,object_id,secondary_object_id,start_frame,end_frame,value,evidence_frames,reason_zh`：value=1明确正例、0明确负例；未确定不是0，另存unknown原因。正/负片段只覆盖实际连续查看且有判断依据的帧，支持sampling_scope=continuous_native_frames_reviewed。负例限明确对象/关系，不从没有提到推导整个场景无交互。银标来源assistant，human始终false。
- transition_events以native半开boundary_frame_range=[lo,hi)表达事件发生帧范围；只有逐帧可见状态转换且范围长度1才可称frame_resolved。手-物首次可见接触、建立抓持控制、操作开始、支撑建立、最终可见脱离分开，未看到区间外此前状态则censored。不可由源文本端点或距离阈值自动产生精确边界。
- `entity_grounding/<UUID>.json`保留实际ID支持锚点/视觉实体别名、原生轨迹来源、候选关联与排除。单帧身份确认不等于整段身份确认；只有同一连续轨迹中的可见观测可形成tracked_candidate范围，遇缺测/遮挡/换实例即断开；仅同名、同角色、同框附近不足以绑定visual实体或合并轨迹。几何仍是候选，不为无3D视觉实例造位姿。
- `dense_sequences/<UUID>/`提供逐帧索引与图数据：所有原帧都有frame_index/timestamp/hand_valid/geometry_valid及review coverage，但数值完整处理与语义标注完成分别计数。源文本区间、旧几何规则候选、实际逐帧银标和未知原因分层存储；近距离只称proximity_candidate，不认证物理接触。
- V4审计逐段报告总帧、手有效、任意场景几何与明确交互对象几何、实际查看连续帧、各关系/转换已知正负与未知、未对齐实体、边界缺项。`dense_index_complete`仅表示原生时间轴无缺帧；`continuous_visual_review_complete`与`full_annotation_complete`必须按实际证据判断，未看帧不能靠unknown占位算完成。
- 逐帧数值补算文件`dense_sequences/<UUID>/observables.npz`保存frame_index/timestamps、hand_valid/hand_speed_mps、object_ids/pose_valid/surface_point_count、nearest_hand_surface_m/distance_observed/near_surface_candidate（F,N,2）、object_speed_mps/relative_speed_mps及v3银标点CSR索引。距离由每帧真实surface_world点与同帧手关节重算；至少8点才认为距离可用，阈值0.03m仅用于接近候选。缺手/缺点/误框为NaN和mask=false，不当作关系负例。数值特征仍来自离线几何，非因果；threshold crossing只能进入待复核边界队列。
- Mamba可读取连续复核片段的真实正/负掩码和有范围的转换标签，保持未知mask=false、原split不变；几何非因果和独立精度不足仍不启动正式训练。补标进度以实际新增连续复核帧、已确认边界及可信实体绑定衡量，不以新增文件数或窗口数代替。
- `dense_delivery_report.json`与`逐段连续标注缺口.csv`分别汇总实际数值索引、逐帧已看、确定正负关系、范围转换、精确物理转换、独立视觉信号、候选身份和几何缺失。必须绑定独立审计和最终review SHA；`delivery_manifest.json`列交付文件/源码快照校验和。工程格式就绪与全标注完成分开，已看帧不能作为全层完成帧计数。
- `dense_reviews.visual_identity_aliases`可在同一物体全范围连续可见时保存V3多个visual ID的对应：alias_id、canonical_entity_id、start_frame/end_frame、全范围evidence_frames和中文理由。此对应仅统一视觉实例；不据此合并obj_ID轨迹、传播整物体位姿或增加几何完成数。缺少连续同一实例证据时保留独立片段ID。
- `mamba_dense_index.json`绑定V2/V3/原split、全部实际review和证据SHA、数值/实体/连续审计报告。`dense_review_windows/{train,val,test}.jsonl`只使用有明确state的目标帧，32帧原生历史及真实至少1秒horizon沿用V3规则；缺历史目标排除。默认旧窗口不变，使用新窗口须显式传绝对`windows_path`。连续窗口独立审计必须验证当前连续审计SHA及全部源文件。
- `EgoDemoAnnotationV4Dataset`新增`dense_event_values`与同形`dense_event_label_valid`（batch后B×9×5×66）。mask=false处的0仅是未知占位；V3 silver mask另存。`graph_targets`保留连续片段的正负图边及范围事件，`v3_graph_targets`单列旧抽样图。whole visual_ID没有等价整物体绑定时不填入旧未知槽、不以component_of部件替代；双客体边不压成单客体监督。现有模型没有新增图预测头，实际可监督数量以全窗口加载报告为准。
- 中文连续审核网页`review_web/index.html`是只读浏览器，显示原生帧、明确正/负/未知、半开边界及距离阈值待审队列。未渲染帧显示缺图，不以邻帧充数；浏览不产生human批准或标注写回。网页QA、实际前向及独立审计均须纳入交付SHA清单。
- 打包先验证独立审计方冻结的`audit_lock.json`、五份审计/索引、逐段来源与汇总计数，不由打包器自行重写锁来放行。`delivery_manifest.json`描述服务器完整V4派生目录；私有增量压缩包用`PACKAGE_SCOPE.json`另列实际包含文件、排除的原像素裁剪及外部V2/V3依赖，不能把轻量镜像冒充完整独立训练集。

## 2026-10-24 推进训练交付：V5观察实体与有向图监督

- 原300采集与V2/V3/V4全部冻结；新根`egodemo_training_v5_300_20261024`单列实际可训练子集和其余待补数据，不用子集数量替代300段完整目标。新增val/test连续复核沿用原split，不能重新划分已看视频以凑覆盖。
- V5补可见整物体的二维观测身份：`visual_observations/<UUID>/seeds.json`记录实际当前原帧中的entity_id、bbox_xyxy_native、frame_index、evidence路径/SHA、助手来源及中文依据。种子只在其当前帧及以后生效，不向前传播、不以未来标签选择历史槽位；跨遮挡无证据不重识别。二维整物体框不产生整物体6D真值。
- 在线重放跟踪只使用已见前缀与当前图像；前后光流一致性检查限相邻已见两帧，不读取未来帧。保存bbox/valid、种子帧和每帧来源，丢失停止；重新初始化必须有当前帧的显式种子。种子来自人工式当前帧标注，观察模式明确为`causal_replay_with_annotated_initialization`，不宣称自动端到端感知。离线V2物体几何不进入该模式Encoder。
- 图模型的实体槽含左手、右手、双手组合及历史已出现的整物体；对象排序仅由历史首次有效观测及稳定ID决定。数值观测不包含角色文本、目标事件、未来出现实体或未来形状。手与物体二维特征采用同一图像归一化坐标，官方3D手部另保留独立mask和参考系；有无几何与对象有无分开。
- 有向预测张量`graph_logits[B,E,E,K]`与同形`graph_values/graph_label_valid`；K沿用冻结66类分层词表，E=3+N。左/右/双手→物体表达操作/关系/协作，物体→物体表达双客体关系；unknown无监督，不强塞unknown槽。模型须保留对象槽置换等变性，不能平均对象后丢失身份。损失只读取明确mask，必须有真实非零监督及梯度，不把全mask零loss当可训练。
- 转换区间即使不精确到单帧，只要有明确非删失的发生范围`[lo,hi)`，且该范围完整位于`(anchor,target]`，可提供“预测时间段内事件发生”正监督；部分重叠、删失、未知则只保留范围，不生成精确帧或负例。此语义需与目标帧持续状态分开记录，光照变化不映射物理扫码成功。
- 验收分别记录连续复核覆盖、二维观察/稳定身份可用窗口、有效图监督正负、各split数量、未来污染/槽位置换/掩码检查和真实Mamba前向/反向。训练数据可用性只对明确验收范围成立；整物体6D、非刚体表示及300段完整标注门槛独立，不以格式或非零loss宣称全部完成。正式长训练不在本次数据交付中启动。

### V5 可加载子集接口定稿

- 默认历史32原生帧、至少1秒预测距离、最多8个物体；物体须在历史中至少8帧有通过当前图像复核的观测。整个历史至目标区间须连续语义复核。固定11实体，特征`entity_features[B,32,11,120]`，`entity_valid[B,32,11]`；前3槽左/右/双手组合，后8槽仅按历史首次出现和ID排序。
- 120维切片：0:4归一化框中心/宽高，4:6过去差分速度，6:12当前框RGB均值/标准差，12:54手部21点二维投影，54:117手部21点三维（最后观测帧左相机系），117二维手有效，118三维手有效，119物体当前人工式种子指示。物体没有伪造手部或3D字段；双手组合只提供包围框/速度，不构造虚假骨架。模型输入字段不含标签和角色文本。
- `tracking_review.json`保存所有已渲染有效物体帧的接受/拒绝范围及中文原因；范围仅覆盖真实有观测帧，拒绝优先。框仅为定性实例二维定位；桌面为可见支撑表面区域，不认证像素IoU、遮挡轮廓或物体6D。种子同样接受审核，不能因手动初始化而豁免错误框。
- 可移植数字包内`sequences/<UUID>/hands.npz`保存timestamps/hands_world/hand_valid/camera_to_world；`visual_observations`保存观测/种子/审核；`dense_reviews`保存监督；`graph_windows`保存固定split窗口。加载实际子集不依赖外部旧V2数组；源RGB视频和证据裁剪用于复核，另列清单范围。
- `training_index.json`冻结输入、窗口、review及来源SHA。独立`training_release.json`绑定index和独立审计/真实数据前向反向报告，仅在`interaction_training_allowed=true`且scope一致时允许非diagnostic加载。范围为`assistant_silver_interaction_forecasting_with_annotated_2d_observations`，不改变`full_joint_training_allowed=false`、`full_annotation_complete=false`和`whole_object_6d_certified=false`。
- 非删失事件必须显式`occurrence_confirmed=true,before_after_observed=true`，保存前/后状态证据；完整有界区间进入预测时间窗才监督事件发生。物理事件无逐帧确定边界时不得写frame_resolved=true。所有V5监督仍为assistant银标，非human金标，val/test仅划分隔离，不称盲测性能。
- 最终准入强制六份非空审核凭据键，缺任意项均拒绝；交付器准入和打包均核全部5段review/observations/metadata/seeds/quality/hands六组冻结SHA，以显式ValueError执行，`python -O`也不能绕过。旧一次预检查报告与源码保存在`admission_history/v1`；最终加载绑定`audit_sources/model_report_crosscheck_v2.json`。

## 2026-10-24 自动化复核流水线 v1

- 新入口`python -m src.annotation.review_pipeline --config <json> --output <独立运行根> [--resume]`。配置仅提供base_root/v3_root/v5_root/source_pool、expected_sequences、workers、max_attempts及显式数值异常阈值；源V2/V3/V4/V5保持只读。运行输出目录不得等于、包含或位于任一源目录内部。自动检查不改label、review_status、training_release或human有效标记。
- 工作图：冻结清单/源划分/词表/重复采集校验 → 全采集逐帧数值与身份/监督检查（并行） → V5既有窗口独立复算及默认Dataset准入验证 → 中文汇总/逐段CSV/优先复核队列。单段失败独立保存、其他段继续，不能因部分失败丢弃其帧数或宣布全量通过。
- 单段接口`review_sequence(base_root, base_row, annotation_path=None, review_path=None, thresholds=None)`返回`sequence_id,split,frame_count,checks,metrics,issues`。每项检查含`code,status(pass/fail/warn/pending),message_zh`；每条问题含`code,severity(error/warning/pending),layer,start_frame,end_frame,entity_id,message_zh,next_action_zh`，范围为原生半开。数值缺测和阈值异常分别统计；速度/骨长/旋转突跳只生成待核项，不自动推翻真实动作。
- 原生帧索引/时间戳、左右手mask/有限值、物体唯一ID/SE(3)、非刚体表面点/类型、几何排除、交互实体引用/范围/正负冲突/删失边界分别检查；只有有实际证据的已保存review才计连续复核覆盖。没有review时全帧为待语义复核；非刚体整物体6D记录不适用，不补造值。
- `jobs.json`记录每项依赖内容SHA、检查代码/配置SHA、尝试数、产物SHA和状态；成功结果只有输入指纹与结果SHA均一致才复用。失败任务可重试，进程中断的running状态下次重新执行；输入执行前后均核对，变化则本次失败。并发锁阻止两个进程同时写同一运行目录。
- `summary.json`分别记录`execution_complete`、`automated_checks_passed`、`full_annotation_complete`与原训练子集准入状态；脚本跑完不等于300段标注完成。`issues.jsonl`保留全部问题，`review_queue.jsonl`把待处理区间切成可执行批次、按错误/未完成排序，含原采集/帧段/原因/建议，不产生自动视觉审核声明。`复核汇总.csv`和`report.html`提供中文定位。
- 队列对相同采集/code/entity/layer/severity/split的相邻或重叠区间先合并再切块，避免同一异常帧重复派工。每个块的`source_issue_lines`绑定`issues.jsonl`一基行号，`messages_zh`及`next_actions_zh`保留与该块相交原项的原因和动作；原始问题记录与标注不合并、不改写。
- 转换删失问题另存`source_record_id,event_type,subject,secondary_object_id,source_reason_zh`，用于回查原事件及其判断依据；缺失字段保留空值，不推断事件身份。队列块引用原问题行，可在一次查看内分别复核多个重叠事件。
- 默认退出码：0=检查执行完且无完整性/结构错误（允许pending/warn），2=检查错误/失败，3=`--require-complete`下仍有未完成标注。默认运行不做模型训练、重建几何或VLM自动确认；这些不能由规则检查通过推定完成。
