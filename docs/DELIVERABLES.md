# 交付物对齐表（Deliverables Alignment）

> 用途：开工前全员对齐——每个阶段**交给下游什么东西、什么格式、谁验收**。
> 格式细节的唯一事实源是 `.agent/memory/03_data_pipeline.md`，本表只做交付关系对齐，不重复格式定义。
> 2026-10-15 Daniel 整理；2026-10-22 按 L40 实测更新。

## 新增工具：300段自动化复核流水线（2026-10-24）

[使用说明与一条运行命令](AUTOMATIC_REVIEW_PIPELINE.md)。入口`src/annotation/review_pipeline.py`，配置`configs/review_pipeline_300.json`，独立输出`automatic_review_300_20261024/`。并行核对全部源数组、手/物几何、交互结构与已有视觉证据；默认Dataset加载和独立复算V5全部窗口。支持失败重试、内容SHA绑定的续跑及重复待办区间归并，保留原问题与事件来源。

实际检查300段868255帧及1238窗口，0项结构错误；3126条数值告警、8404条待确认问题按原生帧范围列入中文报告。80项回归本地/服务器通过。原标注、V5准入与冻结数据包保持不变；自动检查通过不改变下述300段完整标注未完成状态。

## 当前数据交付：V5 可训练交互子集（2026-10-24）

> 验收状态（2026-10-24）：用户确认“复核验收完成”，已记录为本次 V5 交互训练子集（5段、1238窗口）的交付验收。300段完整标注和整物体6D仍为未完成；逐帧标签来源保持原记录。验收凭据与冻结数据包并列保存为 `egodemo_training_v5_300_20261024_user_acceptance.json`。

**已交付可直接由新Dataset加载的小规模交互训练子集；300段完整联合训练集尚未完成。** 完整接手说明见 [TRAINING_V5_HANDOFF.md](TRAINING_V5_HANDOFF.md)。原300段、75任务、868255帧、210/45/45划分及V2/V3/V4保持冻结。

本次持续复核扩至5段1995个唯一原生帧（4段整段看完），含212状态段、42个确认发生物理转换范围；未知/删失不强补。105个当前帧种子经因果重放和97页全部有框帧复核，2357个二维物体帧接受、310个拒绝。整物体6D尚无认证输出；旧场景/部件候选不进入本次Encoder。

| 划分 | 独立采集 | 窗口 | 正监督单元 | 负监督单元 |
|---|---:|---:|---:|---:|
| train | 3 | 592 | 1953 | 1452 |
| val | 1 | 366 | 1172 | 406 |
| test | 1 | 280 | 786 | 1364 |

实际监督并集21类，训练正例12类，9类只出现在val/test，应单列未见类型。66是词表容量；窗口重叠，不能将1238窗口称为1238独立采集。当前是助手银标和已标注二维观测条件下的小规模训练，不是盲测性能数据。

编码`[B,32,11,120]`→实体注意力/真实Mamba→有向图`[B,11,11,66]`。未知mask不参与BCE；对象→对象保留双客体。全部1238窗独立审计通过，真实FP32/BF16正负监督前反向通过，参数未更新。`training_release.json`仅放行交互子集；`full_annotation_complete/full_joint_training_allowed/whole_object_6d_certified`均false。

服务器根`/root/gpufree-data/datasets/ego_hoi/egodemo_training_v5_300_20261024`；同名本地`runs/`镜像。`training_index.json`、`training_release.json`、`交付状态_V5.json`、`逐段连续标注缺口_V5.csv`为当前入口。私有`*_interaction_subset.tar.gz`提供自包含数字输入/标签、源码、中文网页及联系表，逐文件SHA见`PACKAGE_SCOPE.json`；原视频和逐帧PNG留私有服务器。旧V4文件是历史候选交付，不能混用其准入状态。

中文网页只监听127.0.0.1:18791；SSH转发后打开`http://127.0.0.1:18791/review_web/index.html`。剩余866260帧未连续查看，296段未整段看完，全部层完整帧仍0；继续工作必须补真实证据与可靠几何，不能以unknown占位或格式通过替代完成。

## 前版交付：V4连续帧候选与实际复核、Mamba诊断格式（2026-10-24）

**300段的原生帧索引已齐，连续逐帧完整标注尚未完成，正式训练仍未准入。** 本轮root与三个agent分别补逐帧数值、真实连续视觉标注、身份候选、中文网页、Mamba格式和独立审计。保留原300段、75任务、868255帧及210/45/45划分，V2/V3源数据不改。

| 层/检查 | 已完成的实际工作 | 尚缺内容 |
|---|---|---|
| 原生时间轴 | 300段、868255帧逐帧索引、有效mask与数值候选 | 数值索引不表示每帧语义已确认 |
| 连续视觉复核 | 3段1054帧、101页；订书机314帧与手柄/托盘320帧整段看完，扫码片段420帧 | 867201帧尚未连续复核，298段未整段看完 |
| 手部姿态 | 官方左右手原生序列与缺失mask | 2238帧至少一只手无有效姿态；未用插值伪造补齐 |
| 几何 | 156420帧有任意位姿候选、527770帧有任意表面候选 | 340485帧连场景几何候选都缺；“任意场景”不等于当前工件完整几何 |
| 稳定物体身份 | 33个锚点/绑定、20段7236帧连续轨迹候选；2条经连续查看确认的视觉ID别名 | 传播不是逐帧身份真值；4个新视觉绑定中2个仅为部件，整物体对应仍缺 |
| 持续关系与协作 | 91个明确正/负状态段，3876个正值、2603个负值；1050帧至少有一个明确状态 | 未知保持未知；有一条状态不表示该帧全部标注层完成 |
| 转换事件 | 26个物理转换不确定范围；另2个精确照明变化 | 精确物理转换0；照明变化不冒充扫码成功 |
| Mamba | 实际加载866个32帧历史/1秒目标窗口，真实L40 FP32/BF16前向通过 | 全属原train，val/test新增窗口0；现有单客体头新增监督0，图预测头尚未实现 |

866个窗口均含图目标，其中818个有正状态、688个有负状态（两者可重叠），窗口图状态共3071正值/2174负值。标签多数对应尚无整物体几何槽位的视觉实例或双客体关系，保留在`graph_targets`，没有用部件姿态冒充整物体姿态。真实前向为随机权重、0次优化；工程通过不代表交互模型已能学习这些标签。

服务器根：`/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v4_dense_300_20261024`。本地交付镜像在`runs/egodemo_annotation_v4_dense_300_20261024/`；原生裁剪和视频留在私有服务器，基础数值和V3仍依赖原冻结目录。逐文件范围与SHA以`delivery_manifest.json`为准。

| 文件/入口 | 内容 |
|---|---|
| `dense_delivery_report.json`、`逐段连续标注缺口.csv` | 实际汇总与300段逐项缺口 |
| `dense_sequences/`、`entity_grounding/` | 每帧数值/近距离候选、身份候选；不当作已确认交互 |
| `dense_reviews/`、`dense_evidence/` | 实际连续查看范围、明确正负、转换范围及原帧证据 |
| `数据格式与使用说明.json`、`mamba_dense_index.json` | 编解码、加载示例、冻结来源 |
| `dense_review_windows/`、`dense_target_coverage.json` | 866个实际窗口及图/单客体头监督覆盖 |
| `dense_gap_audit.json`、`dense_observables_audit.json`、`entity_grounding_audit.json`、`continuous_review_audit.json`、`dense_window_audit.json` | 五类独立检查，均通过；不认证位姿准确率 |
| `dense_mamba_smoke.json`、`validation_receipt.json` | 真实Mamba前向与回归证据 |
| `review_web/`、`review_web_qa/` | 中文只读审核网页及6项真实浏览器检查 |
| `delivery_manifest.json`、`code_snapshot/` | 文件SHA及可复现源码/文档快照 |

私有增量包为`egodemo_annotation_v4_dense_300_20261024_bundle.tar.gz`，附独立SHA256文件；包含数值候选、标注、窗口、审计、联系表、网页和源码。包内`PACKAGE_SCOPE.json`列明实际文件，原像素逐帧PNG裁剪仍在服务器，V2/V3为外部依赖；本地网页缺少这些裁剪时应使用下述服务器审核入口。交付器先核`audit_lock.json`及全部来源，拒绝审计后改动，检查证据在`delivery_integrity_checks.json`。

网页连接方式：`ssh -p30314 -L 18790:127.0.0.1:18790 root@120.209.70.195`，再打开`http://127.0.0.1:18790/review_web/index.html`。页面可定位每个原生帧、切换证据裁剪、查看正负与未知范围及10748个待审距离变化候选。未渲染帧明确显示缺图，网页浏览不会写回或自动批准标注。旧18788服务保留。

后续交付顺序：按`逐段连续标注缺口.csv`及`dense_annotation_priority.jsonl`补连续证据；先核实整物体/部件身份，再补刚体/非刚体几何与手部缺测证据；复核物理事件边界；在独立新版本重建窗口并审计。遮挡导致无法唯一确定的接触帧保留边界范围，非刚体不补整物体6D。新增全量标注和对齐证据完成前，`full_annotation_complete=false`、`training_allowed=false`不变。

## 前版交付：300段EgoDemo抽样复核与Mamba v3标注层（2026-10-24）

用户要求“完整标注多agent并行做交付 更新文档”。本次交付实际完成的复核、可加载标注和未完成项；不把抽样复核声明为整段逐帧完整标注。

本轮三名子agent与root四路并行并动态分担长序列，完成300段、1666个源区间的实际抽样视觉复核。证据为8330个原生帧文件、899页联系表；复核和最终300份标注审计均0失败。共有2370条明确证据点银标记录，区间操作获支持的实际类型为46种；证据点类型为操作26、关系4、转换0、协作3。这些计数分别描述区间判断、稀疏时间标签和记录条数，不能相互替代。

新增2498条采样可见实体记录（含同形物体集合），未伪造其三维关联；这些不是全部已独立追踪的单体。27个明确误认的既有实体被派生加载器排除几何。排除后仍有161段含刚体/部件6D候选、298段含表面候选，分别有273320候选物体帧、2085808表面实体帧。这是候选覆盖，非独立位姿准确率。

冻结300采集、75原任务、868255帧、53255窗口和210/45/45划分不变。窗口的实际标签覆盖如下；时间未命中已看证据时，目标保持未知。

| 划分 | 全部窗口 | 有有效图边的窗口 | 有当前单客体头银标的窗口 |
|---|---:|---:|---:|
| train | 37158 | 54 | 20 |
| val | 5702 | 12 | 3 |
| test | 10395 | 15 | 4 |

为避免大部分稀疏标签错过固定步长目标，另交付证据点对齐窗口。每个目标都是实际复核点，向前取约1秒预测间隔和32帧历史；保持旧split和旧窗口字节不变。这批窗口正例偏多，不新增独立视频，不作为无偏评测集。实际加载覆盖为：

| 划分 | 补充窗口 | 有有效图边的窗口 | 有当前单客体头银标的窗口 |
|---|---:|---:|---:|
| train | 1150 | 1150 | 284 |
| val | 175 | 175 | 52 |
| test | 328 | 328 | 98 |

62项目标回归通过；L40真实Mamba FP32/BF16前向通过，human标签和旧v2弱监督有效mask均为0，optimizer steps为0。旧候选值仅保留溯源；助手银标独立mask仅确认正例，不能把未知当负例。双客体或无3D映射实体保留graph_targets，当前单客体头尚不实现完整工具→工件预测。

**尚未完成连续逐帧全标注和正式训练准入。** 抽样之间的精确转换、跨遮挡身份、视觉实例与几何对齐及因果观测仍有缺口；full_annotation_complete=false、training_allowed=false。非刚体保持表面点表示，不补造整物体6D。本轮查看了各split标注证据，不能宣称对标注流程盲测或跨被试隔离。

服务器根目录：`/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v3_300_20261023`；基础数值数据仍在同父目录`egodemo_mamba_v2_300_20261023`。本地报告、标注、真实例子与源码快照位于`runs/egodemo_annotation_v3_300_20261023/`。

| 内容 | 新根目录内入口 |
|---|---|
| 中文使用与编解码 | `数据格式与使用说明.json` |
| 逐段四层标注 | `annotations/<UUID>.json`、`逐段标注与缺口.csv` |
| 逐条真实复核及证据 | `reviews/`、`evidence/<UUID>/manifest.json`；原帧文件在私有服务器 |
| Mamba标注索引 | `mamba_overlay_index.json`；适配器`src/mamba/annotation_v3.py` |
| 证据点对齐补充窗口 | `review_windows/index.json`、`review_window_target_coverage.json`、`review_window_audit.json`；显式windows_path加载 |
| 实际数量与审计 | `annotation_delivery_report.json`、`类别与实际标注数量.csv`、`annotation_audit.json` |
| 窗口有效目标与真实前向 | `annotation_target_coverage.json`、`annotation_mamba_smoke.json` |
| 实际NPZ样本、图边和中文解码 | `examples/`；随机权重输出不是标注或效果指标 |
| 校验与源码快照 | `delivery_manifest.json`、`code_manifest.json`、`code_snapshot/` |

已有127.0.0.1:18788网页仍服务原refine版本；本次v3没有覆盖其人工草稿。未训练、Git提交/推送或公开发布数据。

## 真人长序列数据优先（2026-10-22，候选交付、正式训练未准入）

详见 `docs/HUMAN_DATASET_V1_20261022.md`。本轮不训练，实际补充HO-Cap 24个采集/6人/22,729主视角帧，以及EgoDemo 24个新UUID/23,812帧/798.93秒（单条20.13–59.63秒、9任务12场景）；共48个独立采集、46,541帧。低视角6条是重复机位，保留排除原因、不增加样本数。HO-Cap主机位全部逐帧解码/哈希/参考结构审计通过，但真实时基、坐标参考质量等尚未全部核验；EgoDemo只有辅助能力，身份未知不混入跨被试联合基准。

候选按被试冻结train/val/test=16/4/4；不是可训练计数，**正式联合就绪0、training_allowed=false**。完整台账、预览、审计、逐split统计、排除表、源码快照及本地/服务器SHA在 `runs/human_dataset_v1_20261022/`；服务器对应 `/root/gpufree-data/datasets/ego_hoi/human_dataset_v1_20261022/`。本地/服务器各1,081tests＋84subtests通过。真实审计修复了全pending导致全train的分组缺陷，当前使用本地 `frozen_v2/`、服务器 `frozen_server_v2_complete/`，早期失败版本保留但不发布。必须继续补规模/多样性、时间/几何/完整质量证据与因果观测，正式门禁通过后才训练评测，不把已下载或单元测试通过冒充模型精度。

阶段规模目标锁定为120个**验收合格**的独立联合采集，目标约84/18/18，整组隔离优先；已下载联合候选24、合格0/120、合格缺口120，Ego辅助24与重复机位不计数。固定HO-Cap镜像在当前只取官方train策略下有50个候选，另可补26；即使50个全合格仍需其他来源至少70个合格采集。持续计数和来源容量证据为数据根目录 `target_120_progress.json`、`source_capacity_120.json`。本次GitHub仅同步文档，不发布工作区工具源码、测试代码、数据或权重；测试结果对应本地/服务器工作区。

GitHub文档现已通过显式 `miracle-techlink` SSH身份推送并核验远端，旧权限拒绝系默认身份错误。数据发布目标为ModelScope `Liuyue9698/ego_pose_dataset`，仓库读取和Git push --dry-run通过；**此轮尚未上传数据**。目标是public且仍为Apache-2.0模板卡，HO-Cap发布必须保留CC BY 4.0许可和署名，不能按模板重许可；EgoDemo原始数据、预览、记录级清单不得再托管，只发布不可定位记录的聚合说明。候选与正式版本继续分离，120合格目标不因上传改变；预检记录为 `modelscope_preflight_20261022.json`。

## 真人手＋物体未来姿态预测（2026-10-22，工程通过、效果未通过）

详见 `docs/POSE_FORECAST_PILOT_20261022.md`。新增Pose Decoder和因果在线入口，6个真人DexYCB采集436帧，4/1/1隔离，实际48训练/7验证窗口；LSTM/Mamba各3seed真实GPU训练与持位/恒速对照完成，输出+0.5/+1秒手部骨架及物体框。主视频 `runs/pose_forecast_20261022/future_pose_comparison.mp4`，逐帧在线叠加 `future_pose_online.mp4`。**学习模型手误差劣于持位，整体无基线优势；预测激活阶段29.60 FPS、P95 38.55ms，也未通过稳定30FPS。** 这是第三人称录制回放和伪标签一致性，不是GT精度或EgoDemo成功。全仓本地984 tests＋26 subtests，独立产物104项检查通过；不把工程通过转述为预测准确。

## 真人手＋物体同帧在线提取（2026-10-22，明确排除机械臂）

详见 `docs/HUMAN_HAND_OBJECT_LIVE_20261022.md`。交付 DexYCB 真人右手拿碗的44帧同帧预测视频：21点手部骨架＋FoundationPose框/XYZ轴，未使用手部/物体位姿GT。最终三次同序列复测均44/44联合输出，稳态生产41.409–41.524 FPS，环路P95 27.056–27.967ms；每轮仍有2/43慢帧，非硬实时。完整回归876 tests＋26 subtests通过。**这是第三人称录制RGB-D在线回放，不是EgoDemo第一人称完成，也不是已接通实时摄像头；覆盖率非准确率。** 本地主视频 `runs/human_pose_live_20261022/human_hands_foundationpose_30fps.mp4`。之前机械臂样例仅保留为环境测试，不作为真人手联合交付。

## 真实 6D 后端跑通（2026-10-22，公开机械臂样例 737/737）

详见 `docs/FOUNDATIONPOSE_REAL_RUN_20261022.md`。已补齐独立环境、官方权重、原生 CUDA/C++ 扩展及真实 RGB-D/mesh/mask/内参，在官方 `mustard0` 实拍序列完成 737 帧非 mock 注册/跟踪，推理 34.005 秒，含导出总计 82.113 秒。交付完整视频、3 张点云/物体同系三维图、NPZ、CSV、溯源与日志；服务器全仓 797 tests + 26 subtests 通过。**100% 是运行覆盖率，不是精度；独立 GT 未取得，位置/旋转/联合准确率仍 null。** 这是公开样例，不放行 EgoDemo 手—物融合，不改写此前 mock 拒绝结论。下一步补实际刚性目标输入与独立参考。

## 6D 位置/姿态专项验收（2026-10-22，旧 8 份交付 0/8 通过）

详见 `docs/POSE6D_ACCEPTANCE_20261022.md`。现有 8 份 pose 文件共 5,214 视频帧 / 10,428 物体帧，格式 8/8 过但真实交付 0/8：全部 mock、单位旋转、固定 Z=1；真实位置/姿态准确率未评估。新增逐帧 CSV、拒绝诊断图和环境证据位于 `runs/pose6d_acceptance_20261022/`（服务器对应 `datasets/ego_hoi/`）。strict 吞异常与 mock 缺少标记已修复，真实后端/资产未补齐前不放行。

## 手部/物体 6D 融合验收（2026-10-22，未通过）

详见 `docs/FUSION_ACCEPTANCE_20261022.md`。两段真实视频共 7,164 帧，交付 18 张手部/相机多视图 PNG、2 个连续预览 MP4 和全帧 CSV；双手有效覆盖 99.7487%，**不是精度成功率**。真实物体 6D 与独立参考缺失，整体 blocked、联合成功率 null，暂停正式训练。产物位于本地 `runs/fusion_acceptance_20261022_review/` 和服务器 `datasets/ego_hoi/fusion_acceptance_20261022_review/`；本地/服务器回归均 723 tests + 26 subtests 通过。

## 第二轮实测交付（2026-10-22）

详见 `docs/VALIDATION_20261022_PHASE2.md`：本地/服务器均 412 项测试与 26 个 subtests 通过；10 段 / 18,079 帧完整解码，0 errors / 12 诊断警告；4 组 GPU 压力验收与 6 组真实手腕辅助预测完成。LSTM/Mamba 对持位基线提升仅 1.71% / 0.26%，不作为正式 HOI 效果。服务器证据、checkpoint 与 42 条质量复核队列位于 `datasets/ego_hoi/validation_20261022_phase2/`。

下一阶段先复核异常与相机重投影，再补两段物体轨迹和人工三头标签；门槛未通过前不扩大模型、不把原文事件或缺失标签自动当 `none`。

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

### 2026-10-22 正确性修复与标注版试点

- 已修复⑥实体/时间轴混排；新增真实类别 macro-F1 与三头联合 accuracy。旧 `acc_macro` 仍仅表示三头 accuracy 的均值，不能当作类别级 F1。
- ⑤默认 `coordinate_mode: mixed`，禁止图像归一化手部与米制物体之间的几何运算。只有确认同一相机系后才使用 `camera`。
- ④支持 `--horizon-seconds 0.5/1/2`（每次指定一个值）与采集 UUID 分组；⑦新增目标帧、预测间隔、跨机位 split 泄漏校验。接口见契约 2026-10-22 增补。
- 现有 Raw 的 82 个 UUID 全部匹配到标注版：61 个 EgoProStandard、21 个 EgoProStandard-body。已按任务均衡选取首批 10 个 UUID，下载 40 个 LFS payload（约 600 MiB），逐文件校验 SHA-256，复制 30 个 JSON 元数据文件。
- 首批数据为 **18,079 帧、46 段原文语义事件**；10 段的 parquet / manifest / 视频头帧数一致。视频头检查不等于全视频解码或完整标注验收。
- 数据盘入口：`datasets/ego_hoi/annotated_pilot_20261022/selection.json`、`download_receipt.json`、`asset_validation.json`；真实 payload 位于 `datasets/ego_hoi/raw/annotated_pilot_20261022/`。
- 真实暂存层使用 `src/dataset/import_egodemo.py`，保留世界坐标、原始与坏帧掩码后的手部、时间戳和原文事件；不伪造七类 HOI、左右主体、物体 ID 或 6D 真值。下一交付仍是物体轨迹、审核标签和明确的世界→相机约定，之后才能正式训练。
- `annotated_pilot_20261022/staging/` 已完成 10 段真实导入，`staging_validation.json` 记录验收；`annotation_review.csv` 为 46 行待审核事件，三头标签留空，不允许直接当训练标签。该任务均衡小批次用于联调，不是可报告效果的充分训练/测试划分。

复现选择与导入（数据大文件仅放持久数据盘；导入输出目录必须是新的）：

```bash
ROOT=/root/gpufree-data/datasets/ego_hoi
python -m src.dataset.egodemo_manifest \
  --dataset-root "$ROOT/raw/EgoDemo" --raw-root "$ROOT/raw/egodemo_subset" \
  --limit 10 --output "$ROOT/annotated_pilot_20261022/selection.json"
# 按 selection.json 的 fetch_files 用已授权 HF/LFS 客户端定向获取 payload，勿全量 git lfs pull。
# xw_dataset 环境额外需要 pyarrow==19.0.1；不向 base 安装。
python -m src.dataset.import_egodemo \
  --manifest "$ROOT/annotated_pilot_20261022/selection.json" \
  --dataset-root "$ROOT/raw/annotated_pilot_20261022" \
  --output-dir "$ROOT/annotated_pilot_20261022/staging"
```

### ① fish → 给 ②③
- **交付物**：`clips/<clip_id>.mp4`（~20s/段）+ `clips/clips_index.csv`
- **验收人**：daniel（`tests/check_contracts.py clips`）+ ash/panda 各抽 1 段试跑
- **验收标准**：csv 每行可被 ffprobe 复核（时长/帧数一致）；clip_id 全局唯一；<10s 残段已丢弃并备注
- **前置依赖**：EgoDemo 已获访问权；15 场景子集已下载（328 视频，19.47 GiB）
- **当前状态**：2 个真实视频已 `--reencode` 成 6 clips，0 错误；两段 10–18s 残段为预期 warning。下一步批量切分与多机位整理

### ② ash → 给 ④
- **交付物**：`hand_pose/<clip_id>.npz`（left/right (T,21,3) + conf (T,2)，NaN 规则）+ meta.json
- **验收人**：daniel 契约校验 + xw 试读与可视化抽检
- **验收标准**：未检出帧整帧 NaN 而非 0；左右手判定经人工抽检不互换
- **前置依赖**：① clips 已就位；mediapipe 定格 0.10.21
- **当前状态**：2 个真实 clips / 1200 帧通过校验（0 错误），左/右/双手检出率 75.2% / 83.8% / 63.3%；meta 推荐键与可视化抽检待完成

### ③ panda → 给 ④
- **交付物**：`pose6d/<clip_id>.npz`（object_ids / poses (N,T,4,4) / bboxes (N,T,4)）+ meta.json（注明 depth_source）
- **验收人**：daniel 契约校验 + xw 试读 + 可视化抽检（轴向不飘）
- **验收标准**：同一物体跨帧 id 一致；poses 未检出为 NaN、bboxes 未检出为 -1（以契约为准）
- **前置依赖**：① clips 与 L40 已就位；还需相机内参、深度、mask 与物体 mesh/重建路线
- **当前状态**：mock 后端只用于合成链路验证；**真实 FoundationPose/深度/mesh 后端尚未跑通，是当前主要缺口**

### ④ xw → 给 ⑤⑥
- **交付物**：`samples/{train,val,test}/*.npz` + `dataset_index.json` + `vis/<clip_id>.mp4`
- **验收人**：daniel 契约校验 + arc/caron 试读一个 batch
- **验收标准**：split 按 source_video 隔离；subject/object_slot/object_id/interaction 标签键齐全；obj 特征 13 维顺序正确（pos3+rot6d+area1+vel3）
- **前置依赖**：真实 ② 已有首批输出；真实 ③ 待交付；interaction_type 启发式规则需按真实分布校准
- **当前状态**：313 个合成样本 0 error / 0 warning，可供⑤⑥联调；真实样本、标签分布与 ≥5% 可视化是下一步

### ⑤ arc → 给 ⑥
- **交付物**：`src/encoding/` 的 `build_encoder(cfg)` / `build_decoder(cfg)`，import 即用
- **验收人**：caron（训练脚本 forward/backward）+ daniel（`python tests/test_encoding.py`）
- **验收标准**：支持可变物体数（padding mask）；输出三路 logits；NaN/mask/gaze 接口单测通过
- **前置依赖**：④ 样本格式与接口契约已冻结
- **当前状态**：10 项单测及 LSTM/Mamba 集成冒烟通过；下一步真实样本 overfit 与手/物坐标单位联调

### ⑥ caron → 给全组（训练结果）
- **交付物**：`src/mamba/` 训练脚本 + `configs/*.yaml` + `runs/<run>/best.pt` + `metrics.jsonl`（或 tensorboard 日志）
- **验收人**：daniel（冒烟跑通）+ 全组（实验报告）
- **验收标准**：配置驱动；bf16；checkpoint 含契约版本；附 LSTM 基线开关
- **前置依赖**：④⑤ 与 L40 已满足合成冒烟；真实训练等④的真实 samples
- **当前状态**：sm_89 wheel、FP32/BF16 前后向、3 epoch 训练与 checkpoint 重载通过；train loss 4.3740→1.4405，峰值 699 MiB。合成标签单一，准确率不作效果结论

### ⑦ daniel → 给全组（质量闸门）
- **交付物**：`tests/check_contracts.py`（①–④ 产物校验 CLI）+ `tests/test_encoding.py` + 冒烟产物
- **验收人**：全组使用
- **前置依赖**：契约已冻结
- **当前状态**：25 项 checker 自测、合成①→⑥、真实①②与 Mamba GPU 冒烟已通过；③真实后端接入后重跑全真实链路

## 全员对齐要点

1. **只认契约**：上下游对格式有分歧时以 `03_data_pipeline.md` 为准；要改格式先改契约并 @ 上下游（AGENTS.md 铁律 1）
2. **合成与真实分开表述**：合成/mock 只证明代码链路可运行；真实实验必须来自真实③④产物，并检查标签分布和可视化
3. **外部阻塞已解除**：EgoDemo 访问与 GPU 均已就绪；当前瓶颈是③真实后端、真实标签质量，以及 ModelScope 数据集仍为空壳
4. **完工即更新** `05_progress_log.md`
