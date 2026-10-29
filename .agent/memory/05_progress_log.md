# 05 — 进度日志

## 2026-10-24 Codex 多Agent — 300段自动化复核流水线交付

按用户“写一条pipeline自动化复核”实现并部署`src/annotation/review_pipeline.py`、逐段`review_checks.py`及V5隔离复算`review_training.py`；配置`configs/review_pipeline_300.json`，文档`docs/AUTOMATIC_REVIEW_PIPELINE.md`。root负责编排/来源清单/中文报告，三个既有agent分别负责数值与标注检查、训练窗口检查、独立编排测试及最终验收。契约先更新03；V2/V3/V4/V5源、原标注、训练准入和冻结包保持原样。

支持4线程逐段检查、单段失败隔离和重试、内容/代码/配置SHA绑定的断点续跑、嵌套证据页变化重检、子产物损坏失效、执行前后输入核对和输出目录/并发锁保护。检查真实原生时轴、左右手mask/速度/骨长、唯一物体ID/SE(3)/非刚体表面点、几何排除、交互引用/正负冲突/删失及已有连续视觉证据。V5通过独立临时目录复算，避免旧审计器覆盖冻结报告。新队列对相同检查/实体的重叠与相邻范围合并后切块，通过原问题行号保留不同转换事件来源，不改原事件或源问题。

最终服务器全量执行退出0：300独立采集、868255帧、23清单/划分检查通过，900份逐段数值/元数据/源事件文件核验；0结构错误、3126数值告警、8404待确认项。原11530问题完整保留，归并并按最多300帧拆出51758条待办。默认Dataset和独立审计均遍历全部1238窗口，train592/val366/test280，六项训练检查通过。随后同代码实际续跑命中全部302任务，`--require-complete`按预期退出3；没有以数值检查通过解除全标注缺口。

最终针对性测试本地/服务器均80 passed（checker34、training15、pipeline31），服务器JUnit保存`unit_tests.xml`。独立验收核300段/1205源绑定、905小源实际SHA、全部队列范围覆盖且同组0重叠、8条删失上下文、304个HTML链接（含300段报告）及12冻结控制文件末态SHA；冻结包此前实际SHA亦未变。独立凭据`independent_acceptance.json` SHA `6f2283996b38be8f46d64f167d171e8cc02a4ee33b4ba5723afeac16b244c353`，绑定最终summary/jobs/manifest/全量摘要/JUnit；后续再运行须读取新的输出SHA，不把旧凭据延用。

服务器产出`/root/gpufree-data/datasets/ego_hoi/automatic_review_300_20261024`，本地镜像`runs/automatic_review_300_20261024/`。入口`report.html`、`复核汇总.csv`、`review_queue.jsonl`、`summary.json`；`full_run_summary.json`保留全量exit0，当前summary保留严格续跑exit3。文档、04资源、交付物表和接手SOP同步更新。SSH实例短暂不可用后恢复，现有权限连接正常，未新建公开服务。

本次完成的是自动复核工具与全量执行：连续语义实际已复核1995帧、866260帧仍待复核，340485帧无可用场景几何候选，整物体6D与全层标注仍未认证。转换52总记录=42确认物理范围+8删失物理范围+2照明信号，不把总数当作精确物理事件。未新增人工批准、未执行优化或正式训练、未commit/push；现有V5子集交付验收保持原范围。

## 2026-10-24 Codex — 用户确认V5交互子集复核验收完成

用户原话“复核验收完成”。按紧邻已交付范围，记录为V5交互预测训练子集（5段、1238窗口，train592/val366/test280）的交付级验收完成。未把该声明扩展成300段全标注完成或逐帧人工金标；full_annotation_complete、full_joint_training_allowed、whole_object_6d_certified仍false。

新增包旁验收凭据`runs/egodemo_training_v5_300_20261024_user_acceptance.json`，SHA `332c0d36d1f08f23d24e751320ff615af83fd700e44104f5d35ce73873eedcb6`，已同步服务器数据父目录。凭据保留用户原话，并绑定包SHA5c4f7d35...、index4b162d65...、release58b8da46...和独立解包审计；已核对服务器包与release校验和未变。

已更新docs/TRAINING_V5_HANDOFF.md、DELIVERABLES.md及HAND_OBJECT_HANDOFF_SOP.md并同步中央仓库。冻结数据包、包内文档/网页快照、标签、准入凭据均保持原样，本次记录独立保存；无新增训练运行。300段完整几何/交互缺口继续保留，未因交付级验收改变。

## 2026-10-24 Codex — V5实际可训练交互子集、独立封包交付

用户继续要求推进到交付态。root与三个agent完成V5的真实连续复核扩展、当前帧二维实例观测及全观测质检、有向图Mamba、非零监督、独立准入、中文网页、文档和数字可移植包。结果只放行小规模交互预测子集；不把5段替代原300段完整目标。V2/V3/V4、原300采集/75任务/868255帧/210-45-45划分保持冻结；未做正式训练、提交/推送或公开发布。

实际连续查看累计1995个唯一原生帧、5采集：原train3段1054帧，新增val电水壶536帧和test红卷/贴片/信封405帧；4段整段看完。212状态段展开8430正值/6320负值，1991帧至少一条明确状态；42个确认发生有界物理事件、8个删失范围、2个精确照明信号，精确物理事件帧仍0。所有为助手银标，不冒称人工金标准；原train范围事件的V4依据与旧边界保留。

105个当前帧种子经连续前缀光流重放；仅保留原种子存活点、不补点复活、最长30帧，拒绝未来种子或离线6D输入。5段全部97页/1136有框原生帧的2667物体观测实际叠框审核：2357接受、310拒绝（scanner487/20，val355/52，stapler421/72，test522/61，controller572/105）。无框帧不补，桌面仅可见支撑区域；二维框不是整物体6D。

新Dataset选历史32帧内至少8个有效观测的对象，过去首次出现及ID排序，整个历史到未来至少1秒目标均在连续复核区内。输出实体特征[B,32,11,120]，真实Mamba输出有向图[B,11,11,66]。标签有明确正负mask，未知不作负例，物体→物体保留方向。全部1238窗独立重算/实际加载通过：train592窗1953正/1452负，val366窗1172正/406负，test280窗786正/1364负。实际类型并集21（8操作/6关系/3转换/4协作），训练仅12种正例；9类仅在val/test，不用于普通闭集效果声称。

真实L40、torch2.6.0+cu124、mamba_ssm2.2.6.post3，4个真实train样本25已知cells（15正10负）。FP32/BF16 loss约0.69573/0.69571，Mamba骨干梯度L2约1.2717/1.2698且全部finite；未知logit直接梯度0、future labels/metadata不影响输入、对象置换最大误差1.788e-7；0 optimizer steps、参数未变。graph12+tracking14、dataset8、independent audit32项针对性测试通过；这些工程结果不表示模型准确率。

准入复审实际发现并修复：release缺/空/部分evidence键可跳过校验，及只检查各split首窗会漏掉审计后修改的后续采集数组。现loader强制六份凭据，admit/package显式ValueError检查全部5段六组SHA，-O不可绕过。8项真实临时副本拒绝测试通过；旧预检查源码和报告留admission_history/v1。修复后重新实际Mamba运行，最终smoke SHA a0660cd15910cbd58cfa11d2d69d7dca3893df66796ffa74dc2747fda4f58c06；独立v2交叉核对与窗口审计一致。最终index SHA 4b162d6595f7b489a078d86b096732a2060300cc3ea5d92296463fccfcb179e6，release SHA 58b8da4620b027eecb92e35fddde98af07525506b68d66f2a34f085b2cec5983；仅interaction_training_allowed=true，full_joint/full_annotation/whole_object_6d均false。

因果6D实测单列pose6d_feasibility：手柄首帧原双目标定像素中位误差0.623px，但纯可见表面各8匹配点低于原12点门槛；bbox混手/背景虽红19白13点，只在前32帧产生红5/白1帧未认证PnP候选，未进入训练。后续需当前表面分割、更密可靠双目匹配和独立位姿验收，不降低门槛冒称补齐。

服务器根 /root/gpufree-data/datasets/ego_hoi/egodemo_training_v5_300_20261024；本地同名runs镜像。交付状态_V5.json、300行逐段连续标注缺口_V5.csv、docs/TRAINING_V5_HANDOFF.md、DELIVERABLES/HAND_OBJECT_HANDOFF_SOP/03/04已更新。剩866260帧未连续查看，296段未整段看完，全部层完整帧仍0；2238帧至少一手缺失、340485帧无任意场景几何候选仍保留原缺口。

私有包 egodemo_training_v5_300_20261024_interaction_subset.tar.gz：282180054bytes，SHA 5c4f7d35b589bbe41817767726b5bd49cbe525257b9a358d1c56b008164d8e7e。独立解包验收441普通文件逐SHA/大小通过，无软链/绝对路径，新的cwd仅PYTHONPATH=code_snapshot可默认加载3split及全部5采集实际样本。包旁.audit.json SHA 83c42ed4ee7f32e7d36ccf12f936dad031fe616c16638d0e987219872c462280。包自含数字训练所需输入/标签、代码、联系表及网页；原视频与完整逐帧PNG仍私有外部审核依赖，少量原分辨率QA图包含。

中文只读Web127.0.0.1:18791/review_web/index.html，最终页面绑定release和最新loader，Playwright实际5段导航/20链接/拒绝及缺测帧/390px手机布局通过，0控制台或请求错误。QA SHA30ab6ba0de4c3e468834b98ef123e0eb84441a44928c0da8d593695e0f821899；旧18790/V4不变。下步仍为300段按真实证据推进完整标注/刚体与非刚体几何，扩充训练实际类型与独立采集；本次小子集交付不等于该最终目标完成。


## 2026-10-24 Codex — V4连续原生帧补标、三路并行与私有诊断交付

用户继续要求找出“连续逐帧完整标”缺口并推进交付。root与三个agent实际完成300段868255帧逐帧数值候选、身份候选、连续视觉复核、中文网页、Mamba适配和独立审计。V2/V3、300采集/75原任务、210/45/45划分与旧53255窗口保持冻结；本轮未训练、提交/推送或公开发布数据。

逐帧数值已处理868255帧，重新计算3349578个有效帧-物体-手距离观测；0.03米阈值产生10748个待复核变化边界，只是接近候选。156420帧有任意位姿候选、527770帧有任意表面候选；340485帧无场景几何候选，2238帧至少一只手姿态无效。有效数值、任意场景几何与当前交互工件完整几何分别计数，不补造缺失值。

实际连续查看1054个唯一原生帧、101页：root完成订书机整段314帧和游戏手柄/托盘整段320帧；agent完成扫码段[0,260)及[1500,1660)共420帧。订书机旧源文本已有“关抽屉”区间，本次补的是其连续帧证据，不是新发现遗漏的文本。当前仍有867201帧未连续复核，298段未整段看完，全部层完整标注帧仍为0。

连续银标91个状态段，展开3876个明确正值、2603个明确负值；1050帧至少有一个明确状态，不代表各层齐全。物理转换26个不确定范围、精确物理边界0；扫码照明有2个精确视觉信号，不能等同扫码成功。全部为实际助手查看所得银标，未伪造human审核；遮挡、初始已抓持和未看范围明确保留。另有2条整段同一托盘的视觉ID别名，未据此合并任何3D轨迹。

实体处理覆盖全部300段：33个锚点形成33个候选绑定，20段7236个原生帧/9092个实体帧的同点键连续传播。4个新视觉像素锚定中，扫码枪头与纸箱面单2项仅是component_of；传播始终estimated、silver=false，缺帧/丢点/跳变即断，不伪造整物体6D或跨遮挡身份。

V4索引SHA `67c6d066a1d3379bf59e2bf68fb1451b4b0821da809d8dc5eaeb8ed8ed9531c5`。从1050个明确状态目标中排除91个无满1秒锚点、93个不足32帧历史，导出866个诊断窗口，全部仍属原train，val/test新增窗口0。实际866窗均有图标签，818窗有正状态、688窗有负状态，图状态共3071正值/2174负值。新whole视觉实体与历史三维槽位未对齐、双客体关系也无法压入旧单客体头，故新增单客体头有效监督0；现有模型尚未实现新增图预测头。

真实NVIDIA L40、mamba_ssm 2.2.6.post3、PyTorch 2.6.0+cu124，batch3的FP32/BF16前向全部有限，event_logits=[3,9,5,66]，随机权重、0 optimizer steps。五类独立审计全部通过；最终针对性回归57 passed in 1.79s，中文网页6项真实浏览器检查通过、0 JS错误。交付器3次干净检查通过、16种审计后改动均拒绝，含实际NPZ与边界JSON哈希；独立`audit_lock.json`固定5审计及index，打包器不能自行重写锁放行。

服务器V4根`/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v4_dense_300_20261024`；本地镜像同名`runs/`目录。`dense_delivery_report.json`与300行`逐段连续标注缺口.csv`为当前进度入口，中文说明、实际样本、源码、五审计与前向/浏览器证据均随交付保存。私有增量包`egodemo_annotation_v4_dense_300_20261024_bundle.tar.gz`使用`PACKAGE_SCOPE.json`列明范围；逐帧原像素PNG和原视频仍在服务器，基础数值及V3仍依赖冻结目录，不冒称自包含完整训练集。

中文只读审核服务为`127.0.0.1:18790/review_web/index.html`，SSH转发18790访问；可逐帧查看明确正负/未知、裁剪、边界范围及10748候选。未渲染帧不复用旧画面，浏览不产生人工批准。旧18788服务保留。下一步仍是按逐段缺口补连续视觉证据、稳定整物体/部件对应与几何，再核物理边界并构造下一版窗口；几何因果性和独立准确率未完成。`full_annotation_complete=false`、`training_allowed=false`，不能把此阶段交付描述为300段连续全标注或可直接正式联合训练。

## 2026-10-24 Codex — 四路完成300段源区间抽样复核，v3交付与真实Mamba验收

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

- 私有结果根`/root/gpufree-data/datasets/ego_hoi/egodemo_annotation_v3_300_20261023`，本地对应`runs/egodemo_annotation_v3_300_20261023`。实体/交互/证据审计分工后，由root和完成较早的实体组接力长序列；每段唯一负责方及SHA见reviews各冻结记录。最终应用顺序实体→交互→整合→独立审计。reviewer均为assistant，未冒用用户旧样例批准。
- 原始帧、提案、复核图片、组件、最终标注和冻结split/windows均绑定SHA。base index `ae3c08b2590966d949b20ac1224c6b176a63e5b3f91c5355caf8a6477370c981`；overlay `6f8415ec766f159355feafb042ceb2ef0552ce2c45dabf8af9001da7676ee56a`。详细数量见交付报告、窗口覆盖报告与逐段缺口CSV。
- 修正了名词/形容词被当动作、原文颜色/手别与可见事实不符、镜像重复、涂改液瓶误作白笔、工具名误作操作工具、多同形实例强并等问题。区间存在结论不扩成逐帧正例；开始/结束边界不确定则删失，不自动给首次接触/最终脱离标签。
- `.agent/memory/03_data_pipeline.md`、`04_resources.md`、`docs/DELIVERABLES.md`、`docs/HAND_OBJECT_HANDOFF_SOP.md`已更新。18788仍是旧refine中文审核台，本次标注层通过JSON/CSV及真实输入样本交付；未公开、训练、提交或推送。

## 2026-10-23 Codex — 三agent并行完成300段Mamba v2格式交付与真实GPU前向

- 用户明确要求多个agent并行加速、交付Mamba训练数据格式；分工为全量导出、加载器/编码解码与真实Mamba前向、冻结划分与独立全量审计。完整私有数据在`/root/gpufree-data/datasets/ego_hoi/egodemo_mamba_v2_300_20261023`，本地报告/真实样本在`runs/egodemo_mamba_v2_300_20261023`。没有执行训练、提交、推送或公开EgoDemo。
- 实际导出300序列、868255原生帧、53255窗口；NPZ共1043249766字节，加metadata/events共1063253972字节。162段含过滤后刚体/部件6D，299段含未被标定拦截的表面点；274791候选物体帧，55098485个表面点观测，267757点键。1666源区间对应2665操作候选；总9021事件候选（操作2665/关系3834/转换2496/协作26），人工金标准仍0。
- 按scene/environment连通组冻结53/12/12组、210/45/45采集。此前已见34采集关联16组129段全部train；窗口37158/5702/10395，原生帧605865/93604/168786。被试均缺失，不宣称跨被试隔离；任务可跨集合，不能称盲测基准。split SHA256 `2aaa03ab8686a7580bb88f34baa25038074119842545e335cb2141fb25072dcc`；dataset_index SHA256 `ae3c08b2590966d949b20ac1224c6b176a63e5b3f91c5355caf8a6477370c981`。
- 新`src/dataset/export_egodemo_mamba_v2.py,egodemo_v2_split.py,audit_egodemo_mamba_v2.py`和`src/mamba/egodemo_v2.py,egodemo_v2_smoke.py`，独立`configs/egodemo_mamba_v2.yaml`。W32、stride16、native timestamps预测1秒，N8/P16；序列世界系，加载至最后观测左相机共同锚点。slot/point仅按观测窗选取；NaN、missing、weak-positive、gold监督与padding各有mask。
- 词表容量49操作+6关系+6转换+5协作=66；实测有候选49/3/2/1类，不能用容量替代已标注类别数。事件张量(B,9,5,66)，未知客体8、未知主体4；不是旧7类接口。全部9021记录第二客体均-1；当前模型头仅预测单客体，原JSON第二客体字段保留，不声称已实现工具→工件双客体预测。
- 编码(B,32,11,128)转(B*11,32,128)真实Mamba，输出手(B,2,21,3)、物体(B,8,4,4)、表面(B,8,16,3)、交互(B,9,5,66)。旋转6D解码SO(3)，事件sigmoid多标签中文解码；output必须结合prediction masks，不能把非刚体/无效槽数值当真实6D。
- 新增接口22项回归通过；独立全量300/300采集、53255所有窗口、6420源文件SHA、55098485表面观测逐点源对应检查通过；每采集实际加载1个丰富观测窗口，历史slot/point、目标同键、坐标与pending mask共300窗口通过。审计只证明结构与来源一致，独立准确率仍null。
- L40上使用现有`caron_mamba`环境（torch2.6.0+cu124、mamba_ssm2.2.6.post3）4真实窗口FP32/BF16前向均有限，0 optimizer steps；随机权重没有效果指标。真实例子`examples/sample_window.npz/.json`含中文物体蓝色篮子/包裹、32有效物体位姿输入、704表面点输入和false金标准事件mask。最终smoke SHA256 `8fb7e2aa0569bf8641137403293abbaa809cc9208af7e0e011ad95b5593488dc`，sample NPZ SHA256 `3e1dafd7cdcc9fb0b151c1283347372b1d268d310e28d05dd63bdb6b9509e17b`。
- 中文`数据格式与使用说明.json`、`类别与候选数量.csv`与交付清单配套。format_ready=true，formal_training_ready=false；跨遮挡身份、物理关系/边界、少样本类别、因果输入和独立精度仍需完成。`diagnostic=True`仅用于结构检查，不能据此放行非因果离线几何或未审核标签训练。
- 同轮网页修复geometry包装后cache_clear调用错误，最终隔离浏览器7项检查通过；生产18788与QA18789已detach，生产人工草稿0，测试草稿在web_qa_final不入标注。补丁记录`runs/egodemo_refine_300_20261023/web_delivery_patch.json`，冻结数据与原报告未变。

## 2026-10-23 Codex — 用户要求“补齐”后的第二轮自动补算与工程验收完成

- 独立私有产物目录：`/root/gpufree-data/datasets/ego_hoi/egodemo_refine_300_20261023`，本地对应`runs/egodemo_refine_300_20261023`。同一冻结300采集，不删除旧结果，不训练、不公开EgoDemo、不提交Git。
- 新独立venv `/root/gpufree-data/venvs/egodemo_refine_20261023`，继承panda_human_live只读site-packages；安装hydra-core/timm/gdown/decord，不修改团队环境。官方C-FastFoundationStereo权重SHA256 `7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214`；SAM2.1 base-plus HF revision `b7320756a13354e7530a63935656d35b2f91a290`，safetensors SHA256 `2012733a0de5d03efd1bba550a2847c4551be9ef2e0d497c83074df66189f780`。FFS源码通过官方raw的代理逐文件下载，保留研究许可证和哈希清单。模型config缺少normalize开关，按官方代价体默认余弦归一化兼容，不改权重。
- `refine_backends.py`实现真实双目校正、左右一致性校验、恢复米制原相机坐标，以及SAM2视频推理。纸箱/毛巾/白板笔三个实际样例已跑；纸箱31个采样时刻23个PnP候选，毛巾折叠放下后仍有分割和深度，白板笔分割能延续但多数时刻6D不可得。不是独立精度验收，不能据此称全量通过。修复SAM2无帧会话num_frames=None导致首帧种子被忽略的问题，多物体新输入集合也做了保留。
- `refine_seeds.py`批量重建原1666事件采样帧中的表面参考；`refine_tracks.py`同一物体参考系双向追踪原生帧，使用左右光流一致性/标定三角化/刚性拟合，失败保留NaN。`rigid_fit.py`新增等价批量RANSAC，小矩阵测试与原算法一致，局部基准约2.9倍加速；未放宽几何阈值。
- `refine_associations.py`将有同帧表面重合且参考变换稳定的轨迹关联并对齐参考系；相邻静止物体不能仅凭相对不动合并。产生原生帧接近表面、共同运动、抓持及双手协作提案，全部pending，接近不是确认接触。纸箱/书本的运输刚体假设限于对应原文区间；复数多物体与透明表面歧义不能放行整物体6D。
- 两轮目的性视觉抽查分别检查长轨迹和近手轨迹，发现遥控器误认回形针、鼠标误认钉子、背景标签/海报误认工件、多盖子/回形针堆混为单物体等问题。`visual_quality_overrides.json`记录并过滤明确错误；胶带候选实际为胶带切割器壳体，改为刚性部件候选。抽查不是随机准确率，也不是整段人工批准。
- 新私有中文审核服务仅127.0.0.1:18788（旧18787保留），预填身份/交互提案；可保存原生像素多边形种子，关联轨迹变更后使旧参考对齐信息失效。渲染使用过滤并对齐后的objects.npz坐标轴。补译70个原英文展示名，含混标签明确保留“需拆分实例/原标签混合”。浏览器隔离QA的圈选、保存重载、训练门槛、移动布局等5项已通过；测试草稿在web_qa独立目录，生产人工通过仍0。
- 当前批量进度以`refinement_report.json`和`第二轮补全进度.csv`为准，快照：双目参考290/300，原生轨迹约85/300；批处理仍在运行。跟踪主进程398525（需复核PID），关联v3最终进程407052（需复核），审核服务407195（需复核）；新SSH ControlMaster路径`/tmp/egodata-joint-continue.sock`。如恢复工作先查看进程与receipt，不重复处理已完成采集。
- **尚未完成全部标注**。不能把背景物体6D数量或自动身份/边界提案当作实际交互目标标注完成。`full_annotation_complete=false, training_allowed=false, independent_accuracy=null`；后续必须等批算结束、完整数值审计、明确剩余目标/几何/边界缺项，再报告实际完成度。

- 本轮后续：全部300采集已做3个实际图像时刻的像素级标定抽查，25段需复核（不是时间戳问题）。拟合相对外参且保持源基线尺度；20段通过1个原留出帧+10%/45%/90%额外独立帧验证，最大旋转修正1.327度、额外帧极线中位误差最大0.667原生像素。校正是图像估计候选，不是独立位姿真值；另5段证据不足或验证失败，不放宽阈值。原始标定不改写。两段面粉任务校正后参考观测由0/1提高到28/17，原先6段无参考的缺口已补上。
- 手工补救6段分别保存实际视觉多边形、源预览SHA、刚性部件/非刚体范围，空模型分数保持null。订书机底座真实重算得到424部件位姿候选帧；恢复过程依据参考SHA和标定SHA决定重算，不能仅修改状态。自动追踪新一轮300段已接近结束，最终脚本随后会对校正的20段重新重建和追踪。
- 第三轮另外15段目的性抽查已完成，总37段、43条轨迹抽查，27条负面判断（含整物体6D范围拦截）。识别腕部采集器、背景打印设备、多物体堆误检；白板板面和壶身校正为刚性部件。质量判断现按不可变source_observation_id绑定，防止重新跟踪编号变化把拒绝应用到另一物体。
- 新`semantic_drafts_refined`保留旧中文/原文/原生区间，修复106条明确操作动词漏匹配，1666原文区间全部映射；49种实测源语义操作候选（不是人工确认的物理关系类型）。新提起/舀取/分装/夹固/拆解/压平有原文连续证据，工具名词不会误作动作。
- 最终编排脚本`runs/egodemo_refine_300_20261023/finish_refinement.py`在服务器运行，状态`pipeline_completion_state.json`、日志`finalization_v3.log`。原生批日志`track_batch_vectorized.log`，旧关联PID407052结束后再做最终关联/全量工程审计/私有报告。最终脚本目前Python PID需通过cmdline复核，不用旧PID盲目kill。最终报告尚未生成，仍不得称完整标注通过。

- **第二轮最终实测**：300/300原生追踪、关联提案、全量工程审计完成，300审计通过/0失败。300段有原始表面轨迹，其中299段未被像素标定不一致拦截；162段有过滤后刚体/部件6D（原18段），154段有近手6D，294段有近手表面候选。274791有效候选物体帧对应157372唯一视频帧，不能称整段覆盖。3906部件位姿候选帧；4648轨迹片段、4577实体假设、40有几何依据关联。1666原文区间全部有源操作候选，49类型。人工批准仍0，完整联合标注和训练门禁仍false。
- 像素级标定最终275段抽查一致，20段图像外参修正后通过留出核对并重建追踪，4段待核查、1段明确不一致已禁用三维派生交互；这5段没有因数量要求放宽阈值。37段43轨迹目的性抽查，27条负面筛选；补救6段另有实际可见表面追踪预览manual_recovery_visual。
- 私有交付文件已完成并与本地校验SHA：refinement_report.json、engineering_audit.json、第二轮补全进度.csv、逐段待审清单.csv/json、操作候选分布.csv、使用说明.json、model_provenance.json、code_manifest.json和delivery_manifest.json。生产网页127.0.0.1:18788已切换最终v7数据；通过SSH转发访问，不公开EgoDemo。后续用户明确授权多agent并行交付Mamba格式，使用独立egodemo_mamba_v2_300_20261023目录，不把本阶段pending数据变为gold标签。

> 每人完工一个阶段就追加一条（最新在最上）。格式：
> `## YYYY-MM-DD 名字 — 做了什么｜产出路径｜遗留问题`

## 2026-10-23 Codex — 300段联合补全进行中：原生手部完成，双目/物体/中文事件处理

- 用户要求补齐6D、手部姿态和交互事件，并明确选择“保留300段，按刚体/非刚体分别标注”。不把非刚体强制写成整物体刚体6D，不继承原HO-Cap单样例的人工批准。
- 新私有目录 `/root/gpufree-data/datasets/ego_hoi/egodemo_joint_300_20261023/`；本地 `runs/egodemo_joint_300_20261023/`。300段868255帧原生左右21点、关节旋转、同UUID标注版真实K、世界/相机/图像坐标、缺测掩码均已导出；原始文件不改写。首版原文规则提取2218个操作候选、36种实际命中，另218条原文尚未映射；这不是已审核类别数。词表随后扩展以保留翻转、丢弃、收集等动作，旧7类训练编号未改。
- 300份同版本head_right配套视频已下载、逐文件SHA通过，总25.9428GiB；逐帧PTS/完整解码审计另在运行。双目标定全部来自各自标注episode parquet，并通过来源SHA、UUID和帧数绑定，未混用Raw标定/帧号。
- 实际EgoDemo纸箱试验：同帧双目建立101个纹理参考点，基线0.0639839m，三角化内部重投影中位误差0.3404px；151帧试验96帧PnP估计有效，其余NaN。位姿原点/轴为可见表面参考系，非CAD规范轴、非GT、独立精度null。毛巾非刚体试验65个点、151帧，106帧保有至少12个三维点；翻折后失去观测，输出点轨迹而非刚体6D。
- 已下载并验证Grounding DINO tiny（revision a2bb814dd30d776dcf7e30523b00659f4f141c71）及官方SAM ViT-B；SAM源码官方revision dca509fe793f601edb92606367a655c15ac00fdf，复制独立sam-vendor目录导入，不修改团队环境。9帧试验发现细小物体漏检/腕部设备误检；增加手部邻域放大，剔除近腕仪器区域的几何种子。所有检测/分割仍待审，不能当成稳定身份或金标准。
- 新模块 `complete_egodemo.py,source_semantics.py,object_observations.py,stereo_reference.py,nonrigid_tracks.py,rigid_fit.py,track_object_candidates.py`。中文模型仅翻译原文/抽取名词，不采用其自报动作与材料判断；逐字来源验证拒绝幻觉字段。候选稠密点轨迹、局部表面位姿与近手距离独立输出，跨遮挡ID合并、接触/抓持/协作和精确边界仍需审核。
- 中文原文1666/1666条草稿已完成，最新原文规则2466个操作候选、43种实际命中、106条尚未映射；49类词表容量与实际43类命中分开统计。模型格式错误/非原文名词提及经保留原输出的修复恢复14块，补提取仅匹配本批原文名词词典。300/300右目全解码/PTS检查通过。
- 稠密几何v1–v3试验发现腕部采集设备、透明容器内容物和多物体混合污染，旧版本保留但不计入正式候选汇总；v4扩展腕部排除、整堆泛称过滤、容器表面未知保护和至少5连续帧刚性拟合门槛。15个不同任务、每段3帧的目的性助手抽查发现8个轨迹存在明确身份/表面问题，写入visual_quality_overrides.json拦截；不是随机精度评估，人工批准仍0。
- 新私有中文审核服务src/annotation/egodemo_review.py/.html，服务器仅127.0.0.1:18787，支持源事件原生帧定位、左右手骨架、表面点/有效坐标轴、稳定实体ID关联、刚体/非刚体/铰接类别、工具/工件/支撑角色、关系/转换/协作/操作草稿、版本冲突保护和服务器历史保存。仅pending草稿，测试存储独立web_qa/drafts，未混入生产标注。桌面/390px手机、逐帧定位、单帧半开区间、保存刷新、冲突拒绝与JSON导出通过；网页无JS错误，162项Python回归通过。
- **自动预标注阶段300/300已结束**：双目时间轴、手部导出、中文语义、物体观测和连续轨迹候选均覆盖300次采集；几何文件结构/有效掩码/时间戳/SE(3)/SHA检查300通过、0错误。右目868255帧最大PTS差2.082617e-5秒。1307条连续表面轨迹候选（非去重物体数）；材料候选刚体171、非刚体376、铰接49、未知711。仅18段有整物体6D候选，总31835个物体帧；已知视觉失败扣除1254后30581候选物体帧，不是唯一视频帧覆盖、不是精度。近手表面区间1035条仅为接近候选。
- 修正paper clips/binder clips被材料词形误判的问题：共271处候选元数据修正，原JSON备份material_correction_backups，NPZ未改写、未凭分类生成6D。15段45帧助手抽查发现8个明确身份/表面问题；此处拒绝是负面质量筛选，不替代全序列审核。
- 最终入口joint_report.json、candidate_completion.json、联合处理进度.csv、中文交互候选.csv、操作候选分布.csv、使用说明.json；服务器与本地runs/egodemo_joint_300_20261023均保存报告。模型两份权重重验SHA，13份生产源码与本地哈希一致，162项Python回归通过，浏览器草稿保存/边界/原生帧/手机版/拦截轴显示检查通过。生产审核服务仅127.0.0.1:18787，75个源任务名均附中文显示名；SSH转发说明已写入使用说明。
- **完整联合标注仍未验收**：稳定物体ID、跨tracklet参考系对齐、完整目标覆盖、接触/抓持/支撑/协作及精确边界仍待视觉标注；人工通过0，independent_accuracy=null，training_allowed=false。不能称300段6D与交互都补齐。未训练、提交Git或发布EgoDemo视频/逐条元信息。

## 2026-10-23 Codex — 300次独立采集扩充完成，视频/手姿态/原文事件全部验收

- 用户明确“扩充到300段”。本次扩充EgoDemo视频＋官方手部姿态＋原文语义事件配套池，按UUID去重；不以机位、20秒切片或滑窗凑数，不改写既有HO-Cap联合姿态120段阶段门槛或批准新标注。
- 新目录 `/root/gpufree-data/datasets/ego_hoi/egodemo_expansion_300_20261023/`，独立冻结 `selection_frozen.json`。计划300次采集，复用已有34次配套数据；75个原始任务名称，15个基于标题的选样分组（不是已验证交互类别），每个任务最多8次；预计868255帧、29149.19秒、1666条原文事件、29770998804字节。原始任务名称未按同义词合并，108个scene_id和77个environment_id不是已核实的被试数。
- 本地恢复脚本在 `runs/egodemo_expansion_300_20261023/tools/`。原始源SHA和尺寸冻结后，通过项目既有hf-mirror下载；令牌只经SSH stdin传入进程内存，不保存到远端。完整资产原子改名后才进入校验，复用文件先验SHA；数据盘剩余空间已核对。初始系统Python3.8不兼容既有导入器，已改用现有xw_dataset Python3.10，不改任何环境。
- 导入复用 `import_egodemo.py` 的世界系手部/相机位姿、坏帧掩码和原文事件暂存契约；逐段验证SHA、parquet、完整视频解码帧数、每帧视频PTS与源时间戳一致性。报告分离下载、工程验收、物体6D与人工交互审核，保持training_allowed=false；中文采集/任务分布/交互待审清单随最终报告生成。
- 对话服务重启后核查现状并恢复，下载/验收改为独立进程。镜像降速后改用本地官方源与官方CDN短效链接，链接仅经stdin传输、未落盘；SSH并发限制导致的临时传输失败在限流后全部重试通过。旧系统Python与镜像阶段报告归档attempts，不作为最终计数。未修改环境或使用GPU训练。
- **最终300/300通过**，868255帧，实测29149.185916秒（8.096996小时），2400个源资产SHA/尺寸验证通过，逻辑资产29770998804字节（27.7264GiB）。300份世界系手部暂存NPZ、原文事件与时间轴已生成；左/右有效帧867079/866955，视频PTS与源时间戳最大误差2.082617e-5秒。有效覆盖和时间对齐不代表独立姿态精度。
- 75个来源任务名称，每任务最多8段；1666条原文事件全部保留为待审核，操作类型/主体/客体ID保持空白，没有硬映射旧六类或冒充人工批准。15组选样代表共45帧已做助手视觉抽查，可见真人操作；不是全序列标注验收。正式物体6D完成数0、新增人工交互通过数0，training_allowed=false。
- 交付入口 `completion.json`、`delivery_report.json`、`扩充结果.json`；中文 `采集清单.csv`（300行）、`任务分布.csv`（75项）、`交互标注待审清单.csv`（1666行）。服务器上述新目录与本地 `runs/egodemo_expansion_300_20261023/` 保存报告副本与SHA；原视频/原始parquet/完整姿态留在服务器。既有HO-Cap 24段及原48段冻结不覆盖；此次不发布EgoDemo记录、不开训练、不提交Git。

## 2026-10-23 Codex — 交互类别上限与数据规模只读核查

- 服务器旧版 `src/annotation/vlm_annotate.py` 的提示词及SUPPORTED_INTERACTIONS明确只允许reach/grasp/hold/move/place/release；`schema.py`与`egodemo_events.py`同样限制六类。因此121候选中的6类是预设词表覆盖，不能解释成原始数据只有六种真实操作。逐个review CSV与events.jsonl核对：grasp80、place12、move10、release7、reach6、hold6，121条均pending。
- EgoDemo已下载Raw子集按UUID去重82次采集、15个任务目录、每次4视角共328视频，其中Refill Stapler占30次。服务器已有完整目录索引中EgoProRaw/mp4为118个任务名称/486个UUID，EgoProStandard/lerobot为107个任务名称/668个UUID；这是目录索引容量，不代表payload全部下载、语义去重后的类别数或具备物体6D标注，Raw与Standard不可相加。官方当前数据卡将EgoDemo描述为50小时样本，完整EgoSuite的15000+任务不属于本样本规模。
- 向用户建议保留关系/转换/协作层，增加可跨任务共享的操作语义，先以200–300次独立采集和10–15类操作校验方案，再以1000–2000次独立采集、20–30类操作、10000–20000个审核事件作为主实验规划，最终根据验证集学习曲线与类别覆盖调整。这是建议，不是统计充分性保证，也没有替换既有120段阶段配置。
- 本轮未修改标签词表、模型或已有标注，未下载扩容、训练、提交或发布；仅记录只读核查与规模建议。来源：服务器现存数据/代码与 https://huggingface.co/datasets/LightwheelAI/EgoDemo 。

## 2026-10-23 Codex — 用户整批审核通过，Mamba格式/编解码实际核对

- 用户明确“我看了一遍，大致没问题，可以都通过”，据此13条区间标注与5条手别复核accepted，桌面实体候选确认。reviewer为用户（本会话确认），保留原话/应用时间/原SHA与original_label_source。新版本 `runs/pose_relation_annotation_20261023/approved_01/`；服务器 `/root/gpufree-data/datasets/ego_hoi/pose_relation_annotation_20261023/approved_01/` 已同步并校验13+5及源哈希，原pending版本和NPZ不改写，training_allowed=false。
- 私有审核网页同URL更新，版本2部署成功 `appgdep_6adaf0e1ce048191a2fc31748ffc76b7`，独立Site源码commit `b6cc089d76444c882fdbaee4d047fe4bc849a766`。13/13显示通过；自动迁移旧浏览器草稿且保留其内容，确认之后的新修改不被再次通过。13项Node回归通过，浏览器检查初始通过/5帧状态/草稿迁移/后续修改跨刷新仍pending通过。研究项目仓库未提交。
- 服务器实际Mamba配置与encoder/decoder已核对；train.py存在他人新增label_valid严格检查和loss/metric mask，本地尚无这部分，保留远端修改并保存只读快照。builder仍启发式label_clip，未消费新关系标注或生成label_valid，不能把人工审核结果视为已经接入正式训练。
- L40空闲时用现有caron_mamba环境运行一次随机初始化合成前向：B2/W32/N8/D128，encoder(2,32,11,128)→Mamba(22,32,128)→主体(2,3)/物体(2,9)/类型(2,7)，有效logit有限、padding为-inf；峰值张量分配18,544,128bytes。未加载checkpoint、未做优化或训练，不是准确率结论。报告 `mamba_shape_inspection.json`。
- 解码器只读诊断：交换两个物体token，object logits最大变化1.49e-7；平均池化丢失槽位对应。提出逐物体打分、关系多标签、未来事件和协作辅助监督的课题版设计建议，尚未实施架构改动；新多关系不可直接硬压为旧单主体/单物体/单类型标签。详细字段/现状与建议分开写入 `training_format_inspection.json`，SOP更新。
- 尚需标注→样本适配、源时基核实与独立姿态质量门禁；此次只确认标注和核对格式，不启动训练、不扩数据、不改环境。

## 2026-10-23 Codex — 中文网页审核台上线，原生帧定位与人工审核回读验证完成

- 用户要求用Web审核，并尽量使用中文。新增 `src/annotation/review_web.py`、`src/annotation/web_assets/`，先补 `pose_relation_review_web_v1` 契约；数据保持原 `pose_relation_annotation_v1`。界面、物体显示名、关系/事件/协作角色、状态、操作、备注和CSV表头中文化，稳定ID不变。
- 私有审核地址 `https://hand-object-review-zh.justgiant1027.chatgpt.site`，Sites部署状态succeeded，project `appgprj_6adad6b3343481918225d0804eac4a58`，deployment `appgdep_6adada6c1164819185475f79f3b208ad`。独立Site源码仓库提交 `49d7303307ea64db7987e0055e12f9e55ab0ddce`；未提交/推送研究项目仓库。只发布此HO-Cap白名单样例、估计投影和网页，保留作者/CC BY 4.0/来源与修改说明，无EgoDemo或权重/凭据。
- 446原生帧JPEG用于精确定位，播放视频完整解码446帧。原图与当前审核草稿动态绘制骨架/物体轴/关系；不是修改字段后仍看旧烘焙标签。支持逐帧/速度/区间循环、四层新增与编辑、证据帧、中文名称、5帧手别异常独立备注。
- 有效编辑在浏览器自动存为pending；用户填写审核人并显式点击才产生human审核状态/时间，保留原候选来源。已通过关系被修改会使依赖协作回到待审。起止帧UI为包含结束帧，JSON保持半开区间，445正确导出为446。浏览器本地存储不是服务器同步，JSON导出/导入用于备份与交接，来源不匹配拒绝导入。
- 本地10项Node回归与14项Chromium用户流程检查通过（含播放、保存刷新、关系失效、单帧转换边界、JSON/中文CSV导入导出、手机布局），无浏览器运行错误；导出另通过Python既有契约校验。证据 `runs/pose_relation_annotation_20261023/web_qa/`。其中qa_export是隔离测试用例，不是真实人工审核；生产13条候选仍全部pending。
- 服务器已归档代码及离线包 `/root/gpufree-data/datasets/ego_hoi/pose_relation_annotation_20261023/web_review_01/`，unpacked/dist可静态托管；453文件哈希与本地一致。Site工作目录 `/home/nvidia/algotithm/sites/hoi-review-zh/`。交接SOP已更新。未改源NPZ、既有VLM数据或模型环境；源时基/独立姿态精度和正式训练门禁保持待验收。

## 2026-10-23 Codex — 姿态实体与协作关系标注样例完成，本地与L40复现446帧

- 用户明确要求在手部姿态上叠加左右、物体6D上叠加物体信息并定义协作关系。先补契约 `pose_relation_annotation_v1`，再新增 `src/annotation/pose_relations.py` 与24项契约测试；本地/服务器测试均通过。远端既有VLM模块和数据保留，契约增补前备份 `/root/gpufree-data/backups/pose_relations_20261023T031510Z/`。
- 完整446帧显示解剖左右手、4个稳定物体ID/官方名称中文翻译、原6D框/轴、手物/物物有向关系、协作模式及转换区间；关系图的支撑边绕开无关物体节点。输出视频均完整解码446帧，本地/服务器输入与所有输出SHA核验通过，源NPZ不变。
- 本地交付 `runs/pose_relation_annotation_20261023/render_03/`；服务器交付 `/root/gpufree-data/datasets/ego_hoi/pose_relation_annotation_20261023/render_01/`。含 `overlay.mp4`、`annotations.json`、446条 `frame_graphs.jsonl`、`review.csv`、预览及manifest。输入分别绑定本地/服务器源清单，字体随服务器资源保存，未修改base或模型环境。
- 7实体（双手/4物体/桌面参照），6段持续关系、5个转换候选、1段单手操作、1段任务语义。所有交互候选均为助手看图提出的pending，geometry未审核；未将VLM或助手判断接受为人工真值。示例为单手，schema支持双手共同操作、分工、交接、工具作用，未捏造示例中不存在的双手行为。
- 左手441帧、右手输出5帧，异常363/382/383/396/398保留原估计并标识手别待审；手部相对深度与物体米制坐标不混用。未知物理时间只保留原生帧，显示30FPS不作真实时间。关系可同时多条；离线全片标注不得充当预测模型输入，正式训练仍关闭。
- 复现和审核步骤已补 `docs/HAND_OBJECT_HANDOFF_SOP.md`；CSV目前仅审核清单，修改新的JSON副本再渲染。待人工校正关系区间/转换边界、手别异常及独立姿态质量；没有扩大数据批次、训练、Git提交或发布。

## 2026-10-23 Codex — L40恢复连接，446帧完整序列工程验收通过；交互标注方案复核

- 已通过用户提供的凭据连接现役服务器；凭据未写入项目或日志。修改前备份 `/root/gpufree-data/backups/egodata_resume_20261023T025311Z/`，只同步 `full_hocap.py` 的布尔转换与新增CPU回归文件；保留远端VLM标注、审核台及其他在途修改。
- 现有 `panda_human_live` 环境中 **91 passed（0.67秒）**；对原 `full_hocap_20261022/production_01/` 复验成功，新增 `acceptance.json`，17项工程检查通过。原report、NPZ、CSV和视频保留，下载副本SHA均匹配；无需重跑GPU推理。
- 446/446视频帧完整解码；四物体各446帧数值有效，每帧至少一只手，左手441帧、右手5帧。右手输出帧363/382/383/396/398待结合原图复核。覆盖率不等于准确率，源物理时戳未知，正式训练仍不放行；本地证据 `runs/full_hocap_resume_20261023/`。
- 用户将重点转向更适合整个课题的交互标注方案。只读检查远端20261023批次：10段共121候选，全部pending、geometry not_checked、object_id缺失；每段仅12张证据图，199秒绕线序列采样间隔18.07–18.10秒。Prompt定义grasp为建立抓持，但持续握持示例仍标为grasp；审核材料为稀疏面板幻灯片，不能承担精确事件边界验收。
- 本轮正在提出“稳定实体与关系状态＋关键状态转换＋连续视频人工审核”的分层方案；未修改既有类别契约、未将候选接受为真值、未启动新训练或批量推理。完整视觉/物理时间/独立精度验收及正式交互标签仍待完成。

## 2026-10-23 Codex — 接续完整HO-Cap验收：修复JSON布尔值，本地验证通过

- 用户要求连接现役L40继续数据处理；已核对项目契约和交接SOP。首次SSH被会话沙箱拒绝：`socket: Operation not permitted`。用户恢复网络权限后服务器可达；默认agent触发 `Too many authentication failures`，显式限定 `id_ed25519` 后仍为 `Permission denied (publickey,password,keyboard-interactive)`。待用户确认该服务器的有效登录凭据，尚未取得远端实时状态、未同步代码或操作服务器数据。
- 定位 `src/pose6d/full_hocap.py` 中 `unknown_source_time_not_fabricated` 返回NumPy布尔标量，导致验收文件写入和CLI JSON输出失败；在该检查生成处显式转为Python bool，保持真假语义与严格JSON写入，不改变数据格式、原始产物或训练准入条件。
- `tests/test_full_hocap.py` 新增8种合成CPU验收场景，覆盖已知/未知时间、时间内容一致/不一致、真实后端标志/测试后端标志。实际读取合成NPZ/CSV、编码和完整解码3帧MP4，并检查JSON文件/CLI输出一致、失败退出码、原文件摘要不变和历史验收拒绝覆盖。旧代码首先复现相同TypeError；修复后该文件 **91 passed（0.26秒）**。这是合成工程回归，未执行真实模型推理。
- 遗留：须解决SSH认证后同步修复，先核对远端 `full_hocap_20261022/production_01/` 是否已有后续验收，再用现有 `panda_human_live` 环境对已有446帧/4物体产物复验，保留原推理报告与失败日志；视觉、物理时间、参考精度证据仍待验收。未更新正式合格采集数，未扩容、训练、提交或发布。

## 2026-10-22 Codex — 44帧基线SOP与最小代码交接

- 按用户要求优先发布已有真人手＋单物体6D短样本交接；接手入口 `docs/HAND_OBJECT_HANDOFF_SOP.md`，实测说明 `docs/HUMAN_HAND_OBJECT_LIVE_20261022.md`，基线参数 `configs/human_rgbd_baseline.args`。公开说明已移除辅助来源的记录级标识。
- 联合入口及对应测试发布历史44帧稳定快照，不包含未发布forecast依赖；工作区预测变体和其他在途修改保留。手部与RGB-D封装、最小配置及四组回归测试一起交接，不包含原始数据、权重或凭据。
- 干净HEAD导出加拟发布稳定文件的四组测试：147 passed（1.01秒）；SOP产物检查脚本对历史44帧真实输出执行通过。证据在 `runs/handoff_release_verification_20261022/`。本轮没有重新运行GPU实拍，不将轻量测试当精度验收；发布前再核对实际暂存树。
- 446帧/4物体实验已产生推理结果，但验收器NumPy布尔值JSON序列化失败，未取得成功验收；作为未发布WIP交接，不计正式合格。后续模型按SOP接手，120段扩充与训练不由本次交接继续执行。

## 2026-10-22 Codex — 锁定120个合格联合采集目标与持续计数口径

- 阶段目标120个通过验收的真人手部＋物体6D独立采集，候选目标约84/18/18；优先被试/采集整组隔离，不用Ego辅助、机位、滑窗、重复或插值凑数。旧冻结不改写。
- 当前已下载联合候选24（16/4/4）、正式合格0（0/0/0），距120合格缺口120；辅助24另列。目标状态在 `runs/human_dataset_v1_20261022/target_120_progress.json`。
- 交叉核查固定HO-Cap镜像目录与官方split：64个采集档案中train50/valid7/test7。当前只下载官方train策略下最多另补26候选，即使50全合格仍至少需其他来源70个合格采集；不把镜像目录数当整个官方数据规模。逐采集证据 `source_capacity_120.json`。
- 此次只锁定目标、审计来源容量，没有启动下载/训练；需推进既有24条时间/几何/参考质量验收与新来源验证，数量质量都达标再训练。
- GitHub同步限定为本阶段三份文档；只提交这两条数据阶段日志、交付表的数据阶段条目及数据规范文档，不夹带历史未提交阶段、代码、数据、权重或凭据。1,081tests＋84subtests对应本地/服务器工作区，不表示本次文档commit包含新增工具和测试源码。
- GitHub权限诊断已纠正并恢复同步：前次拒绝来自默认SSH身份，不代表目标账号无写权限。显式选择已获授权的 `miracle-techlink` 账号后，已将本阶段文档提交推送至main，并从GitHub读取远端SHA核验成功。未force-push、未更改默认SSH配置或通过只读代理推送；后续仍固定使用正确身份。
- ModelScope目标为 `Liuyue9698/ego_pose_dataset`；本机凭据身份匹配仓库所有者，仓库读取及Git push --dry-run通过，但尚未执行数据写入。仓库为public，master现仅README/.gitattributes，模板许可为Apache-2.0。已核查EgoDemo固定许可证§3禁止再托管及可定位记录的清单；HO-Cap按CC BY 4.0仅可白名单发布，保留署名/来源/修改说明，排除MANO基础模型等第三方受限资产。发布前须纠正数据许可说明，不能将HO-Cap重许可为Apache；此轮ModelScope上传候选0、正式合格0，不把权限预检当上传成功。

## 2026-10-22 Codex 多Agent — 数据先行，扩充48个真人采集并冻结候选划分

- 按用户要求停止新训练，三路并行做HO-Cap采集、EgoDemo长序列和发布治理；主agent独立审图、全量结构检查与整合，不复用旧短片冒充新长数据。
- 实际HO-Cap主机位24个采集/6人/22,729帧，RGB-D/K/mesh/公开参考已落地；56个物体ID、3个任务类型。EgoDemo24个新UUID/23,812帧/798.9307秒，单段20.13–59.63秒、9任务12场景；身份未知，仅辅助池。主视角共48独立采集/46,541帧，另6低视角是重复机位，不额外计数。
- 首/中/末预览确认48个主/辅助采集为真人手；HO-Cap低视角037排除，主视角105322251564全部22,729帧解码、SHA和内部几何结构检查通过。内部重投影一致性不是物理精度，预览不是全序列验收。HO-Cap实际时基未证实，不把默认30FPS算成实测长序列。
- 新增下载器、合并台账、参考结构审计、不可变freeze与测试；候选按subject整体16/4/4（主视角14,032/4,993/3,704帧），24条身份未知Ego辅助隔离。真实审计抓到pending样本权重为0导致全train，修复并新增回归；旧frozen_v1保留失败历史，当前本地frozen_v2，服务器frozen_server_v2_complete。
- 本地/服务器均1,081tests+84subtests通过；数据目录 `runs/human_dataset_v1_20261022/` 与 `/root/gpufree-data/datasets/ego_hoi/human_dataset_v1_20261022/`，详细协议/路径见 `docs/HUMAN_DATASET_V1_20261022.md`，汇总/分布/排除/SHA在数据根目录。未修改base/现有环境、未提交git、未启动新训练或预测评分。
- **未通过正式训练准入**：联合就绪0、training_allowed=false；120合格采集/4动作及各split数量不足，HO-Cap时间、场景、坐标/参考质量未全部核验，因果观测和有效掩码未生成，完整/≥5%帧复核待做。Ego无depth/K/mesh/物体GT和相机补偿，不冒充联合6D基准；固定第三人称HO-Cap不冒充第一人称。

## 2026-10-22 Daniel — 真人手/物体未来姿态预测原型与失败边界实测

- 接口先补 `pose_forecast_v1`，独立实现 `src/forecasting/`：时戳窗口/分组隔离、持位/恒速、复用HOIEncoder的LSTM/Mamba及独立PoseDecoder、严格checkpoint、伪标签误差、因果预测与明确分栏未来对照；旧三头HOI不改。
- 并行分工：数据agent准备6个真人DexYCB独立采集436帧，模型agent实现回归模型/训练器，QA agent查因果泄漏/坐标/掩码/检查点，主agent集成真实提取、训练与视频。公开取得片段仅68–74帧，未拼接伪长序列；4/1/1预定划分，test机位仅在推理前按人手可见性改为857。
- 实际48 train / 7 val窗口，0.5/1秒test28/13个观察时刻。LSTM/Mamba各3seed20epoch在L40真实训练，最佳epoch分别3/4/2、3/4/4，val选优、全部训练后才测试评分。训练+测试约12.63s，未更改任何现有环境/base。
- **预测效果未通过**：1秒手像素误差持位129.01、恒速104.20、LSTM均值136.76、Mamba141.05；物体位置误差分别224.63/207.08/214.94/213.65mm（均相对未来估计，不是GT）。学习模型只有部分平移收益，没有整体优势；保留全部差结果，不据test调参。
- **实时尚未通过**：真实逐帧感知+LSTM预测43激活帧，优化后29.60FPS、P95 38.55ms、22/43超时；含未预测预热帧的32.42FPS不作为预测速度。无直播摄像头、无米制手物融合、无独立精度或HOI准确率声明。
- 交付 `runs/pose_forecast_20261022/future_pose_{comparison,online}.mp4`、`future_pose_preview.jpg`、6组best/last权重和报告；说明 `docs/POSE_FORECAST_PILOT_20261022.md`。本地/服务器均984tests+26subtests、18项独立风险测试、104项产物重算通过。服务器完整测试用已有panda_human_live（daniel_test缺trimesh，不修改环境）。源码路径归一修复不改模型权重；训练时源码快照保留。
- 遗留：需更长、更多采集与可靠参考/标定，再做预先约定的基线优势和完整时延验收；本轮只通过工程闭环，不宣称可可靠预测，不推进正式HOI训练。

## 2026-10-22 Codex 多 Agent — 改为真人手＋FoundationPose在线同帧输出

- **用户范围纠正**：机械臂mustard0不是EgoDemo，不再作为真人手交付。本轮真实输入为DexYCB第三人称右手拿024_bowl，源30–73共44帧/30FPS，非EgoDemo；不把第三人称冒称第一人称。源RGB-D/K/原mesh/首帧真实mask齐备，502个资产文件哈希验证，GT与scene隔离。
- **实现**：新增 `src/hand_pose/online_hands.py`、`src/pose6d/run_human_rgbd.py` 和对应测试。MediaPipe真实逐帧21点、非镜像左右标签修正，CPU手部/GPU物体同帧并行，最多下一帧I/O解码预取、禁止未来帧模型推理；立即叠加并写视频，有桌面时可`--display`。手部相对Z不冒充相机米制融合，缺失NaN，模型失败不回退mock。
- **实测**：最终lite手模型＋FP跟踪1次配置，三轮同44帧均44/44联合输出；生产FPS41.409/41.471/41.524，环路P95 27.056/27.967/27.265ms，读入→输出P95 55.490/61.821/66.899ms。每轮仍2/43慢帧，首帧注册1.63–1.66s，不承诺硬实时。full手模型预取方案两轮P95未达30FPS预算也保留。位置/旋转/手部准确率均null，碗近轴对称，性能模式精度损失未量化。
- **环境/测试**：独立panda_human_live，Torch2.6.0+cu124/NumPy1.26.4/MediaPipe0.10.21；原环境不改。原生GPU探针通过；本地新增模块79项，服务器完整876 passed＋26 subtests passed（12.54s）；独立产物182项检查全过、107文件/702哈希引用匹配。保留每轮慢帧及注册期预取导致的约1.72秒最大读源→输出延迟，不冒充硬实时。
- **交付**：`docs/HUMAN_HAND_OBJECT_LIVE_20261022.md`；本地 `runs/human_pose_live_20261022/`，主视频 `human_hands_foundationpose_30fps.mp4`，预览/拼图、手与物NPZ、逐帧延迟CSV、来源manifest、性能各轮证据。输入是录制RGB-D在线回放，服务器未连接视频设备，未验证实时相机/桌面窗口。
- **实际EgoDemo审计**：吹风机7cd681cb…本地有左目与双目标定，右目/MCAP仍仅LFS指针；938帧视频与1107帧raw外参未建立映射、基线单位未确认、raw内部UUID不一致，实际depth/目标mesh/尺寸/mask未取得。因此不宣称EgoDemo完成，不恢复正式HOI训练。

## 2026-10-22 Codex 多 Agent — 真实 FoundationPose 全序列跑通（737/737，非精度放行）

- **实际运行**：官方 scorer/refiner + 实拍 mustard0，官方脚本3帧烟测、项目新runner3帧烟测、完整737帧全部成功。推理34.005秒，runner总82.113秒；完整MP4解码737帧。运行覆盖100%，位置/旋转/联合准确率均null，未将公开样例视为EgoDemo或手物融合通过。
- **环境与资产**：新建数据盘独立 `panda_fp_real`（torch2.6.0+cu124），源码构建并实际测通pytorch3d CUDA KNN、nvdiffrast CUDA光栅化、mycpp cluster_poses；不改原panda_6d/base。官方权重、737对RGB-D、首帧真实mask、原mesh和K齐备；深度mm→m；所有来源有SHA。GT缺失，官方原始包前69MB有界调查仍未取得参考，不计算精度。
- **实现与测试**：新增 `src/pose6d/run_rgbd_sequence.py` 和67项CPU测试，严格真实mask注册、按RGB stem配深度、仅后验参考评估、原mesh object→camera输出、独立新目录、无mock回退。服务器完整797 passed + 26 subtests passed（13.04秒）。另一个agent独立41项产物复核通过、1489份来源文件SHA一致；737帧NPZ/CSV/PNG/视频逐帧一致。测试与数值复核均不等于真实精度。
- **交付**：报告 `docs/FOUNDATIONPOSE_REAL_RUN_20261022.md`；本地 `runs/foundationpose_real_20261022/`、服务器 `datasets/ego_hoi/foundationpose_real_20261022/`。含NPZ/737行CSV、完整视频、3张真实点云/相机轴/物体mesh同系图、轨迹图、运行与安装/测试日志、环境冻结、来源manifest；全部737张叠加PNG保留服务器。
- **下一关**：真实模型不可运行已解决；实际EgoDemo仍缺刚性目标的同采集米制深度/真实分割/几何模型/独立参考，不能拿公开瓶子样例替代；旧8份mock仍拒绝，正式HOI训练不恢复。

## 2026-10-22 Codex 多 Agent — 6D 位置/姿态专项验收（0/8 交付通过）

- **报告**：`docs/POSE6D_ACCEPTANCE_20261022.md`。主 agent 逐帧量化现有 8 份 pose，两个 agent 分别核查真实运行环境、独立数值门禁；本轮验收拒绝，不将代码回归通过当模型精度。
- **真实磁盘证据**：`/home/daniel/pipeline_run/data/pose6d/` 有 8 对 NPZ/JSON，合计 5,214 视频帧、10,428 物体帧；旧格式检查 8/8 过，但全部 mock/no depth、R=I、Z=1，X/Y 与框中心反投影残差至多 1.47e-8。真实6D交付通过 0/8；位置/旋转准确率与联合成功率均 null。历史文件未改，16 个源文件摘要一致。
- **失败报告修复**：严格模式不再吞 tracker 创建/位姿推理异常或忽略必需深度缺失；非严格创建失败明确标记。新 mock 输出强制 `mock_pose` 警告，提前关闭视频迭代器释放资源。旧源码 8 项选定回归 6 failed/2 passed，修复后测试文件 13 passed、本地/服务器全套均 730 + 26 subtests 通过；旧文件备份至 `backups/egodata_process_20261022_pose6d/`。
- **独立验证**：137 个场景/1,632 断言通过，26,778 次矩阵分类无参考分歧；仅合成数值验证。L40 CUDA 可用，但真实 FoundationPose 导入缺依赖，mycpp仅命名空间，无native功能；scorer/refiner 权重及目标 mesh/mask/逐帧米制深度尚未齐备。DepthAnything metric/relative源码可导入但未加载权重，不能将受限探针临时目录异常当作环境缺陷；矩形 bbox mask 与未接入重建也是后续缺口。
- **交付目录**：本地 `runs/pose6d_acceptance_20261022/`，服务器 `datasets/ego_hoi/pose6d_acceptance_20261022/`，含10,428行CSV、mock拒绝曲线、输入副本、严格门禁、运行探针和测试日志。下一步先补真实刚性目标短序列与独立精度参考，不继续正式训练。

## 2026-10-22 Codex 多 Agent — 手部/物体 6D 融合验收证据（未通过）

- **结论**：整体 `blocked`，暂停训练。两段真实物体 6D 和独立精度参考缺失，不将 hand/camera 图称为完整融合，也不将有效率当准确率。交付报告 `docs/FUSION_ACCEPTANCE_20261022.md`。
- **分工与实现**：几何 agent 完成严格 world/camera 双向变换与投影；门禁 agent 完成非 mock/米制来源/标定/时间/刚体检查；主 agent 找到标注版 `meta/episodes` 真实内参，完成重投影、多视图、视频和全帧统计。不混用 Raw 的外参与帧号。
- **真实结果**：货架标签 1,194 帧、电缆 5,970 帧，共 7,164 帧；双手同时有效 7,146 帧（99.7487%，仅标注覆盖率）。有效关节全部正深度；电缆左手最后 19 帧未全部在图内。真实物体覆盖 0 帧，联合可评估 0 帧，精度/联合成功率 `null`。
- **证据交付**：本地 `runs/fusion_acceptance_20261022_review/` 与服务器 `datasets/ego_hoi/fusion_acceptance_20261022_review/`：18 张 PNG、2 个 8–18 秒预览 MP4、7,164 行 CSV、JSON 门禁/溯源/逐图抽查记录。MP4 各 100 帧完整解码；图上明确相机轴非物体轴。抽查不是独立人工真值，42 条原有跳变复核队列未全部关闭。
- **回归**：本地/服务器均 723 passed + 26 subtests passed。只读复核发现并修复文件未绑定采集/机位、时间仅递增未绑定帧号/PTS 两处缺口，补 51 项回归并重渲染。既有 602 帧 mock pose 即使矩阵数值有效也被拒绝。渲染 CLI 返回 1 为预期 blocked；未安装新包、未改 base、未 commit/push。
- **下一关**：定义刚性目标及稳定 ID，补真实 mask/米制深度/mesh 与真实 pose，接入同帧同相机 world 融合，再按预先确定的独立参考与阈值验收；通过前不恢复正式 HOI 训练。

## 2026-10-22 Codex 多 Agent — 第二轮并行测试与真实轨迹结果

- **交付报告**：`docs/VALIDATION_20261022_PHASE2.md`，明确验收标准、复现命令和下一阶段门槛。3 个 agent 分别完成模型边界、真实资产 QA、轨迹对照；主 agent 完成刚体约定、GPU 联调与结果归档。
- **根因修复**：invalid 手/物在非线性运算前清零；decoder 池化前排除 padding；masked object logits 用 `-inf`；CE 只计算有效标签行。新增边界测试最初 13 failed，修复后 26 passed；完整本地/服务器均 **412 passed + 26 subtests passed**，checker 25/25、合成构建/可视化自测通过。
- **真实 QA**：10 段配套 head_left 视频逐帧解码 **18,079 帧，0 errors / 12 warnings**，PTS 最大偏差约 7.45 微秒。9 段合计 64 帧未覆盖原文事件，不能当 none；3 项手腕跳变警告涉及 42 条手别记录/40 帧，已输出 `quality_review.csv`，仍待复核，不自动剔除。
- **相机证据**：双目刚体残差在 10 段均强烈支持四元数 camera→world；尚未核验内参/光轴/重投影，未转换源数据或启用正式 camera 模式。
- **真实辅助预测**：官方双手腕 16 帧观察→1 秒未来；UUID train/val/test=6/2/2，样本 1,715/383/1,404，train-only normalization。L40 完成 LSTM/Mamba × 3 seed × 20 epochs（约 11.51 秒），六组 checkpoint 重载差异 0。test episode 等权终点误差：持位 **0.059997**、恒速 **0.096332**、LSTM **0.058974±0.000491**、Mamba **0.059843±0.000153**（原始坐标单位；±为 seed 样本标准差）。相对持位仅 +1.71%/+0.26%，不构成 Mamba 优势或 HOI 有效性证据。
- **GPU 压力验收**：原三头模型 LSTM/Mamba × FP32/BF16 各 100 step，全 7 类合成压力 batch，四组通过、梯度有限，因果前缀/重载差异均 0。最终验收目录 `gpu_acceptance_fixed/`；不把合成记忆准确率当泛化效果。
- **服务器产物**：`/root/gpufree-data/datasets/ego_hoi/validation_20261022_phase2/`，含完整 QA、相机诊断、42 行复核队列、6 组轨迹 checkpoint/预测/日志、4 组 GPU checkpoint、测试日志及源码摘要。备份在 `/root/gpufree-data/backups/egodata_process_20261022_phase2/`；仅 `daniel_test` 补齐 pytest/subtests/headless OpenCV，未改 base，未 commit/push。
- **下一阶段**：先复核 42 条跳变与相机投影，再做电缆/货架标签两段物体轨迹与 46 条原事件拆分复核；稳定物体 ID、坐标、人工三头标签、UUID 隔离和非单类分布全部满足后，才能进入正式 0.5/1/2 秒 HOI 对照。当前不扩大模型、不用 test 调参、不直接全量训练 82 段。

## 2026-10-22 Codex 多 Agent — P0 正确性修复与 10 段真实标注导入

- **并行分工**：4 个 agent 分别处理⑥时序/指标、④预测间隔/UUID 分组、③深度模块、标注清单；其中③ agent 后续接续标注导入。主 agent 负责契约先行、⑤坐标安全、⑦校验与服务器联调。
- **正确性修复**：⑥ `(B,W,E,D)` 先 permute 再 reshape，增加能击中旧实现的因果累积回归；新增每头 macro-F1、完整标签样本的三头联合准确率。⑤默认 mixed 坐标模式屏蔽手/物跨空间几何。③按 metric/relative 选择独立包命名空间，避免错误模型与模块缓存串用。
- **预测协议**：契约升级 2026-10-22；④支持显式预测秒数和采集 UUID 级 split，保留旧版下一帧调用兼容性。⑦检查新时间字段、观察/目标分离和跨机位泄漏。服务器旧合成输入重新生成 **1 秒间隔 300 样本，0 error / 0 warning**，路径 `datasets/ego_hoi/validation_20261022/samples_h1s/`；仍是合成验收，不是真实训练集。
- **测试**：本地 `pytest tests src/pose6d/tests src/hand_pose/test_extract_hands.py` 为 **267 passed + 26 subtests passed**；原 checker 自测 **25/25**、样本构建/可视化合成自测通过。服务器训练新增 10 项 standalone 回归通过；L40 上小模型 LSTM/Mamba × FP32/BF16 共 4 组前后向损失/梯度有限，记录在 `validation_20261022/gpu_smoke.json`，不作为收敛或效果结论。
- **真实资产**：82 个 Raw UUID 全部匹配标注版；任务均衡选择 10 段，40 个 LFS 文件 **599.91 MiB** 完成 SHA-256 校验，另复制 30 个 JSON 元数据。payload 在 `datasets/ego_hoi/raw/annotated_pilot_20261022/`；利用已有授权 HF SSH 身份取临时 LFS 下载授权，经本地网络下载后传回服务器，未落盘 token/私钥。
- **导入验收**：`annotated_pilot_20261022/staging/` 已导入 **10 段 / 18,079 帧 / 46 个原文事件**，视频头、manifest、parquet 帧数全部一致；有效左/右手帧分别 18,062 / 18,059，坏帧掩码已应用并保留 raw 数组。`annotation_review.csv` 的 46 行三头标签留空、状态 pending；世界→相机与七类 HOI 不做猜测。
- **真实元数据差异**：manifest FPS 与 info 标称 FPS 的统计定义不同；导入器保留二者、实际 timestamp 间隔及差异警告，不重采样、不误判为损坏。原文事件字节保留，重叠事件可保留多个 ID，未覆盖帧不标为 none。
- **可视化边界**：已抽看货架标签 9.0s 与绕线 9.5s 两帧，场景与原文事件一致；`previews/` 与 `preview_review.json` 留存证据，但不代表全视频/所有边界或 3D 投影验收。
- **环境与部署**：仅向 xw_dataset 增加 pyarrow 19.0.1、panda_6d 增加 opencv-python-headless 4.11.0.86，base 未变更。服务器实际 DepthAnything relative/metric 两类模块导入与构造签名检查通过（未加载权重/未推理），见 `validation_20261022/depth_import.json`。修改同步到 `/workspace/egodata_process`；修改前备份在 `/root/gpufree-data/backups/egodata_process_20261022_correctness/`，未 commit/push。
- **下一步**：落实物体跟踪/ID、逐交互阶段主体/客体/类型审核、明确相机位姿方向与坐标变换，然后生成正式 samples、扩充每任务采集量并按 UUID 做 LSTM/Mamba 对照。真实 FoundationPose 权重、mask、mesh 与推理仍未验收；本轮没有启动正式模型训练。

## 2026-10-22 Daniel — 真实①② + Mamba GPU 验证 + 工位/清理收尾

- **EgoDemo 子集下载完成**：15 个机电类场景、328 个 mp4、19.47 GiB、0 个 incomplete；路径 `/root/gpufree-data/datasets/ego_hoi/raw/egodemo_subset/`。使用 hf CLI + hf-mirror + read token，未把 token 写入 git
- **真实①②复核通过**：2 个 head_left 视频重编码为 6 clips，clips 契约 0 error（2 个 10–18s 残段 warning）；2 个 hand_pose / 1200 帧 0 error，左/右/双手检出率 **75.2% / 83.8% / 63.3%**。产出在 `datasets/ego_hoi/{clips_real,hand_pose_real}/`
- **Mamba 在 L40 全验证通过**：causal_conv1d 1.7.0 + mamba_ssm 2.2.6.post3 本机 sm_89 wheel；FP32/BF16 前向反向、cuobjdump sm_89 cubin、3 epoch 训练与 checkpoint 重载均通过。train loss **4.3740→1.4405**、val loss **2.9758→0.9296**、峰值显存 **699 MiB**；产物 `/home/daniel/pipeline_run/runs/mamba_smoke/`
- **依赖兼容定格**：caron_mamba/daniel_test 均为 torch 2.6.0+cu124；transformers 5.20.2 与 mamba_ssm 不兼容，已降为 **transformers 4.56.2 + tokenizers 0.22.2 + huggingface-hub 0.36.2**。已独立复跑 Mamba GPU 前后向与 Arc 10 项 standalone 单测
- **工位最终结构**：7 个 `/home/<缩写>@<群昵称>/` 均无整仓 clone，只保留 README、实体 TASK、data 软链及本人任务实验/测试代码、refs 和产物；公用主代码仅 `/workspace/egodata_process`
- **清理完成**：删除 6 份工位冗余 clone、旧不兼容 wheel 与 `/tmp/miniconda.sh`（约 842 MiB）；Mamba 验证后删除临时 `src_build`（约 926 MiB），保留 `wheels/ubuntu2004/`。uv-cache 约 16G 保留用于加速重建；数据盘当前约 450G 可用
- **边界/遗留**：真实③ FoundationPose/深度/mesh 后端与真实④样本尚未完成；当前合成样本标签分布单一，冒烟准确率 1.0 不能作为模型效果结论。ModelScope ego_pose_dataset 仍为空壳

## 2026-10-22 Daniel — 迁移新服务器（L40）+ 全链路 GPU 冒烟

- **换机**：旧 4090 机器 GPU 挂不上弃用 → 新机器 `ssh -p 30314 root@120.209.70.195`（L40 46G，Ubuntu 20.04 glibc 2.31，数据盘 246G）
- 7 个 conda 环境在新机器全部重建（uv 阿里云；弃 1-2MB/s 的 rsync 方案）；wheels/工位/pipeline_run 已迁移
- **①→⑥ 全链路已端到端跑通**：切分（发现 -c copy 漂移，真实数据须 --reencode）→ 手部（mediapipe 定格 0.10.21）→ mock pose → 313 样本（check 全绿）→ **LSTM 训练 L40 上 loss 2.49→0.70**
- **EgoDemo 已获访问权**：元数据 clone 完成，全量 1.44TB 只拉子集；hf-mirror 7.3MB/s vs 直连 40KB/s，gated 走镜像必须 read token（已配）
- **mamba 走源码编译**（glibc 2.31 + sm_89 补丁 + FORCE_BUILD + --no-deps），详见 mamba-training skill；wheel 备份 `wheels/ubuntu2004/`
- 7 张任务卡已追加"下一步开发方向"（各模块实测发现 + 下一步）
- 遗留：mamba 编译验证与真实数据 ①② 由两个 agent 并行执行中；ego_pose_dataset（ModelScope）仍空壳

## 2026-10-15 Daniel — 七模块代码并行开发完成 + 契约首轮冻结

- 7 个 agent 并行完成各模块初版：① `src/preprocessing/split_clips.py`（fake-ffmpeg 编排级 e2e 过）② `src/hand_pose/extract_hands.py`（12 单测过）③ `src/pose6d/`（24 单测过，三段式可插拔：Detector/Depth/PoseBackend）④ `src/dataset/`（build_samples+label_rules+visualize+rot6d，合成数据 e2e 过）⑤ `src/encoding/`（10 单测过，CPU torch 实测）⑥ `src/mamba/train.py`（LSTM 基线 15 epoch 冒烟过，loss 正常下降）⑦ `tests/check_contracts.py` + `make_fixtures.py`（25 项自测全绿）
- **契约 2026-10-15 首轮冻结**：样本 label 分离键（label_subject/object_slot/object_id/interaction + name）、subject 0=left/1=right/2=none、interaction 7 类索引、obj_mask/obj_valid/hand_valid、dataset_index 顶层 {train,val,test}、pose6d bboxes 未检出填 -1、clips csv 固定 6 列+残段记 meta.json
- 新增 `docs/DELIVERABLES.md` 交付物对齐表（每阶段交付什么、谁验收、当前阻塞）
- 工位精简：删除每人工位冗余的 `shared` 软链与 `/home/shared`（工位只保留 clone + TASK.md + data 软链）
- 遗留：④ build_samples 有 4 处待修（恢复帧速度 NaN、index 结构、contract_version、窗口 off-by-one）；GPU 仍未挂载；EgoDemo 申请中

## 2026-10-15 Daniel — 全环境终验通过（无 GPU 条件下）

- daniel_test 补齐 torch 2.6.0 + causal_conv1d/mamba_ssm wheel（与 caron_mamba 同栈）
- 修复三处：xw_dataset 补 pyyaml；panda_6d 补系统库 libegl1 + libusb-1.0-0（open3d import 通过）；fish 的 ffmpeg 改走 **apt 系统级安装**（conda 装 ffmpeg 解依赖会被 2GB cgroup OOM kill——conda 大求解一律避免，优先 uv/apt）
- 终验结果：ash_hands(mediapipe 1.1.1) ✅ / xw_dataset ✅ / panda_6d(torch+trimesh+open3d) ✅ / arc_enc(torch 2.14.1+cu126) ✅ / caron_mamba+daniel_test(torch 2.6.0 + mamba wheels 在位) ✅ / ffmpeg 4.4.2 系统级 ✅
- ⚠️ 唯一阻塞：GPU 仍未挂载进容器（无 /dev/nvidia*），mamba import 验证与 ③⑥ 训练任务等 GPU 就位后进行；gpufree 控制台操作待定

## 2026-10-15 Daniel — 全环境预配完成

- 7 个 conda 环境建好（全落数据盘）：fish_split(ffmpeg→已改 apt 系统级)、ash_hands(mediapipe 1.1.1)、xw_dataset、arc_enc/panda_6d/daniel_test(torch)、caron_mamba
- 修复 mediapipe 缺 libGL（apt 装 libgl1/libglib2.0-0）
- mamba-ssm 路线定案：torch 2.14 无预编译 wheel 且源码编译不兼容 → caron_mamba/daniel_test 定格 **torch 2.6.0+cu126 + causal_conv1d 1.5.2 + mamba_ssm 2.2.6.post3 预编译 wheel**
- 参考仓库预拉：panda 工位 FoundationPose + Depth-Anything-V2；caron 工位 mamba
- 服务器自带 nvcc CUDA 12.6（编译能力有，但优先预编译 wheel 原则不变）

## 2026-10-15 Daniel — 工位弹药备齐 + 数据集情报

- 每个成员目录配好独立仓库 clone（`~/egodata_process`，origin 走 gh-proxy，全部同步到最新）
- 服务器系统 Python 装好 modelscope 1.40.2 + huggingface_hub[hf_xet] 下载工具
- 数据盘 `ego_hoi/README.md` 说明目录分工；`raw/download_egodemo.sh` 下载脚本就绪
- **重要情报**：① EgoDemo 是申请制（gated-manual），fish 需先去 HF 申请访问；② ModelScope 的 ego_pose_dataset 目前是空壳（数据未上传）——已在 04_resources.md 标注
- 遗留：fish 的 HF 申请待批；ego_pose_dataset 待上传方补数据

## 2026-10-15 Daniel — hdk 目录归属确认并迁移

- `hdk` = Caron（误用真名缩写建目录），其 repo clone + conda env 已迁移至 `/home/caron@Caron/`，附 MIGRATION_NOTE.md（prefix 环境重建提醒、`src/encoding/` 代码落点约定、remote 已改代理前缀）

## 2026-10-15 Daniel — 成员目录实名制

- `/home/` 成员目录改名 `<缩写>@<群昵称>`（fish@🐠、ash@Prick1yAsh、panda@空腹panda、xw@谢武、arc@Arc、caron@Caron），新增 `/home/README.md` 总览表
- GitHub 已打通（deploy key + ssh.github.com:443）：Mac 可 push；服务器仓库 origin 配 gh-proxy.com 前缀只读拉取
- 遗留：`hdk` 目录归属未登记；`daniel` 目录群昵称待确认

## 2026-10-15 Daniel — 收录环境配置指南 + 服务器加速落地

- 《云端GPU服务器环境配置指南》（[gpu-rsi-plugin](https://github.com/miracle-techlink/gpu-rsi-plugin)）精华版收录为 `skills/env-setup/SKILL.md`
- 服务器落地加速项：全局安装 aria2c + uv；`HF_ENDPOINT`/`UV_INDEX_URL`/`UV_CACHE_DIR`(数据盘) 写入 `/root/.bashrc`
- 定位网络问题：服务器直连 GitHub git 端口 TLS 不稳 → clone 用 ghproxy 前缀（push 不可走代理）
- 遗留：BBR/sysctl 容器内无权限跳过；GitHub push 仍需凭据

## 2026-10-15 Daniel — 项目初始化

- 服务器环境：安装 Miniconda（`/opt/miniconda3`，清华镜像），conda 环境与数据集规划到数据盘 `/root/gpufree-data`
- 建立 `/home/` 下 7 个成员目录（fish / ash / panda / xw / arc / caron / daniel）+ `/home/shared`
- 仓库初始化：README、AGENTS.md、`.agent/`（memory×6 + skills×7）、`src/` 模块骨架
- **遗留问题**：
  1. ⚠️ 服务器 GPU 当前未挂载进容器（`nvidia-smi -L` 无输出）——训练前需在 gpufree 控制台确认
  2. 数据集尚未下载到 `/root/gpufree-data/datasets/`
  3. 《训练环境配置优化》文档链接待补充进 `04_resources.md`
