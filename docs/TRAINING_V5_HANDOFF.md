# V5 交互预测训练子集交付说明（2026-10-24）

> 验收状态（2026-10-24）：用户确认“复核验收完成”，已记录为本次 V5 交互训练子集（5段、1238窗口）的交付验收。300段完整标注和整物体6D仍为未完成；逐帧标签来源保持原记录。验收凭据与冻结数据包并列保存为 `egodemo_training_v5_300_20261024_user_acceptance.json`。

**本包可以用于小规模交互预测训练；300 段的手姿态＋整物体 6D＋交互联合全标注尚未完成。** 本次交付将实际确认的助手银标、过去可见的二维实例观测和官方手姿态接入真实 Mamba 图预测头。原 300 段、75 原任务、868255 原生帧及 210/45/45 采集划分保持原样，未把子集替代全部数据。

服务器目录：`/root/gpufree-data/datasets/ego_hoi/egodemo_training_v5_300_20261024`。
本地镜像：`runs/egodemo_training_v5_300_20261024`。
私有数字包：同级 `egodemo_training_v5_300_20261024_interaction_subset.tar.gz`，附 `.sha256`。

## 可用范围与实际数量

| 划分 | 独立采集 | 窗口 | 正监督单元 | 负监督单元 | 有正例的类型 |
|---|---:|---:|---:|---:|---:|
| train | 3 | 592 | 1953 | 1452 | 12 |
| val | 1 | 366 | 1172 | 406 | 10 |
| test | 1 | 280 | 786 | 1364 | 16 |
| 合计 | 5 | 1238 | 3911 | 3222 | 并集 21 |

窗口高度重叠，1238 个窗口不是1238段独立视频。66 是词表容量；实际监督并集21类，为8操作、6关系、3转换、4协作。训练只有12类正例，另9类只在val/test出现：对齐、翻转、持物、提起、放置、按压、稳定支撑、互补协作、交接（精确代码见 `training_release.json`）。评估应分别报告训练已见和未见类型；目前不构成充分、多样或盲测的课题结果集。

连续实际查看1995个唯一原生帧，其中1991帧至少有一条明确状态；212个状态段展开为8430个正值、6320个负值。物理转换42个确认发生的有界范围、8个删失范围，精确物理转换帧仍为0；另有2个精确照明变化，不代表扫码成功。范围发生标签与目标帧状态监督分开。

105个当前帧种子产生2667个候选物体帧，全部97页/1136个有框原生帧实际复核后，接受2357个、拒绝310个。拒绝包括手腕漂移、局部框及严重遮挡；缺失观测未补造。二维框仅为定性实例位置，桌面按可见支撑区域记录，不认证精确轮廓或物体6D。

## 文件与准入

| 文件 | 用途 |
|---|---|
| `training_index.json` | 冻结窗口、实体输入、标签、划分和来源SHA |
| `training_release.json` | 仅允许本交互子集训练；全量联合与6D认证仍为false |
| `sequences/<UUID>/hands.npz` | 原生时间戳、官方左右手3D、有效mask、相机到世界变换 |
| `visual_observations/<UUID>/` | 当前种子、因果重放框、RGB统计、手投影、逐对象接受/拒绝 |
| `dense_reviews/` | 中文实体、持续关系、转换范围和未知原因 |
| `graph_windows/{train,val,test}.jsonl` | 保持原采集划分的1238个窗口 |
| `sources/`、`all_300_acquisitions.json` | 原划分/词表/来源及300段保留清单 |
| `model_checks/` | 真实batch、解码示例、全窗口统计、FP32/BF16验收 |
| `training_index_audit.json`、`review_seed_audit.json` | 全窗口及原生像素证据审计 |
| `tracking_evidence/`、`review_web/` | 已查看联系表和中文只读页面 |
| `pose6d_feasibility/` | 因果6D实测失败原因；不进入训练输入 |
| `PACKAGE_SCOPE.json`、`delivery_manifest.json` | 包含范围、逐文件SHA和版本绑定 |

默认加载检查准入凭据、窗口及实际读取的数组/标注SHA。修改任何冻结输入后应创建新版本并重新审计，不能只改 `interaction_training_allowed`。第一次准入发现并修复了缺证据键、非首段文件校验遗漏；原报告保存在 `admission_history/v1`，当前入口使用修复后的凭据。

数字训练子集自包含，不需要旧V2大数组或原视频来加载；包内代码入口位于 `code_snapshot`。联系表包含在包内，原视频和逐帧PNG裁剪仍在私有服务器；重新做完整像素来源审计需要原V2/V3/V4及源视频。继承的V4证据在压缩包中为普通文件，不依赖服务器绝对软链接。

## 编码与解码

历史为32个原生帧，预测目标位于历史末帧至少1秒之后，不插值。整个历史到目标必须处于连续复核范围。物体须在历史里至少8帧有接受的观测，按首次历史出现帧及稳定ID选最多8槽；未来才出现的物体不进入历史。

输入 `entity_features[B,32,11,120]`：11实体为左手、右手、双手组合和8物体槽。120维为归一化框4、过去速度2、框内RGB均值/标准差6、手部二维关节42、手部三维关节63及3个有效/种子标记。物体的手关节字段为0，双手组合不伪造骨架；缺失由 `entity_valid` 表示。手3D以最后观测帧左相机为锚点，单位米。离线双向估计的物体6D不进入Encoder，角色名称和目标标签也不进入输入。

逐帧共享投影及实体注意力得到128维实体token，按每个实体的32帧时间轴交给共享真实Mamba；有向实体对头输出 `graph_logits[B,11,11,66]`。左/右/双手→物体以及物体→物体分别保留方向，避免把两个物体合并成一个槽。对象槽置换不改变物理含义。

`graph_values` 与 `graph_label_valid` 同形。只有mask=true的明确正/负值进入BCE；mask=false处的0是未知占位，绝不能直接当负例。状态标签描述目标帧；转换标签仅表示完整确认的事件发生范围落在预测时间段内，不解码为精确接触时刻。

解码对每个有效实体对、每种类型单独sigmoid，返回中文类型、主体/客体ID及概率；不是互斥softmax。几何本版没有预测监督，不输出“有效整物体6D”。随机初始化的解码示例仅检查字段，不是可用预测结果。

## 加载与最小训练接入

服务器已验证环境：`/root/gpufree-data/conda/envs/caron_mamba/bin/python`，PyTorch 2.6.0+cu124、mamba_ssm 2.2.6.post3、NVIDIA L40。数字加载需要NumPy/PyTorch，真实Mamba执行需要兼容CUDA的mamba_ssm。不要直接传入旧七类 `src.mamba.train`。

```python
import torch
from torch.utils.data import DataLoader
from src.mamba.dataset_v5 import EgoDemoGraphV5Dataset, collate_graph_v5
from src.mamba.graph_v5 import GraphV5Mamba, masked_graph_bce, decode_graph

root = "/root/gpufree-data/datasets/ego_hoi/egodemo_training_v5_300_20261024"
ds = EgoDemoGraphV5Dataset(root, split="train")  # 默认检查实际准入，无需diagnostic
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_graph_v5)
batch = next(iter(loader))
batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
model = GraphV5Mamba(input_dim=120, event_type_count=66).cuda()
out = model(batch)
loss_info = masked_graph_bce(out["graph_logits"], batch["graph_values"],
    batch["graph_label_valid"], out["graph_prediction_mask"])
loss_info["loss"].backward()
decoded = decode_graph(out["graph_logits"], ds.event_types,
    batch["metadata"], prediction_mask=out["graph_prediction_mask"])
```

上例只算梯度，不执行参数更新。本次交付也没有正式训练或效果指标。实际验收batch4共25个明确监督单元，FP32/BF16梯度有限且非零，未知单元直接梯度0；未来标签/metadata污染检查及物体槽置换检查通过。全1238窗口另经独立逐项审计。

解压到其他位置后，在Python启动前设置 `PYTHONPATH=<解压目录>/code_snapshot`，把示例root改成解压数据目录即可。真实GPU复现命令 `python -m src.mamba.graph_v5_smoke --root <数据目录>` 会在该目录写验收产物；冻结交付目录只读使用，重新验收请在副本中执行。

## 中文网页

运行SSH转发：

```bash
ssh -p 30314 -L 18791:127.0.0.1:18791 root@120.209.70.195
```

然后打开 `http://127.0.0.1:18791/review_web/index.html`。页面显示范围、各split统计、5段当前框、接受/拒绝决定、中文状态及事件。缺框或缺图明确显示，不使用邻帧冒充；浏览不会生成human批准。旧18790/V4页面保留。

## 300段完整目标的剩余工作

仍有866260帧未连续视觉复核，296段尚未整段看完；全部标注层完成的帧数仍为0。原官方手姿态2238帧至少一手缺失，340485帧没有任意场景几何候选。已有离线场景/部件候选不能抵充交互对象的可靠整物体6D，非刚体也不能强行赋刚体位姿。

此次对刚体手柄的首帧双目与单向PnP实测：可信表面各仅8个匹配点，低于原12点门槛；扩大bbox虽能得到红19/白13点，却混入手与背景，前32帧仅产生红5帧/白1帧的未认证候选，故没有放入训练。后续需当前帧表面分割与更密可靠双目对应、持续跟踪、身份核对和独立位姿验收；不能靠复制邻帧或降低门槛宣布补齐。

交互子集可开始验证模型训练流程。课题的完整联合训练数据仍需继续按300段逐项补证据、刚体/非刚体几何与边界，并扩充训练中的实际类型和独立采集数量。
