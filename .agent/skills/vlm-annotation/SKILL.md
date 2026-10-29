# VLM 语义预标注与审核

## 目标与边界

用本地视觉语言模型减少 HOI 标注工作量，服务于未来交互主体、客体和类型预测。VLM 只生成待审核语义候选，不生成稳定物体 track，不单独认证几何，不直接产生训练真值，也不用于补齐历史缺失帧。

唯一接口事实源：`.agent/memory/03_data_pipeline.md` 的 `hoi_annotation_v1`。修改字段前先改契约并通知②③④⑤⑥⑦。

## 环境与模型

- 独立 conda 环境：`/root/gpufree-data/conda/envs/daniel_vlm`，不安装到 base。
- 基线模型：`Qwen/Qwen2.5-VL-7B-Instruct`，权重放 `/root/gpufree-data/models/`，不放根盘或 git。
- 兼容栈：Python 3.10、torch 2.6.0+cu124、transformers 4.56.2、SDPA；不强制 flash-attn。
- 公共模型优先 ModelScope 国内 CDN；其他 HF 资产使用已配置的镜像与授权。凭据只放运行时配置，不写代码、日志或仓库。

## 必需流程

1. 核对采集 UUID、相机和 native timeline。EgoDemo Raw 与 annotated 视频不得按帧号混用；时间戳缺失时先解决上游，不假定 30 FPS。
2. 抽取有限证据帧，保存实际帧号、PTS 和图像 SHA。模型权重、revision、prompt SHA、解码参数、原始响应和运行环境全部留档。
3. VLM 输出 left/right、语义客体描述、六类实际交互、粗略边界和证据帧。未知可以留空或判 invalid，不能强行分类；无交互区间与未标注区间严格区分。
4. 将语义客体与既有稳定 `object_id` 链接。VLM 的物体描述不是 track ID；没有轨迹时保持 null。
5. 加入手腕—bbox、接触、运动、遮挡等几何证据。坐标不一致或缺深度/K 时状态保持 `not_checked/inconclusive`，不宣称完成 3D 验证。
6. 人工查看边界、证据帧和连续预览，执行 accepted/rejected/needs_revision。accepted 必须有 reviewer、reviewed_at 和真实 object_id；保留原候选及修订历史。
7. 只有 accepted 事件或 accepted none interval 才能投影到旧三头标签。重叠 accepted 事件、pending、未知、遮挡、ID 未链接均 `label_valid=0`，不进入 loss 或指标。
8. 正式发布还需不可变 freeze、training_manifest、时间/几何/资产/标签质量门禁及采集级 split。VLM 候选数量、滑窗数量均不能抵扣 120 个独立合格采集。

## 在线因果限制

离线标注默认可看整个事件，只能作为训练监督或审核材料，`allowed_as_model_input=false`。若把 VLM 特征用于模型输入，必须另行执行只含当前及历史帧的因果推理，保存每个窗口的输入帧清单；不得复用看过未来的离线结果。

## 验收口径

分别报告：模型实际成功运行的采集数、原始/结构有效候选数、track 链接数、几何状态、accepted/rejected/pending 数、显式 none 与无效帧数。mock 或 replay 单测只证明接口，不证明真实 VLM 已运行。自动预标注准确率需要独立人工抽样对照，不能用模型自报 confidence 替代。
