# egodata_process

**课题**：基于动态人-物交互建模的行为意图预测
**Research on Action Intention Prediction Based on Dynamic Human-Object Interaction Modeling**

通过视觉感知、人体与手部状态估计、目标物体空间运动信息，构建人-手-物动态交互关系，基于 **Mamba 时序网络**预测下一时刻可能发生交互的主体、客体及交互类型，并探索视线等先验信息的增强作用。

## 仓库结构

```
.
├── AGENTS.md          # Agent / 新成员入口，先读这个
├── .agent/            # 项目知识库：memory（共识与进度）+ skills（各任务开发手册）
├── src/
│   ├── preprocessing/ # ① 数据集切分（长视频 → ~20s 片段）
│   ├── hand_pose/     # ② 手部姿态提取
│   ├── pose6d/        # ③ 物体 6D pose 提取
│   ├── dataset/       # ④ 训练样本构建与可视化
│   ├── encoding/      # ⑤ 交互关系编码器 / 解码器
│   └── mamba/         # ⑥ Mamba 时序模型训练
└── tests/             # ⑦ 集成测试 / 接口校验
```

## 快速上手

1. 阅读 `AGENTS.md`，再按 `.agent/memory/` 编号顺序读完 5 个文件（10 分钟）
2. 在 `.agent/skills/` 找到自己任务的 SKILL.md，按手册开发
3. 服务器上在 `/home/<你的目录>` 工作，conda 独立环境，完工后提 PR
4. 完工 checklist 见 `.agent/skills/integration-test/SKILL.md`

## 资源

- 处理好的数据集（ModelScope）：https://www.modelscope.cn/datasets/Liuyue9698/ego_pose_dataset
- 原始数据参考（HuggingFace）：https://huggingface.co/datasets/LightwheelAI/EgoDemo
- 服务器：4090 24GB（gpufree），使用规范见 `.agent/memory/02_server_guide.md`
