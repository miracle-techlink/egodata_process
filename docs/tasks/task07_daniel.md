# TASK ⑦ — 集成测试 / debug / 优化（负责人：Daniel / 工位 `/home/daniel`）

## 你的使命

全链路的质量闸门：契约校验工具、端到端冒烟、debug 支援、性能优化。你不属于单一环节，你属于**整条流水线**。

## 职责清单

1. **契约守门**：`tests/check_contracts.py` 校验 ①–④ 所有产物（字段/shape/dtype/NaN 规则/meta 必备项）
2. **冒烟链路**：1 个 clip 走完 ①→⑥，任何环节首批产出先过你这关再放行下游
3. **debug 支援**：复现包规范（报错全文 + `conda env export` + 最小样本）；修别人模块的 bug 必补回归测试
4. **优化**：正确性 > 吞吐 > 显存；链路全对之后才谈加速

## 当前状态

- ✅ 环境已预装：`conda activate daniel_test`（torch + numpy + opencv）
- ✅ 服务器基建、知识库、GitHub 通道已就绪（见进度日志）
- ⏳ 待办：`tests/check_contracts.py` 实现（各环节 checker）

## check_contracts.py 设计

```bash
python tests/check_contracts.py clips <mp4|目录>
python tests/check_contracts.py hand_pose <npz|目录>
python tests/check_contracts.py pose6d <npz|目录>
python tests/check_contracts.py samples <目录>
```

每类 checker 断言：文件可读、字段名逐字匹配契约、shape/dtype、NaN 规则、meta.json 必备字段（fps/帧数/来源/生成时间）、csv 列名与实测一致（ffprobe 抽查）。

## 实施步骤

1. **D1**：写 clips checker（fish 开工即要用）
2. **D2**：hand_pose / pose6d checker（②③ 交付前就绪）
3. **D3**：samples checker + 全链路冒烟脚本
4. 持续：环境配置支援（conda/wheels/网络——`.agent/skills/env-setup`）

## 验收标准

- [ ] 四类 checker 实现并各有通过/失败用例自测
- [ ] M1 里程碑：①→④ 全链路冒烟通过
- [ ] 进度日志持续更新

## 参考

- 手册：`.agent/skills/integration-test/SKILL.md`（含 PR checklist 模板）
- 契约唯一事实源：`.agent/memory/03_data_pipeline.md`

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：①→⑥ 全链路已端到端跑通（切分→手部→mock pose→313 样本→LSTM 训练），各阶段契约校验全过；check_contracts 25 项自测全绿。

1. **校验器增强**：clips 模式增加"相邻 clip 时间重叠检测"（① copy 模式漂移暴露的问题）；meta.json 键名与 ②③ 统一后把 warning 收成 error
2. **真实数据全链路**：EgoDemo 子集下载完成后，从真实 clip 重跑 ①→⑥，重点看手部检出率与标签分布
3. **mamba vs LSTM 对照**：真实 samples 上跑对照实验并出报告
4. 遗留研究问题跟踪：坐标系统一（②③④⑤）、gaze 先验接入、EgoDemo manifest 内参解析
