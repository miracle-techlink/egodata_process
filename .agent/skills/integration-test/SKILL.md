---
name: integration-test
description: 任务⑦ 全链路集成测试、接口契约校验、debug 与优化规范。负责人 Daniel。
---

# 集成测试 / debug（integration-test）

## 核心工具：`tests/check_contracts.py`

校验任意环节产物是否符合 `.agent/memory/03_data_pipeline.md`：

```bash
python tests/check_contracts.py clips data/ego_hoi/clips/demo01_007.mp4
python tests/check_contracts.py hand_pose data/ego_hoi/hand_pose/demo01_007.npz
python tests/check_contracts.py pose6d data/ego_hoi/pose6d/demo01_007.npz
python tests/check_contracts.py samples data/ego_hoi/samples/train/
```

检查项：文件存在、npz 字段名、shape/dtype、NaN 规则、meta.json 必备字段、csv 列名。

## 冒烟流程（端到端）

1. 选 1 个 clip，手动走 ①→②→③→④ 产出 1 个 sample
2. ⑤ encoder/decoder 前向 1 个 batch（随机权重）
3. ⑥ Mamba 跑 10 step，确认 loss 下降、显存正常
4. 任何一步失败：先 `check_contracts.py` 定位断点环节，再带最小复现找对应负责人

## debug 规范

- **复现包**：报错信息全文 + `conda env export > env.yml` + 最小输入样本 + 运行命令
- 修别人模块的 bug：修完补一个回归测试到 `tests/`，PR 里 @ 原负责人
- 接口格式问题不允许单方面"适配"——回契约文件讨论（AGENTS.md 铁律 1）

## PR checklist（所有人完工通用）

- [ ] 代码在 `src/` 对应模块，无绝对路径、无硬编码密码
- [ ] 自测：契约校验通过 + 冒烟相关环节通过
- [ ] 更新 `.agent/memory/05_progress_log.md`
- [ ] 若动了接口：已先更新 `03_data_pipeline.md` 并 @ 上下游
- [ ] 大数据/权重/checkpoint 未提交进 git（.gitignore 已挡，仍自查 `git status`）

## 优化优先级

正确性 > 吞吐 > 显存。先保证契约全链路正确，再谈加速（并行、混合精度、缓存）。
