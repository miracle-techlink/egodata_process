# AGENTS.md — 项目入口

> 无论你是新成员还是 AI agent，按下面的顺序读，不要跳。

## 必读顺序（约 10 分钟）

1. `.agent/memory/00_project_overview.md` — 课题目标与技术路线
2. `.agent/memory/01_team.md` — 分工表：谁做什么、在哪做
3. `.agent/memory/02_server_guide.md` — 服务器使用规范（必守）
4. `.agent/memory/03_data_pipeline.md` — **数据接口契约**：各环节输入/输出格式
5. `.agent/memory/04_resources.md` — 数据集、代码、参考文献链接汇总
6. `.agent/memory/05_progress_log.md` — 当前进度（开工前先对齐）

## 按任务查手册

`.agent/skills/` 下每个目录对应分工中的一个任务，内含 SKILL.md：

| 任务 | skill 目录 |
|---|---|
| ① 数据集切分 | `dataset-split` |
| ② 手部姿态提取 | `hand-pose` |
| ③ 6D pose 提取 | `pose6d` |
| ④ 训练数据构建+可视化 | `build-training-data` |
| ⑤ 编码器/解码器 | `encoder-decoder` |
| ⑥ Mamba 训练 | `mamba-training` |
| ⑦ 测试/debug/集成 | `integration-test` |
| VLM 语义预标注与人工审核 | `vlm-annotation` |
| 通用：环境搭建/下载加速 | `env-setup` |

## 任务卡（每人一份：目标/步骤/验收标准）

`docs/tasks/task01_*.md` ~ `task07_*.md`，与分工一一对应；公用代码统一在 `/workspace/egodata_process`，各工位的 `TASK.md` 是对应任务卡的实体副本。

## 铁律

1. **接口改动先改契约**：任何输入/输出格式变化，先改 `03_data_pipeline.md` 并在 PR 描述中 @ 上下游负责人
2. **完工必更新** `05_progress_log.md`（日期、做了什么、产出路径、遗留问题）
3. 服务器规范见 `02_server_guide.md`：自己的 `/home` 目录、独立 conda 环境、不污染 base
4. 合入主仓库的代码进 `src/` 对应模块，通用工具函数进 `src/common/`；工位只保留本人任务的实验/测试代码与产物，不放整仓 clone
