# 01 — 团队分工

| # | 成员 | 服务器目录 | 任务 | 输出物 | 参考 |
|---|------|-----------|------|--------|------|
| ① | 🐠 | `/home/fish@🐠` | 数据集处理：长视频切成 ~20s 一段 | `clips/*.mp4` + 索引 | [EgoDemo (HF)](https://huggingface.co/datasets/LightwheelAI/EgoDemo) |
| ② | Prick1yAsh | `/home/ash@Prick1yAsh` | 手部姿态提取 | `hand_pose/*.npz` | [MediaPipe Hands 教程 (CSDN)](https://blog.csdn.net/g11d111/article/details/115539407) |
| ③ | 空腹panda | `/home/panda@空腹panda` | 物体 6D pose 提取 | `pose6d/*.npz` | [FoundationPose (NVlabs)](https://github.com/NVlabs/FoundationPose) |
| ④ | 谢武 | `/home/xw@谢武` | 构建切分训练数据 + 数据集可视化（与 ②③ 协作） | `samples/` + 可视化 | — |
| ⑤ | Arc | `/home/arc@Arc` | 交互关系编码器 / 解码器研究 | `src/encoding/` | — |
| ⑥ | Caron | `/home/caron@Caron` | Mamba 时序模型训练 | `checkpoints/` + 日志 | [mamba](https://github.com/state-spaces/mamba) |
| ⑦ | Daniel（+ 协作同学） | `/home/daniel` | 集成测试、debug、优化所有 part | 测试报告、修复 PR | — |

## 协作依赖

```
①fish → ②ash → ④xw → ⑤arc → ⑥caron
        ③panda ↗              ↘
                ⑦daniel 贯穿全流程测试/优化
```

- ④ 依赖 ②③ 的输出格式，开工前先和他们对齐 `.agent/memory/03_data_pipeline.md`
- ⑤ 的编码器输入即 ④ 的样本结构，二者是同一份契约的两端
- 任何人改动接口格式：**先改契约文件，再 @ 上下游**，见 AGENTS.md 铁律

## 目录命名说明

- 格式：`<缩写>@<群昵称>`（如 `ash@Prick1yAsh`），`ls /home` 一眼定位归属；`/home/README.md` 有总览表
- 进入带 @/中文/emoji 的目录：输入开头字母后按 **Tab** 自动补全，不要手打全名；脚本里一律用 `~/` 或软链，不硬编码家目录路径
- **conda 环境名保持纯 ASCII**（如 `ash_hands`），与目录名解耦，不带 @/中文/emoji
- `hdk` 目录暂未登记归属；`daniel` 目录待本人确认群昵称后再改名
