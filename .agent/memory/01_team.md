# 01 — 团队分工

| # | 成员 | 服务器目录 | 任务 | 输出物 | 参考 |
|---|------|-----------|------|--------|------|
| ① | 🐠 | `/home/fish` | 数据集处理：长视频切成 ~20s 一段 | `clips/*.mp4` + 索引 | [EgoDemo (HF)](https://huggingface.co/datasets/LightwheelAI/EgoDemo) |
| ② | Prick1yAsh | `/home/ash` | 手部姿态提取 | `hand_pose/*.npz` | [MediaPipe Hands 教程 (CSDN)](https://blog.csdn.net/g11d111/article/details/115539407) |
| ③ | 空腹panda | `/home/panda` | 物体 6D pose 提取 | `pose6d/*.npz` | [FoundationPose (NVlabs)](https://github.com/NVlabs/FoundationPose) |
| ④ | 谢武 | `/home/xw` | 构建切分训练数据 + 数据集可视化（与 ②③ 协作） | `samples/` + 可视化 | — |
| ⑤ | Arc | `/home/arc` | 交互关系编码器 / 解码器研究 | `src/encoding/` | — |
| ⑥ | Caron | `/home/caron` | Mamba 时序模型训练 | `checkpoints/` + 日志 | [mamba](https://github.com/state-spaces/mamba) |
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

服务器 `/home/` 下目录名按群昵称易读性取的缩写，如与实际姓名缩写习惯不符，直接改目录名并同步更新本文件即可。
