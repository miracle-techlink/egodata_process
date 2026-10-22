# TASK ⑤ — 交互关系编码器/解码器（负责人：Arc / 工位 `/home/arc@Arc`）

## 你的使命

设计"人/手/物 → token 序列"的编码器和"token → 下一时刻交互"的解码头。你是数据与模型之间的桥梁——⑥caron 的 Mamba 直接 import 你的模块。

## 输入 → 输出

| | 内容 |
|---|---|
| 输入 | ④ 的样本（`hand_seq`、`obj_seq`），可选 gaze 通道 |
| 输出 | `src/encoding/` 模块：`build_encoder(cfg)`、`build_decoder(cfg)` |

接口约定（caron 直接依赖，**改签名必须先回契约+通知**）：

```python
from src.encoding import build_encoder, build_decoder
enc = build_encoder(cfg)   # (B,W,...) 样本 -> (B,W,d_model)
dec = build_decoder(cfg)   # 时序表示 -> dict(subject=, object=, type=) logits
```

## 当前状态

- ✅ 环境可用：`conda activate arc_enc`（torch 2.14.1+cu130）
- ✅ 样本契约与 `build_encoder/build_decoder` 接口已冻结；10 项单测及 LSTM/Mamba 集成冒烟通过
- ⏳ 下一步是接 ④ 的真实 samples 做 overfit 与坐标单位联调，不再等接口冻结

## 设计要点（已实现，细节见契约 ⑤节）

- **实体 token**：每帧 {人体全局, 左手, 右手, obj_1..N}；token = 自身状态 + **关系特征**（与其他实体的相对位置/距离、相对速度、contact 标志）+ 实体类型 embedding + 时间 embedding
- **关系层**：1–2 层轻量 cross-attention；d_model=128 起步——**初版宁小勿大**，跑通再加深
- **预测头**：subject 3 类（none/左/右）、object N_max+1 类、type 按标签体系
- **gaze 预留**：`gaze_seq=None` 可选参数（课题要求的增强探索）
- 全部超参走 cfg（yaml），方便 caron 扫参

## 实施步骤

1. **D1**：按契约 shape 用随机张量写完 encoder/decoder + shape 单测（`tests/`）✅
2. **D2**：和 caron 联调前向（随机权重 loss 能 backward）✅
3. **D3**：接 xw 真实样本做 overfit，保持已冻结接口不变

## 验收标准

- [x] shape 单测全过（含 N 变化、NaN mask 输入不炸）
- [ ] 真实样本上 overfit 小实验 loss 显著下降（与 caron 配合）
- [ ] 更新进度日志，提 PR（分支 `arc/encoding`）

## 求助与参考

- 手册：`.agent/skills/encoder-decoder/SKILL.md`
- 数据侧问题找 xw，训练侧问题找 caron，接口争议回契约文件

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：encoder/decoder 10 单测全过（含前向反向、mask、gaze 开关）；接口已冻结进契约（实体顺序/类别索引/build_encoder/build_decoder 签名）。LSTM 基线已带你的编码器在 L40 上训练冒烟通过。

1. **真实样本联调**：④ 的真实 samples 出来后跑 overfit 小实验（单 batch 能过拟合 = 链路没问题）
2. **关系特征的单位问题**：手部归一化坐标 vs 物体米制坐标混算距离/contact 目前不合理——短期方案：contact_threshold 按归一化空间单独设、或对两类特征分组归一化；长期等 ④ 坐标系统一
3. **gaze 先验落地**：接口已预留（use_gaze/gaze_dim），下一步定义 gaze_seq 的数据来源与格式（与 ② 或新视线估计模块对齐），这是课题"进一步增强"的加分项
4. 参数量目前 ~350K 很小，真实数据后可按欠拟合情况加深
