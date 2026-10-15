---
name: encoder-decoder
description: 任务⑤ 设计动态人-物交互关系的编码器/解码器（人/手/物 → token → 下一时刻交互预测头）。负责人 Arc。
---

# 交互关系编码器 / 解码器（encoder-decoder）

## 目标

把 ④ 的样本（`hand_seq`、`obj_seq`）编码成 token 序列，供 ⑥ Mamba 做时序建模；解码器输出下一时刻**主体 / 客体 / 交互类型**三路预测。

## 设计要点【草案，实现后回写契约】

### 实体 token 化

- 每帧实体集合：{人体全局状态, 左手, 右手, obj_1..obj_N}
- 每实体 token = 自身状态特征（关键点展平 / 6D pose 特征）+ **关系特征**：
  - 与其他实体的相对位置、距离
  - 相对速度（相邻帧差分）
  - contact 标志（手-物距离阈值 / 学习得到）
- 线性投影到 d_model 后加实体类型 embedding + 时间 embedding

### 交互关系建模

- 候选：轻量 cross-attention（1–2 层）或图消息传递；输出关系感知的实体表示序列 → Mamba
- 视线（gaze）先验：预留一路可选输入通道（课题要求探索），接口上先留 `gaze_seq=None` 参数

### 解码器 / 预测头

- 取最后时刻（或注意力池化）表示：
  - `subject_head`：3 类（none/left/right）
  - `object_head`：N_max+1 类（含 none）
  - `type_head`：交互类型数（标签体系见契约④）

## 模块接口（⑥ 直接依赖，勿擅自改签名）

```python
# src/encoding/__init__.py
def build_encoder(cfg) -> nn.Module   # 输入 hand_seq, obj_seq, gaze_seq=None -> (B, W, d_model)
def build_decoder(cfg) -> nn.Module   # 输入时序表示 -> dict(subject=, object=, type=) logits
```

## 要点

- 纯 PyTorch 实现，不引重型依赖；单测覆盖 shape（`tests/`）
- 维度、层数、dropout 全部走 `cfg`（yaml），方便 ⑥ 扫参
- 初版宁小勿大：d_model 128、2 层以内，先让 ⑥ 跑通再加深

## 环境

```bash
conda create -n arc_enc python=3.10 -y && conda activate arc_enc
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
