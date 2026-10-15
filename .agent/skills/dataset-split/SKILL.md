---
name: dataset-split
description: 任务① 将原始长视频切成约 20 秒一段的 clip，产出 clips/ 与索引 csv。负责人 fish。
---

# 视频切分（dataset-split）

## 契约（详见 .agent/memory/03_data_pipeline.md ①）

- 输入：原始长视频（EgoDemo 下载到 `data/ego_hoi/raw/`）
- 输出：`data/ego_hoi/clips/<clip_id>.mp4`（~20s）+ `clips_index.csv`
- `clip_id = <原视频名>_<3位段号>`，如 `demo01_007`

## ⚠️ 数据集获取（开工前置）

EgoDemo 是**申请制数据集**（gated-manual）：
1. 浏览器打开 https://huggingface.co/datasets/LightwheelAI/EgoDemo 点 Request access，等批准邮件
2. 批准后到 HF Settings → Tokens 建一个 Read token
3. 服务器上 `export HF_TOKEN=hf_xxx`，跑 `data/ego_hoi/raw/download_egodemo.sh`
4. 全集约 50 小时视频，数据盘仅 49G——**先只拉一个子集**（脚本里加 `--include "EgoStand/*"`）

modelscope 与 huggingface_hub(xet) 已装在服务器系统 Python，直接用。

## 推荐做法（ffmpeg）

**stream copy（快，但切点只能落在关键帧，时长有漂移）：**
```bash
ffmpeg -i in.mp4 -ss 00:00:00 -t 20 -c copy -avoid_negative_ts make_zero out.mp4
```

**重编码（切点精确，推荐用于训练数据）：**
```bash
ffmpeg -ss <start> -i in.mp4 -t 20 -c:v libx264 -preset fast -crf 20 -c:a aac out.mp4
```

批量并行（按 CPU 核数调 `-P`）：
```bash
ls data/raw/*.mp4 | xargs -P 8 -I{} python src/preprocessing/split_one.py {}
```

## 要点

- 切分脚本写成 `src/preprocessing/split_one.py`（单个视频）+ `run_split.sh`（批量），参数化段长（默认 20，范围 18–22）
- 每段写完后立刻 `ffprobe` 校验实际时长与帧数，写进 `clips_index.csv`
- 结尾不足 10s 的残段丢弃，并记录到 csv 备注列
- 先拿 1 个视频跑通全流程，把样本交给 ⑦daniel 用 `tests/check_contracts.py` 校验，再批量跑

## 环境

```bash
conda create -n fish_split python=3.10 -y && conda activate fish_split
# ffmpeg 用系统级（apt 已装好 4.4.2，直接可用）：
# ⚠️ 不要 conda install ffmpeg —— 本容器 2GB cgroup 内存限制会让 conda 解依赖被 OOM kill
ffmpeg -version
```
