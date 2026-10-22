# TASK ① — 数据集切分（负责人：🐠 / 工位 `/home/fish@🐠`）

## 你的使命

把原始长视频切成 **~20 秒一段** 的 clip——这是全流水线的源头，所有人都在等你的产出。

## 输入 → 输出

| | 内容 | 位置 |
|---|---|---|
| 输入 | EgoDemo 原始长视频 | `data/ego_hoi/raw/EgoDemo/` |
| 输出 | `clips/<clip_id>.mp4`（18–22s）+ `clips_index.csv` | `data/ego_hoi/clips/` |

契约细节（字段、命名规则）：仓库 `.agent/memory/03_data_pipeline.md` ①节

## 当前状态

- ✅ 环境已预装：`conda activate fish_split`（ffmpeg 已装好）
- ✅ 下载脚本就绪：`data/ego_hoi/raw/download_egodemo.sh`
- ⚠️ **卡点：EgoDemo 是申请制数据集**，你需要：
  1. 打开 https://huggingface.co/datasets/LightwheelAI/EgoDemo 点 Request access（等批准邮件，通常几小时到一天）
  2. 批准后 HF Settings → Tokens 建 Read token
  3. `export HF_TOKEN=hf_xxx` 后跑下载脚本
  4. 全集约 50 小时视频 > 数据盘 49G，**先只拉一个子集**：脚本后加 `--include "EgoStand/*"`
- 💡 等批准期间：可以先随便找一段 ego 视频（手机自拍一段桌面操作也行）把切分链路跑通

## 实施步骤

1. **D1 跑通**：写 `src/preprocessing/split_one.py`（ffmpeg 重编码精确切点，勿用 `-c copy`），切 1 个视频 → 交 Daniel 过契约校验
2. **D2 小批**：10 个视频批量切（`xargs -P 8` 并行），验证 csv 索引完整性（ffprobe 逐段校验时长/帧数）
3. **D3 全量**：子集全量切分，产出完整 clips_index.csv

## 验收标准

- [ ] `tests/check_contracts.py clips <clip>` 通过
- [ ] csv 每行时长/帧数与 ffprobe 实测一致
- [ ] 不足 10s 尾段已丢弃并备注
- [ ] 更新 `.agent/memory/05_progress_log.md`，提 PR（分支 `fish/dataset-split`）

## 求助

- 切分参数/格式问题 → Daniel；数据集访问问题 → 群里喊，大家一起想办法
- 手册：`.agent/skills/dataset-split/SKILL.md`

## 参考

- https://huggingface.co/datasets/LightwheelAI/EgoDemo

---

## 下一步开发方向（2026-10-22 Daniel 链路实测后）

**现状**：split_clips.py 已在真实 ffmpeg 上跑通；EgoDemo 访问权已拿到，全量 1.44TB（多机位），只能按子集工作。

1. **真实数据一律 `--reencode`**：实测 `-c copy` 关键帧漂移严重——clip 最长达 26.9s（超 22s 上限）、相邻 clip 重叠 3.5s。copy 模式仅留作快速预览，交付数据必须重编码精确切点
2. **多机位目录适配**：EgoDemo 结构是 `EgoProRaw/mp4/<场景>/<uuid>/<camera>/xxx.mp4`（head_left/head_right/left_wrist/right_wrist 四机位）。写一个 prepare 脚本：把 head_left 视频软链成平铺目录，文件名 `<场景>_<uuid>` 保证唯一（这就是 source_video 名，影响 split 隔离）
3. **机位选择评估**：head_left vs head_right 哪个手部视野更好，抽帧对比后定主用机位；wrist 机位留给 ② 做手部增强（与 ash 对齐）
4. **子集扩大策略**：当前先 10-15 场景 ≤30GB；后续按训练需求扩，注意 246G 盘容量
5. 校验：每批 clips 过 `tests/check_contracts.py clips`
