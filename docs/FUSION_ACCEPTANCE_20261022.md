# 真实手部 / 物体 6D 融合验收 — 2026-10-22

## 结论

**整体未通过（blocked），本轮不继续训练。** 已交付真实视频上的手部重投影、统一世界系/相机系诊断图及逐帧统计；没有这两段的真实物体 6D，不能称为手—物完整融合。图中的 RGB 三轴是相机轴，不是物体轴。

- 证据生成和程序回归通过，不代表姿态准确性通过。
- 左右手输入来自发布方标注，不是本项目新跑出的手部模型预测；其覆盖率不能冒充模型识别成功率。
- 未获得独立 2D 手部参考或物体 6D 真值。重投影精度、物体姿态准确率、手—物联合成功率全部为 `null`，不是 0%，更不是 100%。

## 范围与实测统计

| 项目 | 货架标签 | 整理电缆 | 合计 |
|---|---:|---:|---:|
| 总帧数 | 1,194 | 5,970 | 7,164 |
| 左手有效帧 | 1,183 | 5,966 | 7,149 / 99.7906% |
| 右手有效帧 | 1,194 | 5,965 | 7,159 / 99.9302% |
| 双手同时有效帧 | 1,183 / 99.0787% | 5,963 / 99.8827% | 7,146 / 99.7487% |
| 左手 21 点全部在图内 | 1,183 | 5,947 | 7,130 |
| 右手 21 点全部在图内 | 1,194 | 5,965 | 7,159 |
| 真实物体 6D 可用帧 | 0 | 0 | 0 / 7,164 |
| 可评估联合精度帧 | 0 | 0 | 0 |

有效指上游有效掩码与全部 21 个三维关节有限性一致；百分比分母为该列总帧数。两手有效帧的全部关节均为相机正深度。电缆左手末尾第 5951–5969 帧共 19 帧未全部投影到图内，不能把有限坐标理解为可见手部。

两段世界→相机→世界最大往返差分别为 `7.45e-16`、`6.81e-16` 米量级。这只是浮点逆变换一致性，不验证标定精度、尺度或姿态真值。

## 交付位置

本地根目录：`runs/fusion_acceptance_20261022_review/`。

服务器根目录：`/root/gpufree-data/datasets/ego_hoi/fusion_acceptance_20261022_review/`。

两段子目录为：

- 货架标签：`d057beac-dab1-43a1-9192-6b8b06abddaf/`
- 整理电缆：`ce22bfbe-3daa-4cac-b43a-7c202f6c87ee/`

每段提供 9 张 PNG、多视图连续预览 `hand_camera_preview.mp4`、全帧 `frames.csv`、单段 `acceptance.json`。根目录提供总报告、`object_pose_gate.json`、`artifact_validation.json`、`visual_review.json` 和测试/源码摘要。总计 **18 张 PNG、2 个 MP4、7,164 行逐帧数据**。

优先查看货架 `frame_000270.png`、电缆 `frame_000285.png`；负例查看电缆 `frame_005969.png`。视频覆盖原片 8–18 秒，约 10 fps，各 100 帧；不是全片融合视频。18 张图片尺寸均为 1600×1080；两个视频均完整解码 100 帧、尺寸 1280×864。CSV 的 `joint_pass` 和独立重投影误差保持空值。

## 标定与溯源

实际使用同一标注 episode 的 `meta/episodes/chunk-000/file-000.parquet`，不是 Raw 的相机参数或 Raw 帧号。输入视频及该 parquet 的 SHA-256、大小均匹配下载清单；暂存 sequence/meta 的摘要写入报告。清单原有 `manifest_ready=false` 不等于 payload 缺失，当前实际校验以 `payload_verified=true` 为准。

| 参数 | 货架标签 | 整理电缆 |
|---|---:|---:|
| fx = fy | 788.9683837890625 | 788.253173828125 |
| cx / cy | 960 / 728 | 960 / 728 |
| 图像宽 / 高 | 1920 / 1456 | 1920 / 1456 |
| 标注内参畸变系数 | 全零 | 全零 |

使用 `p_camera = R_c2w.T @ (p_world - camera_position_world)`、四元数 `wxyz`、OpenCV 相机系 x 右/y 下/z 前。单位米和旋转方向有发布方文档依据，且此前双目刚体诊断支持该旋转方向；本轮没有独立测量真实尺度或重新标定。

复核发现并修复两项来源检查缺口：文件仅在 dataset 根目录内还不够，现绑定具体任务/UUID 与 `head_left` 路径；时间戳仅递增还不够，现要求 `frame_index == 0..T-1`，并用 ffprobe 全片解码逐帧比较绝对 PTS（容差 0.1 毫秒、起始时间为零）。另核对标注 UUID、帧数和视频尺寸。错误相机、另一采集、整体错移时间戳和错移帧号均须拒绝；允许真实任务名称包含安全的子路径。此前全片证据见 `docs/VALIDATION_20261022_PHASE2.md`，本轮亦重新解码。上述检查仍不替代外部标定精度测试。

发布方原始依据：

- https://docs.lightwheel.net/egocentric_data/lerobot_data/format_specification
- https://docs.lightwheel.net/egocentric_data/mcap_data/algorithm/hand_pose
- https://docs.lightwheel.net/egocentric_data/mcap_data/sensor/rgb_camera/head

## 可视检查与门禁

助手逐图抽查货架第 270、660 帧，电缆第 285、5969 帧：接触阶段骨架与图中手部大体对应；遮挡/模糊时不能逐关节判定。电缆末帧左手在视野外，右手运动模糊。记录为定性检查，不是独立人工标注，更没有计算“4/4 验收成功率”。原有 42 条手腕跳变记录未因此全部关闭。

同 UUID 真实物体文件在预期 `pose6d_real/` 缺失，数据根目录下 `pose6d_real`、`pose6d`、`depth`、`meshes`、`masks` 均不存在；此结论只针对本次检查的路径，不声称服务器任何位置都没有资产。已有 `/home/daniel/pipeline_run/data/pose6d/demo01_000.npz` 是负例：

- 用目标段长度验收，直接因 602 与 1,194 帧不一致拒绝。
- 再用其真实 602 帧验收，虽然 602 帧矩阵数值有效，仍因 `pose_estimator=mock`、`depth_source=none`、缺少已审核来源/标定/时间证据拒绝；真实姿态覆盖保持 0。
- 门禁可达到的最佳状态仅为 `coverage_only`，不会仅凭来源声明或刚体矩阵合规输出精度通过。

## 回归与复现

本地与服务器完整回归均 **723 passed + 26 subtests passed**（分别 6.49 秒、10.13 秒）。新增测试覆盖几何变换/投影、无效数据、来源与刚体门禁、真实标定读取及源文件绝对路径回归；其中最后补充 51 项专门阻止错误采集/相机与时间错位。合成测试只验证代码行为，不计入真实姿态成功率。

服务器使用已有 `xw_dataset` 环境渲染，`daniel_test` 环境测试，未修改 base。复现时必须选择新的输出目录；命令生成证据后返回 **1** 是缺物体时的预期拒绝状态，不代表渲染失败：

```bash
cd /workspace/egodata_process
/root/gpufree-data/conda/envs/xw_dataset/bin/python -m src.dataset.render_fusion_acceptance \
  --manifest /root/gpufree-data/datasets/ego_hoi/annotated_pilot_20261022/selection.json \
  --dataset-root /root/gpufree-data/datasets/ego_hoi/raw/annotated_pilot_20261022 \
  --staging-root /root/gpufree-data/datasets/ego_hoi/annotated_pilot_20261022/staging \
  --output-dir /root/gpufree-data/datasets/ego_hoi/fusion_acceptance_repeat_NEW \
  --episode d057beac-dab1-43a1-9192-6b8b06abddaf \
  --episode ce22bfbe-3daa-4cac-b43a-7c202f6c87ee
/root/gpufree-data/conda/envs/daniel_test/bin/python -m pytest -q
```

## 下一关：先补真实物体，再谈成功率

1. **③物体负责人**：先定义可验收的刚性物体及稳定 ID；电缆形变和薄标签不能默认整段是刚体。优先选择清晰刚性目标作短片验证，获得真实 mask、米制深度与 mesh/可追溯几何来源，跑非 mock 估计。
2. **④融合负责人**：将真实 `T_object_to_camera` 接入相同帧/相同相机，生成 `T_object_to_world = T_camera_to_world @ T_object_to_camera`，再输出手—物同系视图。现有 renderer 明确是 hand/camera 诊断器，尚不接收物体文件。
3. **⑦验收负责人**：先固定独立参考、误差阈值、可见/遮挡分组和分母，再计算手部重投影与物体姿态误差及联合通过率；不在看到结果后调整阈值。补齐以前，保持整体 blocked，不启动正式 HOI 训练。
