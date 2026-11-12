# 完整Pipeline技术报告 · 连续Demo更新

中文单栏图文报告，包含手部姿态、物体6D、双目原理、交互预测、指标验证与多轮RSI。

- **[在线播放：完整流程、手部与6D](https://egodata-full-pipeline-demo.miracle-techlink.chatgpt.site)**
- **[打开更新版PDF](https://egodata-full-pipeline-demo.miracle-techlink.chatgpt.site/egodata-technical-report-zh.pdf)**
- [下载PDF](https://github.com/miracle-techlink/egodata_process/releases/download/technical-report-2026-11-12/egodata-technical-report-zh.pdf)
- [GitHub代码](https://github.com/miracle-techlink/egodata_process)
- [ModelScope预留入口](https://www.modelscope.cn/datasets/Liuyue9698/ego_pose_dataset)

2026-11-13新增同一HO-Cap真人序列上的RGB、手部/四物体6D、132D历史编码与LSTM实际关系输出：446帧连续回放，32帧历史后得到415帧推理。另有DexYCB手物联合和HO-Cap四物体感知两段视频，可在线播放、慢放或下载。

这是跨数据集接口演示，源时戳未核验、特征时基假设30Hz，没有交互真值，不对应约0.93测试结果。报告分别保留扩展阶段0.938853的原始评估、RSI验证记录与项目方最终约0.93的陈述。EgoDemo原始演示保留在授权工作区。

ModelScope在2026-11-12检查中仅有README.md和.gitattributes，尚无数据文件。来源、修改说明与许可见THIRD_PARTY_NOTICES.txt。旧发布tag仍保留11月12日快照；本页与Release附件为11月13日更新版。
