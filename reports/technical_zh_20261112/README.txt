完整Pipeline技术报告 · 单栏图文工作区版

入口：完整技术报告_单栏图文版.pdf（与main.pdf字节一致）。

按用户选定的三页配图预览样式排版：A4单栏、16mm页边距、10pt中文正文。19页、18张图，包含完整pipeline、感知原理与演示、模型、指标、RSI和附录。素材册demo_materials.pdf为5页，visual_check.pdf为三页配图预览；10段MP4在media/。

main.tex为正文，report_style.tex为模板，sections/perception.tex为感知章节。运行bash build.sh编译，需要XeLaTeX、BibTeX及TeX Live/Fandol；Overleaf选择main.tex与XeLaTeX。图像和原始统计记录随包保存，PDF编译无需网络或GPU。

源码与10段视频：完整技术报告_单栏图文版_源码与素材.zip。运行python3 scripts/package_report.py重新打包。图页码见素材清单.csv和evidence/figure_page_index.json；验证见evidence/artifact_validation.json。

本工作区版包含授权环境中的原始素材。可对外分享的公开版另见：
https://github.com/miracle-techlink/egodata_process/releases/tag/technical-report-2026-11-12
公开版用独立图示替代受限EgoDemo原帧，并保留来源说明。本目录完整素材包未对外上传。

2026-11-13 Demo更新：打开 demo_player.html 查看完整流程、原始交互三栏与手部/6D视频。完整报告21页；公开版20页。media/full-pipeline.mp4 已包含物体中文名称、实物缩略图和带语义的交互关系。原始EgoDemo媒体仅限授权环境，不能公开再分发。
