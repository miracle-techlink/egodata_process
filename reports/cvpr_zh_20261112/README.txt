中文CVPR技术报告｜全流程结果对齐稿
2026-11-12

已纠正实验范围
旧稿只看了10月24日五段试点，漏掉后续扩充、132维特征、多轮RSI与现有demo。旧稿已撤回，放在archive_initial_subset，不纳入当前交付ZIP。
项目方确认最终全量结果约0.93。本稿已按完整技术链重写；最终精确指标键、冠军run、全量索引和RSI完整轮数仍待与服务器记录绑定。
已恢复的10月26日扩展冻结阶段原始评估输出为：test micro-F1，Mamba 0.905，LSTM 0.939，Transformer 0.939。这是历史阶段结果，不能仅因接近0.93就命名为最终冠军。

交付
- main.pdf / main.tex：中文CVPR双栏技术报告，含120/132维特征、图预测、自动复核、RSI与结果版本分析。
- demo_materials.pdf / .tex：现有六段视频盘点、优先补齐素材、八类图表安排与110秒分镜。
- 素材清单.csv：UTF-8 BOM，可用Excel打开；已有与待同步分开列出。
- figures/：系统结构、RSI机制、阶段结果、版本历史与真实demo预览五类图；结构图有独立TikZ源文件。
- media/：已有六个MP4的原样副本；最新本地为demo_v6.mp4（1500×500，8秒，24帧，3fps）。部分中文方框仍需使用原始渲染数据修复。
- evidence/：选定原始工具输出、代码快照、视频哈希与来源清单。只保存相关摘录，不包含完整会话或连接凭据。

资料边界
0.93来自本轮项目方说明；0.939来自已保存的扩展阶段工具输出；最终RSI冠军尚未直接读取。
现有demo为LSTM预测，画面显示阈值0.80；必须与benchmark评估阈值区分。手和物体姿态是观测，交互边才是预测。
132维包含离线几何候选标记，不能直接声称严格因果的全自动在线系统。
不跨数据版本复用全正参照，不将不同测试集的柱值差解释为RSI净收益。

编译
bash build.sh
依赖：TeX Live、XeLaTeX、BibTeX、Fandol中文字体。
Overleaf上传ZIP，主文档main.tex，编译器XeLaTeX；素材文档可切换demo_materials.tex。
所有图已生成，编译不需要Python、GPU或服务器。
cvpr.sty及ieeenat_fullname.bst来自工作区CVPR2026_IR_draft，样式文件未修改。
官方模板：https://github.com/cvpr-org/author-kit
重画统计图：python3 scripts/revise_figures.py（matplotlib、numpy、Noto Sans CJK SC）。
独立结构图：xelatex -output-directory=figures figures/architecture.tex；rsi_loop.tex同理。

回收最新记录
在能够访问原文件的服务器终端运行：
python3 scripts/collect_remote_evidence.py --runs-root /root/gpufree-data/runs --output /root/gpufree-data/report_evidence_20261112
默认收集rsi_final、rsi_auto、graph_v5_auto、rsi_v5。可用--series显式限定系列；用--data-root追加冠军配置绑定的数据发布目录。
仅复制小型记录并计算checkpoint哈希，不加载模型、不训练、不修改原run、不发送网络请求，拒绝覆盖既有输出。运行中的文件不保证原子快照，需检查各run原本绑定的哈希。本次未在服务器执行。

当前环境的SSH连接被沙箱阻止，提权也被权限策略自动拒绝。最新远端结果尚未核验。
本地已完成排版与构建检查；未提交、未推送、未公开发布。
