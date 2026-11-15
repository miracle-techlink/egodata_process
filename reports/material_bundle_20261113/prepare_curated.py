from pathlib import Path
import re, shutil, json

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
OLD=WORK/'手物交互预测_全套素材与详细技术解读'
OUT=WORK/'手物交互预测_精选定稿资料包'
PUB=ROOT/'reports/public_technical_zh_20261112'
OUT.mkdir(exist_ok=True)
def cp(src, dest):
    dest=OUT/dest; dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)
def write(path, s):
    p=OUT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)

# Keep one report source; remove superseded draft comparisons and off-topic visuals.
main=(PUB/'main.tex').read_text()
main=main.replace('数据构建与版本演进','数据构建与实验版本')
main=re.sub(r'初始V5（10月24日）.*?\\\\\n','',main)
main=re.sub(r'自动复核试点（10月25日）.*?\\\\\n','',main)
main=main.replace('项目演进，斜杠依次为train/val/test。','本文涉及的数据版本，斜杠依次为train/val/test。')
start=main.index('\\subsection{不同版本之间的比较原则}')
end=main.index('\\section{完整流程的连续Demo与可视化}',start)
main=main[:start]+r'''\subsection{评价口径与基线}
不同数据版本的测试采集、监督分布和输入条件不同。RSI收益应比较同一冻结数据、同一特征和同一评估口径下的初始配置与冠军。全正预测的参照值为$F_1^{all+}=2P/(2P+N)$，其中$P,N$为该版本已知监督的正负单元数；应与相同测试集的模型结果一起计算，不能跨版本比较。

'''+main[end:]
start=main.index('\\subsection{原始交互演示与补充感知片段}')
end=main.index('\\section{复现与分析要求}',start)
main=main[:start]+r'''\subsection{配套演示与离线素材}
精选资料包提供四段视频：HO-Cap完整流程、同源原始RGB、同源四物体感知，以及DexYCB真人手物联合感知。前三段各446帧，DexYCB片段44帧；均保留完整片段与真实输出。打开包内“00\_素材导航.html”即可离线播放。当前物体关系阶段提示来自视觉复核，实际模型概率另列；未对模型失败帧进行筛除。

'''+main[end:]
main=main.replace('系统还完成了多版三栏关系图demo。','系统提供了包含物体内容、当前交互与实际预测的连续Demo。')
start=main.index('\\textbf{FoundationPose后端示例：}')
end=main.index('\\section{复现和仍需补充的验证}',start)
main=main[:start]+main[end:]
main=main.replace('本报告是项目技术研究说明。','本报告为2026年11月13日素材筛选版。点云和数值验证改用当前HO-Cap真人场景，移除了历史版本对比图与机械臂后端样例；原始阶段指标保持不变。')
main=main.replace('公开版不包含EgoDemo原始画面','本版不包含EgoDemo原始画面')
main=main.replace('figures/','../04_图片与图表/')
write('05_报告源码/main.tex',main)
style=(PUB/'report_style.tex').read_text().replace('单栏公开版 · 连续Demo更新','单栏图文版 · 素材筛选修订').replace('（单栏公开版）','（素材筛选修订版）')
write('05_报告源码/report_style.tex',style)
per=(PUB/'sections/perception.tex').read_text()
per=per.replace('pose6d_pointcloud.png','hocap_same_scene_pointcloud.png')
per=re.sub(r'\\caption\{FoundationPose独立后端验证：.*?\}\n',r'\\caption{与完整流程Demo同源的HO-Cap第223帧：原始RGB、实测深度、反投影点云和估计物体模型。点云由该帧深度及相机内参计算；模型顶点仅应用已保存的FoundationPose位姿，没有利用真值重新对齐。三维视图采用横轴相机X、纵轴相机Z、竖轴负相机Y，单位为米。图示说明坐标对应，不作为独立姿态精度评测。}\n',per)
per=re.sub(r'FoundationPose后端 &.*?\\\\\n','',per)
per=per.replace('perception_validation.pdf','hocap_perception_validation.pdf')
start=per.index('图\\ref{fig:perceptionvalidation}直接从737')
end=per.index('\\begin{figure}',start)
per=per[:start]+r'''图\ref{fig:perceptionvalidation}直接使用当前HO-Cap完整Demo的446帧四物体位姿与手部有效掩码。左图给出四物体与左右手的输出覆盖；中图展示主要操作对象的相机系平移；右图检查四物体旋转矩阵的正交性与行列式。缺失手部观测保留为无效，不补造骨架。平移变化是估计轨迹，矩阵偏差是数值自洽检查，两者都不是与真值比较的误差。

'''+per[end:]
per=re.sub(r'\\caption\{逐帧感知验证。.*?\}\n',r'\\caption{当前HO-Cap序列的输出覆盖、红盖圆瓶平移轨迹和四物体旋转矩阵数值检查。覆盖分母为446帧；矩阵曲线每帧取四物体的最大偏差。全部帧参与统计，覆盖与矩阵合法性均不等于独立精度。}\n',per)
per=per.replace('figures/','../04_图片与图表/')
write('05_报告源码/sections/perception.tex',per)
for n in ['references.bib','ieeenat_fullname.bst']:cp(PUB/n,'05_报告源码/'+n)
write('05_报告源码/build.sh','#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\nxelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass1.log\nbibtex main > build_bibtex.log\nxelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass2.log\nxelatex -interaction=nonstopmode -halt-on-error main.tex > build_pass3.log\nmkdir -p ../01_技术报告\ncp main.pdf ../01_技术报告/手物交互预测_技术报告_精选修订版.pdf\n')

guide=(OLD/'02_详细技术解读/技术解读.tex').read_text()
guide=guide.replace('../04_图片与图表/当前报告插图/','../04_图片与图表/')
guide=guide.replace('本册与20页公开技术报告、21页授权完整版配套。','本册与本包中的单栏技术报告配套。')
guide=guide.replace('\\textbf{本地完整材料包：}包含授权工作区中的原始演示与证据。公开网页只含可公开的HO-Cap／DexYCB素材；具体来源和使用范围见包内许可文件。此包不是全部原始训练数据集的镜像。','\\textbf{精选资料包：}只保留当前完整流程、真人手物与四物体感知材料，以及相应方法图和结果依据。旧版Demo、机械臂后端示例、历史报告和排版截图已移出；真实模型输出和必要实验记录保留。此包不是全部原始训练数据集的镜像。')
guide=guide.replace('包内不同版本的\\code{training_index.json}及审核记录。','数据发布时绑定的索引及审核记录（完整训练索引另在授权工作区保存）。')
guide=guide.replace('\\fig[.91]{pose6d_pointcloud.png}{真实后端样例的RGB、深度点云和估计mesh。此样例含机械臂，不能冒充真人第一视角实验。}', '\\fig[.98]{hocap_same_scene_pointcloud.png}{同一HO-Cap第223帧的RGB、实测深度、反投影点云与估计模型。模型仅使用保存的估计位姿，没有按真值对齐。三维显示轴为相机X、相机Z与负相机Y，单位为米。}')
guide=guide.replace('实际候选表面图与源码保存在素材目录。','图为原理示意；实际同帧RGB-D点云见上图。')
guide=guide.replace('早期小子集的全正参照约0.5354是该历史版本的重要诊断，但\\textbf{不能跨数据版本拿它否定后来全量与多轮RSI的结果}。','全正参照依赖所用测试集的监督分布，不能跨数据版本与最终全量及RSI结果比较。')
start=guide.index('\\section{素材索引：每段视频适合说明什么}')
end=guide.index('\\section{如何离线浏览和复核实际输出}',start)
guide=guide[:start]+r'''\section{素材索引：当前四段视频}
\begin{longtable}{p{5.6cm}p{1.5cm}p{6.3cm}}
\toprule 文件 & 时长约 & 内容与用途\\\midrule\endhead
\code{full-pipeline.mp4} & 14.87秒 & 物体名称、手部／6D、当前交互与实际LSTM输出\\
\code{raw_rgb.mp4} & 14.87秒 & 同一HO-Cap片段的原始RGB，供前后对照\\
\code{hocap_four_objects.mp4} & 14.87秒 & 同一片段的四物体6D及手部感知\\
\code{hands_and_object_rgbd.mp4} & 1.47秒 & DexYCB真人手骨架与物体6D联合感知\\\bottomrule
\end{longtable}
四段视频均保留完整序列，以H.264兼容编码提供，不重复收录相同内容的原始编码。图表目录只收录当前报告和本册实际使用的图；矢量图另提供PNG便于演示文稿插图。源码目录保留一套可重新编译的当前报告。

'''+guide[end:]
guide=guide.replace('另可打开“08\\_公开网页离线副本”目录中的\\code{index.html}，直接浏览公开演示的离线副本。','')
guide=guide.replace('10_复核工具/','08_复核工具/')
guide=guide.replace('第05目录的两套\\code{build.sh}','第05目录的\\code{build.sh}')
guide=guide.replace('图像和矢量图依赖已放在对应\\code{figures}目录。','图像和矢量图依赖统一放在第04目录。')
guide=guide.replace('第07目录是本地实现快照，不等同于GitHub主分支的历史版本；','第07目录只收录当前方法说明与复跑所需的实现快照，不等同于GitHub主分支；')
guide=guide.replace('三个系列的\\code{history.jsonl}、候选\\code{score.json}、\\code{genome.yaml}','三个系列的\\code{history.jsonl}、\\code{genome.yaml}与汇总CSV')
guide=guide.replace('公开页与本地PDF是否同版 & 发布状态、线上哈希核验记录、包内SHA256清单','本地修订材料是否完整 & 包内文件清单及SHA256；在线入口仍保留此前发布版本')
guide=guide.replace('本包完整提供本次报告和演示所需的已整理视觉材料、指标记录与图模型复核输入','本包提供经过筛选的当前报告和演示素材、必要指标记录与图模型复核输入')
guide=guide.replace('原始素材、实际日志','精选素材、必要实验记录')
write('02_详细技术解读/技术解读.tex',guide)
cp(OLD/'02_详细技术解读/build.sh','02_详细技术解读/build.sh')

# Figures are admitted by actual document references, not directory traversal.
names=set(re.findall(r'includegraphics\[[^\]]*\]\{\.\./04_图片与图表/([^}]+)\}', main+per))
names.update(re.findall(r'\\fig(?:\[[^\]]*\])?\{([^}]+)\}',guide))
for name in sorted(names):
    if name.startswith('hocap_same_scene_') or name=='hocap_perception_validation.pdf':continue
    cp(PUB/'figures'/name,'04_图片与图表/'+name)
    tex=(PUB/'figures'/name).with_suffix('.tex')
    if tex.exists():cp(tex,'04_图片与图表/绘图源码/'+tex.name)
write('04_图片与图表/选图清单.json',json.dumps(sorted(names),ensure_ascii=False,indent=2))
for dest,src in [('full-pipeline.mp4','full-pipeline.mp4'),('raw_rgb.mp4','raw_rgb_browser.mp4'),('hocap_four_objects.mp4','hocap_four_objects.mp4'),('hands_and_object_rgbd.mp4','hands_and_object_rgbd_browser.mp4')]:
    cp(OLD/'03_视频素材'/src,'03_视频素材/'+dest)

e=OLD/'06_实验依据/报告证据'
for name in ['rsi_series_summary.json','rsi_observed_curve.csv','expanded_benchmark_20261026.csv','perception_metrics_derived.json','user_correction_20261112.json']:
    cp(e/name,'06_实验依据/报告证据/'+name)
for name in ['calibration_summary.json','hand_rgbd_report.json','hocap_acceptance.json','hocap_report.json','performance_trials.json','rgbd_frames.csv','refinement_report.json','shelf_acceptance.json','cable_acceptance.json']:
    cp(e/'perception'/name,'06_实验依据/报告证据/perception/'+name)
for model in ['lstm','mamba','transformer']:
    d=Path('remote/runs/graph_v5_auto')/(model+'_auto_v4')/'eval'
    for name in ['test_benchmark.json','val_benchmark.json']:cp(e/d/name,Path('06_实验依据/报告证据')/d/name)
for series in ['rsi_v5','rsi_auto','rsi_final']:
    d=Path('remote/runs')/series
    for name in ['history.jsonl','genome.yaml']:cp(e/d/name,Path('06_实验依据/报告证据')/d/name)
d=OLD/'06_实验依据/Demo推理与制作'
for name in ['demo_arrays.npz','hocap_sequence.npz','history_inputs.npz','lstm_final_private.pt','demo_manifest.json','model_vocabulary.json','缓存模型复跑验证.json','infer_hocap.py','render_demo.py']:
    cp(d/name,'06_实验依据/Demo推理与制作/'+name)
# Current implementation is useful for the detailed companion and cache replay.
for f in (OLD/'07_实现源码/src').rglob('*.py'):cp(f,Path('07_实现源码/src')/f.relative_to(OLD/'07_实现源码/src'))
for n in ['requirements.txt','pyproject.toml']:
    if (OLD/'07_实现源码'/n).exists():cp(OLD/'07_实现源码'/n,'07_实现源码/'+n)
for n in ['check_bundle.py','inspect_demo.py','replay_model.py']:cp(OLD/'10_复核工具'/n,'08_复核工具/'+n)
cp(PUB/'THIRD_PARTY_NOTICES.txt','第三方来源与许可.txt')
print(json.dumps({'root':str(OUT),'figure_count':len(names)},ensure_ascii=False))
