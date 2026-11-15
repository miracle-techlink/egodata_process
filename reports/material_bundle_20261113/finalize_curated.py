from pathlib import Path
import json, csv, hashlib, html, re, shutil, subprocess
import pymupdf as fitz

WORK=Path(__file__).resolve().parent
OUT=WORK/'手物交互预测_精选定稿资料包'
figs=[
('full_pipeline.pdf','完整流程与监督边界','原理'),
('graph_architecture.pdf','动态交互图模型结构','原理'),
('hand_pose_principle.pdf','手部21点与坐标原理','原理'),
('public_hand_coordinates.pdf','世界系、相机系与图像投影','原理'),
('hand_object_rgbd.png','DexYCB真人手部与物体6D','实际输出'),
('pose6d_principle.pdf','物体6D注册与跟踪','原理'),
('hocap_four_objects_f180.png','HO-Cap四物体与手部输出','实际输出'),
('hocap_same_scene_pointcloud.png','同源真人场景：RGB、深度与点云','实际输出'),
('stereo_geometry_principle.pdf','双目重建与刚性拟合','原理'),
('public_surface_motion.pdf','刚体、部件与非刚体表示','原理'),
('hand_calibration_coverage.pdf','手部来源覆盖与标定状态','验证'),
('hocap_perception_validation.pdf','当前Demo的覆盖与数值验证','验证'),
('perception_runtime.pdf','同配置场景的性能优化','验证'),
('sample_timeline.pdf','历史窗口与未来标签','原理'),
('rsi_loop.pdf','RSI候选生成与验证选优','原理'),
('rsi_observed_history.pdf','实际RSI账本曲线','验证'),
('expanded_results.pdf','已核验扩展阶段模型结果','验证'),
('full_pipeline_demo.jpg','物体内容、交互关系与模型输出','实际输出'),
('full_pipeline_sequence.jpg','同一连续Demo的关键时刻','实际输出'),
]
gallery=[]
for name,title,kind in figs:
    src=OUT/'04_图片与图表'/name
    if name.endswith('.pdf'):
        doc=fitz.open(src);page=doc[0];scale=max(1.4,min(4,1700/page.rect.width))
        dst=src.with_suffix('.png');page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).save(dst)
    else:dst=src
    gallery.append({'name':name,'title':title,'kind':kind,'image':str(dst.relative_to(OUT)),'original':str(src.relative_to(OUT))})
(OUT/'04_图片与图表/选图清单.json').write_text(json.dumps(gallery,ensure_ascii=False,indent=2))
videos=[]
for name,title,note in [
('full-pipeline.mp4','完整流程 Demo','同一HO-Cap片段：物体名称、手部与6D、当前复核阶段、实际LSTM候选。'),
('raw_rgb.mp4','同源原始 RGB','与完整流程按帧对应的原始画面，供前后对照。'),
('hocap_four_objects.mp4','四物体与手部感知','同一HO-Cap片段的FoundationPose与手部叠加。'),
('hands_and_object_rgbd.mp4','真人手物联合感知','DexYCB补充样例，展示手骨架与物体6D。')]:
    rel='03_视频素材/'+name;p=OUT/rel
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(p)]))
    videos.append({'path':rel,'title':title,'note':note,'probe':probe})
(OUT/'03_视频素材/视频清单.json').write_text(json.dumps(videos,ensure_ascii=False,indent=2))
report='01_技术报告/手物交互预测_技术报告_精选修订版.pdf';guide='02_详细技术解读/技术解读.pdf'
rp=len(fitz.open(OUT/report));gp=len(fitz.open(OUT/guide))
cards=''.join(f'<article data-kind="{g["kind"]}"><span class="tag">{g["kind"]}</span><h3>{i+1:02d}　{g["title"]}</h3><a href="{g["image"]}" target="_blank"><img loading="lazy" src="{g["image"]}" alt="{g["title"]}"></a><a class="file" href="{g["original"]}" download>下载原图</a></article>' for i,g in enumerate(gallery))
buttons=''.join(f'<button data-video="{i}" aria-pressed="{"true" if i==0 else "false"}">{v["title"]}</button>' for i,v in enumerate(videos))
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>手物交互预测 · 精选资料</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:#173446;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1140px;padding:34px 22px;margin:auto}header{padding:32px;background:#163b49;color:white;border-radius:18px}h1{font-size:30px;line-height:1.3;margin:10px 0}h2{font-size:24px;margin:38px 0 14px}h3{font-size:18px;margin:8px 0 16px}p{margin:12px 0}a{color:#067c75}header a{color:#d1fff1}.actions,.switch,.filter{display:flex;gap:10px;flex-wrap:wrap}.actions a{padding:10px 16px;background:#e7fbf5;color:#175047;border-radius:8px;text-decoration:none}button{border:1px solid #c0d3d8;background:white;padding:10px 15px;border-radius:8px;color:#173446;cursor:pointer;font:inherit}button[aria-pressed=true]{background:#166b60;color:white;border-color:#166b60}video{width:100%;max-height:72vh;background:#111c2a;border-radius:12px;margin-top:18px}.note{color:#506875;font-size:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:20px 0}article{background:white;border:1px solid #dce5e8;border-radius:12px;padding:18px;min-width:0}article img{width:100%;height:260px;object-fit:contain}.tag{font-size:12px;color:#166b60;background:#e4f3ed;padding:4px 8px;border-radius:4px}.file{font-size:14px}.files{padding:18px;background:white;border-radius:12px;overflow-wrap:anywhere}[hidden]{display:none!important}footer{margin-top:40px;border-top:1px solid #ccd8dd;padding-top:20px;font-size:14px;color:#506875}@media(max-width:650px){main{padding:18px 12px}header{padding:22px}h1{font-size:25px}.grid{grid-template-columns:1fr}article img{height:auto;max-height:430px}}
</style><main><header><small>EGODATA / CURATED MATERIALS · 2026.11.13</small><h1>从手物感知到交互预测</h1><p>当前技术报告、详细解读、4段视频与19组核心配图。</p><div class="actions">'''
page+=f'<a href="{report}">技术报告 · {rp}页</a><a href="{guide}">详细技术解读 · {gp}页</a></div></header>'
page+='<h2>连续演示</h2><div class="switch">'+buttons+'</div><video id="player" controls preload="metadata" playsinline poster="04_图片与图表/full_pipeline_demo.jpg" src="03_视频素材/full-pipeline.mp4"></video><p id="video-note">'+videos[0]['note']+'</p><a id="video-download" href="03_视频素材/full-pipeline.mp4" download>下载当前视频</a>'
page+='<p class="note">完整流程采用HO-Cap跨数据集输入，真实概率按原样保留。画面中的当前交互阶段属于视觉复核提示；本片段没有交互真值，不用于计算约0.93的测试指标。</p><h2>配图与验证</h2><div class="filter"><button data-filter="全部" aria-pressed="true">全部</button><button data-filter="原理">方法原理</button><button data-filter="实际输出">实际输出</button><button data-filter="验证">指标验证</button></div><div class="grid">'+cards+'</div>'
page+='<h2>源码与复核</h2><div class="files"><p>报告源码在 <b>05_报告源码</b>，必要实验记录在 <b>06_实验依据</b>，实现与复核工具分别在 <b>07_实现源码</b>、<b>08_复核工具</b>。</p><p><a href="先读我.txt">阅读与复跑说明</a>　·　<a href="本次筛选说明.txt">素材筛选说明</a>　·　<a href="文件清单.json">文件清单与校验</a>　·　<a href="第三方来源与许可.txt">素材来源</a></p></div><footer>本页和两份PDF为本包修订版。<a href="https://egodata-full-pipeline-demo.miracle-techlink.chatgpt.site">公开Demo入口</a> 保留此前发布版本，图表以本包为准。<br>保留真实实验局限与模型输出；历史草稿和中间制作文件另在工作区归档。</footer></main>'
page+='<script>const videos='+json.dumps(videos,ensure_ascii=False)+''';const player=document.getElementById('player');document.querySelectorAll('[data-video]').forEach(b=>b.addEventListener('click',()=>{const v=videos[Number(b.dataset.video)];player.pause();player.removeAttribute('poster');player.src=v.path;player.load();document.getElementById('video-note').textContent=v.note;document.getElementById('video-download').href=v.path;document.querySelectorAll('[data-video]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)))}));document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('article').forEach(a=>a.hidden=b.dataset.filter!=='全部'&&a.dataset.kind!==b.dataset.filter);document.querySelectorAll('[data-filter]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)))}));</script></html>'''
(OUT/'00_素材导航.html').write_text(page)
(OUT/'先读我.txt').write_text(f'''手物交互预测｜精选修订资料包｜2026年11月13日

完整解压后打开 00_素材导航.html，无需安装环境即可查看PDF、视频与配图。

01_技术报告：当前单栏修订报告（{rp}页）。
02_详细技术解读：原理、特征、模型、RSI、评价与复现（{gp}页）。
03_视频素材：4段完整片段，统一H.264；不重复收录不同编码。
04_图片与图表：19组当前配图；矢量PDF附PNG便于直接使用，绘图源码另列。
05_报告源码：一套当前报告的LaTeX与参考文献，bash build.sh即可编译。
06_实验依据：相应阶段评估、RSI账本、Demo数组、缓存输入、图模型权重和同帧点云。
07_实现源码：详细解读所述模块及其导入依赖。
08_复核工具：文件校验、数组查看、缓存推理与三张修订图重绘。

在包根目录运行：
  python 08_复核工具/check_bundle.py
  python 08_复核工具/inspect_demo.py --frame 223
  python 08_复核工具/replay_model.py
  python 08_复核工具/replot_figures.py --output 重绘图表
  python 08_复核工具/replot_coverage.py --output 重绘图表

查看数组需要numpy；缓存推理需要PyTorch及本包实现所用依赖；重绘需要numpy、matplotlib和中文字体（--font可指定字体）。PDF源码使用XeLaTeX与BibTeX。缓存重放仅验证输出可复现，不是准确率评测。

结果口径：项目方确认最终全量约0.93；包内可直接复算的扩展阶段test micro-F1为0.9388532001。RSI验证成绩单独标明。当前HO-Cap Demo用于跨数据集接口展示，没有逐帧交互真值，不计算F1。

本包修订了报告、选图和离线导航；公开网站与GitHub仍保留此前发布版本。解压后以本包两份PDF为准。
此包不包含完整原始训练数据集和全部上游感知权重。素材来源与使用范围见第三方来源与许可.txt。
''')
(OUT/'本次筛选说明.txt').write_text('''本次交付只保留当前方法、展示与实验结果所需材料。

已移出本包：
1. 两套旧CVPR报告、三页排版预览和早期Demo规划稿。
2. 6段早期交互Demo、旧版手部坐标视频、机械臂后端视频，以及重复转码。
3. 乱码截图、旧版三栏界面、排版检查截图、网页调试截图和逐帧重复导出。
4. 与当前真人流程场景不一致的机械臂点云图，及旧版独立后端指标图。
5. 不同测试集混排的历史版本柱状图、未使用的旧模型结构图和旧交互示意图。
6. 部署状态日志、历史阶段文档、无关代码和整目录候选文件。

已替换：
1. 用当前HO-Cap第223帧的RGB、实测深度、点云和估计模型替换机械臂示例。
2. 用同一HO-Cap序列446帧的覆盖、平移和矩阵数值检查替换无关场景验证图。
3. 结构图从当前报告实现图导出，不沿用旧版图。
4. 重绘缺少标签的手部覆盖与标定图，补齐坐标、分母和数值；剔除修正前的旧标定摘要。

保留：
当前完整流程视频与全部帧、真实模型概率和缺失掩码；手部、6D、双目与图预测原理；支撑结果的同口径阶段评估及完整RSI账本。未按模型表现筛掉失败帧，未修改实验分数。

所有移出内容仍在工作区与原始全套归档中保存。它们不再作为本包推荐展示素材。
''')
# Build auxiliaries, old QA screenshots and duplicate report copies are not deliverables.
for p in list(OUT.rglob('*')):
    if not p.is_file():continue
    if p.suffix in {'.aux','.log','.toc','.out','.blg','.bbl','.pyc'} or p.name=='main.pdf':p.unlink()
for p in sorted(OUT.rglob('*'),key=lambda p:len(p.parts),reverse=True):
    if p.is_dir() and not any(p.iterdir()):p.rmdir()
manifest=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name not in {'文件清单.json','SHA256SUMS.txt'}:
        manifest.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/'文件清单.json').write_text(json.dumps({'edition':'2026-11-13-curated','videos':4,'figure_groups':19,'report_pages':rp,'companion_pages':gp,'files':manifest},ensure_ascii=False,indent=2))
(OUT/'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in manifest))
print(json.dumps({'files':len(manifest)+2,'bytes':sum(r['bytes'] for r in manifest),'report_pages':rp,'guide_pages':gp,'figures':len(figs),'videos':len(videos)},ensure_ascii=False))
