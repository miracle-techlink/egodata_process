from pathlib import Path
import json,subprocess,hashlib,csv,html,urllib.parse,shutil
from collections import Counter
WORK=Path(__file__).resolve().parent;OUT=WORK/'手物交互预测_全套素材与详细技术解读'
def href(path):return urllib.parse.quote(str(path),safe='/')
def title(path):return html.escape(str(path))
# Compilation intermediates are not useful source materials.
for folder in [OUT/'02_详细技术解读',OUT/'05_报告源码/公开版',OUT/'05_报告源码/授权完整版']:
 for p in folder.glob('*'):
  if p.suffix in {'.aux','.log','.out','.toc','.blg','.bbl'}:p.unlink()
video_notes={
 'full-pipeline.mp4':'最新完整流程：物体名称与实物图、手部/6D、当前交互与模型候选',
 'raw_rgb.mp4':'同一HO-Cap原始RGB回放，作为对照',
 'demo_v6.mp4':'EgoDemo原始三栏交互预测（授权工作区）',
 'demo_v5.mp4':'交互预测历史版本 v5（授权工作区）',
 'demo_pose_v4.mp4':'位姿与交互显示历史版本 v4（授权工作区）',
 'b857b2a8_demo.mp4':'早期交互演示（授权工作区）',
 'b857b2a8_demo_v2.mp4':'早期交互演示改进版（授权工作区）',
 'e923ec44_demo.mp4':'另一早期交互样例（授权工作区）',
 'hands_ego_coordinates.mp4':'第一视角手部世界/相机坐标（授权工作区）',
 'hands_and_object_rgbd.mp4':'DexYCB真人手与物体6D联合感知',
 'hocap_four_objects.mp4':'HO-Cap四物体完整感知分支',
 'pose6d_backend_full.mp4':'FoundationPose完整后端样例，包含机器人，73.7秒'}
videos=[]
for p in sorted((OUT/'03_视频素材').glob('*.mp4')):
 info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate','-of','json',str(p)]))
 stream=info['streams'][0];base=p.name.replace('_browser.mp4','.mp4');note=video_notes.get(base,base)
 if '_browser' in p.name:note+=' · H.264浏览器兼容版'
 videos.append({'path':str(p.relative_to(OUT)),'name':p.name,'description':note,'duration_seconds':float(info['format']['duration']),**stream})
(OUT/'视频索引.json').write_text(json.dumps(videos,ensure_ascii=False,indent=2))
with (OUT/'视频索引.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(videos[0]));w.writeheader();w.writerows(videos)
images=[]
for p in sorted((OUT/'04_图片与图表').rglob('*')):
 if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.svg','.pdf'}:
  images.append({'path':str(p.relative_to(OUT)),'name':p.name,'kind':p.suffix,'group':p.relative_to(OUT/'04_图片与图表').parts[0]})
readme='''手物交互预测：全套素材与详细技术解读（2026-11-13）

先完整解压，再打开 00_素材导航.html。视频、图片与PDF均使用相对路径；不要在压缩软件里只打开单个HTML。

01_技术报告：20页最新版公开报告、21页授权完整版、手部与6D配图预览、Demo素材规划。
02_详细技术解读：新增22页技术解读PDF及XeLaTeX源码，含坐标变换、6D/双目、120/132维字段、三骨干、稀疏监督、RSI、指标复算和Demo解读。
03_视频素材：全部16个视频文件，包含6版原始交互、手部坐标、真人手物、四物体、完整6D后端、新完整流程及原RGB；兼容转码不算新实验。
04_图片与图表：当前图表、原理图、矢量PDF/TikZ、原始Demo预览、连续帧、逐帧感知输出和历史排版素材。
05_报告源码：两套独立报告源码与依赖，进入子目录运行 bash build.sh。
06_实验依据：原始阶段评估、三个RSI账本、感知记录、Demo数组、415个完整历史输入、该演示使用的LSTM权重。
07_实现源码：当前本地Python实现快照、配置、报告制作脚本及阶段文档；不是GitHub main的同版保证。
08_公开网页离线副本：已经发布的完整演示页面，可离线观看3段公开视频与报告。
09_历史报告：早期CVPR版本只用于溯源；不要把其小子集结论代替最新实验。
10_复核工具：校验SHA256、查看指定帧和用缓存输入复跑LSTM。已实际复核415个窗口，最大概率差2.98e-7，掩码一致，无训练。

结果口径：项目方确认最终全量约0.93；已恢复扩展阶段test micro-F1为0.938853；后续RSI的val与test分开报告。公开HO-Cap Demo是跨数据集流程展示，不是上述0.93测试视频。

范围：本包汇总本次报告与Demo的全部已整理素材，另含图模型的可复核输入；不包含整个原始训练数据集、所有RSI权重和上游感知模型大权重。旧脚本的原环境路径作为溯源信息保留；直接播放和缓存图推理不依赖原服务器。

使用范围：本地完整包包含EgoDemo授权工作区材料，不作为公开分发包。对外分享请使用已上线网页、公开PDF及其注明许可的素材。各来源权利不被本包重新许可。

公开网页：https://egodata-full-pipeline-demo.miracle-techlink.chatgpt.site
GitHub发布：https://github.com/miracle-techlink/egodata_process/releases/tag/technical-report-2026-11-12
ModelScope预留入口：https://www.modelscope.cn/datasets/Liuyue9698/ego_pose_dataset （既有检查仅模板文件，未在本次上传数据）

文件清单.json / 文件清单.csv：逐文件SHA256、大小、类别与来源。
重复来源映射.json：报告不同版本间完全一致的素材只收一份时，记录原路径到收录路径的对应。
'''
(OUT/'先读我.txt').write_text(readme)
shutil.copy2(WORK/'deduplicated_aliases.json',OUT/'重复来源映射.json')
for rel,target in [('01_技术报告/demo_player.html','../00_素材导航.html'),('05_报告源码/授权完整版/demo_player.html','../../00_素材导航.html')]:
 (OUT/rel).write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={target}"><a href="{target}">打开完整素材导航</a>')
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>手物交互预测 · 全套素材导航</title><style>
*{box-sizing:border-box}body{background:#f2f4f2;color:#172e3d;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif;max-width:1400px;margin:auto;padding:30px}h1{font-size:36px;line-height:1.3}h2{font-size:26px;margin-top:45px}a{color:#066d57;text-underline-offset:3px}button,select,input{font:inherit;border:1px solid #b7c7c2;border-radius:7px;background:white;padding:10px;color:#172e3d}select{width:100%;margin-bottom:16px}button{cursor:pointer}.links{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.links a{padding:11px 16px;background:#fff;border:1px solid #d4dfda;border-radius:7px;text-decoration:none}video{display:block;width:100%;max-height:78vh;background:#101826;border-radius:10px}.note{font-size:14px;color:#53636c}.info{background:white;border-left:4px solid #168568;padding:18px 22px;margin:20px 0}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{border-bottom:1px solid #dce4df;text-align:left;padding:10px;overflow-wrap:anywhere}details{margin:14px 0;background:white;border:1px solid #d6dfdb;padding:14px;border-radius:8px}summary{cursor:pointer;font-weight:bold}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-top:20px}.tile{background:white;border:1px solid #d7e0db;border-radius:7px;padding:12px;overflow-wrap:anywhere;font-size:13px}.tile img{width:100%;height:150px;object-fit:contain;background:#f8faf8}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}#filter{width:min(100%,650px)}footer{border-top:1px solid #ccd8d2;margin-top:40px;padding-top:20px}@media(max-width:650px){body{padding:16px}h1{font-size:28px}.grid{grid-template-columns:1fr 1fr}table{font-size:12px}th,td{padding:6px}.tile img{height:105px}}
</style></head><body><p class="note">EGODATA · 本地完整材料包 · 2026-11-13</p><h1>手物交互预测<br>全套素材与详细技术解读</h1><p>从物体身份、手部与6D，到交互图、时序模型与RSI。先读解读，再看对应素材，最后核对原始记录。</p><div class="links">'''
links=[('02_详细技术解读/技术解读.pdf','新增：22页详细技术解读'),('01_技术报告/最新公开技术报告.pdf','20页公开技术报告'),('01_技术报告/授权完整版技术报告.pdf','21页授权完整版'),('08_公开网页离线副本/index.html','公开网页离线副本'),('先读我.txt','先读我与目录说明'),('文件清单.csv','完整文件清单')]
for path,label in links:page+=f'<a href="{href(path)}">{title(label)}</a>'
page+='''</div><div class="info"><strong>阅读口径</strong><br>项目方最终全量约0.93、已恢复扩展阶段test 0.938853、后续RSI验证值分别报告。HO-Cap公开Demo用于跨源流程展示。当前视觉复核阶段与模型预测分开显示。<br><span class="note">本包含授权工作区中的原始EgoDemo材料；对外分享使用公开网页及注明许可的素材。</span></div><h2>视频素材 · 全部版本</h2><select id="choose" aria-label="选择视频"></select><video id="player" controls playsinline preload="metadata"></video><div class="controls"><button id="slow">0.5× 慢放</button><button id="normal">1× 正常</button><a id="download">打开／保存原文件</a></div><p id="description"></p><p id="metadata" class="note"></p><details><summary>查看全部视频文件、时长与编码</summary><table><thead><tr><th>文件</th><th>内容</th><th>时长</th><th>帧数 / 编码</th></tr></thead><tbody>'''
for v in videos:page+=f'<tr><td><a href="{href(v["path"])}">{title(v["name"])}</a></td><td>{title(v["description"])}</td><td>{v["duration_seconds"]:.2f}s</td><td>{v["nb_frames"]} / {v["codec_name"]}</td></tr>'
page+='''</tbody></table></details><h2>图片、原理图与指标图</h2><p>可按文件名或分组搜索。矢量PDF可直接打开，绘图源码与图表放在对应目录。</p><input id="filter" type="search" placeholder="搜索：hand、pose、rsi、pipeline、原理图…" aria-label="搜索图片"><p id="imageCount" class="note"></p><div class="grid" id="gallery"></div><h2>技术解读包含什么</h2><ol><li>任务、三种运行条件、采集级划分与32帧历史窗口。</li><li>21点手部、世界/相机坐标、缺失掩码与深度表面点。</li><li>FoundationPose注册跟踪、点云、双目三角化、RANSAC/SVD与非刚体。</li><li>物体身份、正负未知标签、120/132维字段及有向图头。</li><li>LSTM、Mamba、因果Transformer、稀疏损失与F1复算。</li><li>RSI真实账本、2%选优门槛、性能分析、Demo逐帧解读和复现。</li></ol><h2>文件与复核入口</h2><ul>'''
for path,label in [('06_实验依据/Demo推理与制作/缓存模型复跑验证.json','415个窗口的实际模型复跑结果'),('06_实验依据/Demo推理与制作/demo_manifest.json','Demo来源、语义阶段及模型绑定'),('视频索引.csv','视频索引CSV'),('文件清单.json','逐文件大小与SHA256'),('第三方来源与许可.txt','来源与许可'),('重复来源映射.json','重复文件来源对应')]:page+=f'<li><a href="{href(path)}">{title(label)}</a></li>'
page+='''</ul><footer><a href="https://egodata-full-pipeline-demo.miracle-techlink.chatgpt.site">公开网页</a> · <a href="https://github.com/miracle-techlink/egodata_process/releases/tag/technical-report-2026-11-12">GitHub发布</a><p class="note">浏览不需联网。代码与实验记录为本地快照，历史环境路径用于溯源；缓存模型复跑不依赖原服务器。</p></footer><script>'''
page+='const videos='+json.dumps(videos,ensure_ascii=False)+';const images='+json.dumps(images,ensure_ascii=False)+';'
page+='''const player=document.getElementById('player'),choose=document.getElementById('choose');videos.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v.description+' — '+v.name;choose.appendChild(o);});function select(i){const v=videos[i];const compatible=videos.find(x=>x.name===v.name.replace('.mp4','_browser.mp4'));player.src=(v.codec_name!=='h264'&&compatible)?compatible.path:v.path;document.getElementById('description').textContent=v.description;document.getElementById('metadata').textContent=v.nb_frames+' 帧 / '+v.duration_seconds.toFixed(2)+' 秒 / '+v.width+'×'+v.height+' / '+v.codec_name+(v.codec_name!=='h264'?'；在线播放自动使用同内容H.264兼容版，下载保留原文件':'');const a=document.getElementById('download');a.href=v.path;a.download=v.name;}choose.onchange=()=>select(Number(choose.value));const first=videos.findIndex(v=>v.name==='full-pipeline.mp4');choose.value=first;select(first);document.getElementById('slow').onclick=()=>player.playbackRate=.5;document.getElementById('normal').onclick=()=>player.playbackRate=1;function gallery(){const q=document.getElementById('filter').value.toLowerCase(),list=images.filter(x=>(x.name+' '+x.group).toLowerCase().includes(q)),g=document.getElementById('gallery');g.replaceChildren();document.getElementById('imageCount').textContent=list.length+' 个图像／图表文件';list.forEach(x=>{const a=document.createElement('a');a.className='tile';a.href=x.path;if(x.kind!=='.pdf'){const im=document.createElement('img');im.src=x.path;im.loading='lazy';im.alt=x.name;a.appendChild(im);}const caption=document.createElement('div');caption.textContent=x.name+' · '+x.group;a.appendChild(caption);g.appendChild(a);});}document.getElementById('filter').oninput=gallery;gallery();</script></body></html>'''
(OUT/'00_素材导航.html').write_text(page)
# Provenance of original copies, plus explicit generated entries.
old={r['path']:r for r in json.loads((WORK/'copy_records.json').read_text())}
files=[]
for p in sorted(OUT.rglob('*')):
 if not p.is_file() or p.name in {'文件清单.json','文件清单.csv','SHA256SUMS.txt'}:continue
 rel=str(p.relative_to(OUT));row=old.get(rel,{'path':rel,'source':'本次生成或补充','category':p.relative_to(OUT).parts[0],'usage':'本地完整包；按来源使用'})
 row.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest());files.append(row)
summary={'files':len(files),'video_files':len(videos),'image_and_chart_files':len(images),'total_bytes':sum(x['bytes'] for x in files),'guide_pages':22,'public_report_pages':20,'internal_report_pages':21}
manifest={'version':'2026-11-13','scope':'已整理的报告与Demo全套素材、详细解读、代码与证据；非完整原始训练数据集','summary':summary,'files':files}
(OUT/'文件清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
with (OUT/'文件清单.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=['path','category','bytes','sha256','source','usage']);w.writeheader();w.writerows(files)
(OUT/'SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in files))
(WORK/'bundle_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,ensure_ascii=False))
