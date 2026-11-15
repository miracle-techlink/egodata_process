from pathlib import Path
import json, subprocess, hashlib
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'site/dist'; MEDIA=DEST/'media';MEDIA.mkdir(parents=True,exist_ok=True)
with np.load(ROOT/'demo_arrays.npz') as saved:z={k:saved[k] for k in saved.files}
meta=json.loads((ROOT/'demo_manifest.json').read_text())
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
bold='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
fonts={s:ImageFont.truetype(font,s,layout_engine=ImageFont.Layout.BASIC) for s in [16,18,20,22,24,28,34]}
fonts[38]=ImageFont.truetype(bold,38,layout_engine=ImageFont.Layout.BASIC)
text_cache={}
glyph_cache={}
W,H=1800,1040
bg='#101826';panel='#192437';fg='#edf3ff';muted='#a8b9cf';accent='#69dfc5'
colors=['#ffc857','#55c9ff','#8cdf91','#ee82cf']
raw=cv2.VideoCapture(str(ROOT/'raw_rgb.mp4'))
overlay=cv2.VideoCapture(str(ROOT.parent/'technical_zh_20261112/media/hocap_four_objects.mp4'))
writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
 '-video_size',f'{W}x{H}','-framerate','30','-i','-','-an','-c:v','libx264','-preset','fast',
 '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(MEDIA/'full-pipeline.mp4')],stdin=subprocess.PIPE)
names=meta['object_display_names']
node_names=['左手','右手','双手']+names
contact=next(i for i,t in enumerate(meta['event_types']) if t['code']=='contact')
initial=Image.open(ROOT.parents[1]/'runs/human_dataset_v1_20261022/acquisition/captures_primary/subject_7/20231022_192832/105322251564/105322251564-000000.color.jpg')
thumbs=[]
for box in z['boxes'][0]:
 x0,y0,x1,y1=box;crop=initial.crop((max(0,x0-8),max(0,y0-8),min(640,x1+8),min(480,y1+8)))
 crop.thumbnail((118,98));thumbs.append(crop)
for f in range(meta['frames']):
 ok,a=raw.read();ok2,b=overlay.read();assert ok and ok2
 im=Image.new('RGB',(W,H),bg);d=ImageDraw.Draw(im)
 def text(x,y,s,size=22,fill=fg):
  key=(s,size,fill)
  if key not in text_cache:
   pieces=[]
   for c in s:
    gkey=(c,size,fill)
    if gkey not in glyph_cache:
     ft=fonts[size];ww=max(1,round(ft.getlength(c)));g=Image.new('RGBA',(ww+3,size*2),(0,0,0,0))
     ImageDraw.Draw(g).text((0,0),c,font=ft,fill=fill);glyph_cache[gkey]=(g,ww)
    pieces.append(glyph_cache[gkey])
   tile=Image.new('RGBA',(sum(ww for _,ww in pieces)+3,size*2),(0,0,0,0));xx=0
   for g,ww in pieces:tile.alpha_composite(g,(xx,0));xx+=ww
   text_cache[key]=tile
  im.paste(text_cache[key],(x,y),text_cache[key])
 text(30,18,'手物交互预测 · 完整流程',38)
 text(1110,30,f'HO-Cap 实际回放  |  帧 {f:03d} / 445',24,muted)
 phase=next(p for p in meta['visual_review_phases'] if p['start']<=f<=p['end'])
 text(30,77,'当前交互：'+phase['phase']+'  ·  视觉复核阶段提示'+('（过渡边界待核验）' if phase['uncertain'] else ''),24,accent)
 for x in [20,610,1200]:d.rounded_rectangle((x,125,x+580,938),radius=14,fill=panel)
 text(40,141,'01  画面与物体身份',28);text(630,141,'02  手部 / 6D 与当前交互',28);text(1220,141,'03  模型未来关系候选',28)
 for x,frame in [(30,a),(620,b)]:
  pic=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((560,420),Image.Resampling.LANCZOS)
  im.paste(pic,(x,193))
 # Object identity uses appearance-based reviewed names, not product recognition.
 for n,box in enumerate(z['boxes'][f]):
  if not z['object_valid'][f,n]:continue
  x0,y0,x1,y1=box*.875;x0+=30;x1+=30;y0+=193;y1+=193
  d.rectangle((x0,y0,x1,y1),outline=colors[n],width=2)
  yy=max(195,y0-28);xx=min(max(31,x0),590-len(names[n])*18-10)
  d.rounded_rectangle((xx,yy,xx+len(names[n])*18+10,yy+27),radius=4,fill=bg)
  text(round(xx+4),round(yy),names[n],18,colors[n])
 text(40,635,'四个实体 · 固定身份与颜色',24,accent)
 for n,crop in enumerate(thumbs):
  xx=40+n*138;d.rounded_rectangle((xx,675,xx+124,787),radius=7,outline=colors[n],width=2)
  im.paste(crop,(xx+(124-crop.width)//2,681+(98-crop.height)//2));text(xx,795,names[n],18,colors[n])
 text(40,840,'红盖圆瓶为此片段的主要操作对象',22)
 text(40,878,'名称按外观人工标注；未声称商品识别',18,muted)
 text(630,635,'当前可见交互（视觉复核）',24,accent)
 d.rounded_rectangle((638,688,734,742),radius=8,fill='#284d49');text(660,698,'手',24)
 d.rounded_rectangle((992,688,1160,742),radius=8,outline=colors[0],width=2);text(1011,700,'红盖圆瓶',22,colors[0])
 edgecolor='#f4c979' if phase['uncertain'] else accent
 d.line((738,716,987,716),fill=edgecolor,width=3);d.polygon([(988,716),(974,709),(974,723)],fill=edgecolor)
 d.rectangle((754,686,977,714),fill=panel);text(760,683,phase['relation'],20,edgecolor)
 text(630,758,'阶段提示来自画面复核，不作为逐帧真值计分',18,muted)
 text(630,804,'红盖圆瓶 · 相机坐标（米）',20,colors[0])
 p=z['poses'][f,0,:3,3]
 text(630,839,f'x {p[0]:+.3f}    y {p[1]:+.3f}    z {p[2]:.3f}',22)
 hs=['有效' if v else '缺失' for v in z['hand_valid'][f]]
 text(630,886,f'骨架观测：左 {hs[0]} / 右 {hs[1]}；缺失不补造',18,muted)
 if f<31:
  text(1240,270,'历史窗口积累中',34,accent)
  text(1240,335,f'{f+1} / 32 帧 · 暂不输出预测',24)
 else:
  # Semantic triplets retain actual probabilities; visual review never changes them.
  text(1220,199,'有向关系：主体 → 关系 → 客体',22,accent)
  text(1220,237,'左手 → 物体 · 接触关系概率',18,muted)
  for n in range(4):
   yy=289+n*72;p=float(z['probabilities'][f,0,n+3,contact]);good=z['prediction_mask'][f,0,n+3]
   text(1220,yy,'左手',20)
   d.line((1280,yy+20,1580,yy+20),fill=colors[n] if good and p>=.5 else muted,width=2)
   d.polygon([(1580,yy+20),(1568,yy+14),(1568,yy+26)],fill=colors[n] if good and p>=.5 else muted)
   d.rectangle((1320,yy-3,1528,yy+25),fill=panel)
   text(1324,yy-6,('接触  '+f'{p:.4f}') if good else '未观测',20,colors[n])
   text(1590,yy,names[n],20,colors[n])
  text(1220,585,'箭头表示模型候选，概率不是正确率',18,muted)
  scores=[]
  for h in range(3):
   for n in range(3,7):
    if z['prediction_mask'][f,h,n]:
     for k,t in enumerate(meta['event_types']):scores.append((float(z['probabilities'][f,h,n,k]),h,n,t['name_zh']))
  scores=sorted(scores,reverse=True)[:5]
  text(1220,636,'其他关系候选 · 概率 Top 5',24,accent)
  for i,(p,h,n,name) in enumerate(scores):
   yy=680+i*39
   text(1220,yy,f'{node_names[h]} → {name} → {node_names[n]}',18)
   text(1690,yy,f'{p:.4f}',18,accent)
   d.rectangle((1220,yy+29,1220+int(500*p),yy+31),fill=accent)
  text(1220,893,'跨源高置信输出保留原值；尚未独立验证',18,muted)
 text(30,957,'LSTM / 132D / epoch 43  ·  历史 32 帧  ·  特征时基假设 30 Hz  ·  跨数据集接口演示',22)
 text(30,995,'来源：HO-Cap（CC BY 4.0）  |  此演示不对应报告中的约 0.93 测试结果，未验证实时运行或未来 1 秒时距。',18,muted)
 writer.stdin.write(np.asarray(im).tobytes())
 if f in [31,100,223,379]:im.save(MEDIA/f'pipeline-frame-{f:03d}.jpg',quality=94)
writer.stdin.close();assert writer.wait()==0
raw.release();overlay.release()
# Three chronological, readable crops for the printed report.
sheet=Image.new('RGB',(1800,3*660),'white')
for i,f in enumerate([100,223,379]):
 frame=Image.open(MEDIA/f'pipeline-frame-{f:03d}.jpg')
 sheet.paste(frame.crop((0,0,1800,625)),(0,i*660))
 dd=ImageDraw.Draw(sheet);dd.text((30,i*660+630),f'Frame {f:03d} · 同步原图 / 感知 / 实际模型输出',font=fonts[20],fill='#182539')
sheet.save(MEDIA/'pipeline-sequence.jpg',quality=94)
print('Rendered 446 synchronized frames.')
