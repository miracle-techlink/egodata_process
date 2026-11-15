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
node_names=['左手','右手','双手','物体 1','物体 2','物体 3','物体 4']
contact=next(i for i,t in enumerate(meta['event_types']) if t['code']=='contact')
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
 text(30,77,'RGB 输入  →  手部 / 物体感知  →  132 维历史特征  →  LSTM  →  有向交互关系',24,accent)
 for x in [20,610,1200]:d.rounded_rectangle((x,125,x+580,938),radius=14,fill=panel)
 text(40,141,'01  原始 RGB',28);text(630,141,'02  手部骨架 + 物体 6D',28);text(1220,141,'03  模型关系输出',28)
 for x,frame in [(30,a),(620,b)]:
  pic=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((560,420),Image.Resampling.LANCZOS)
  im.paste(pic,(x,193))
 # Current observations and history validity, not fabricated labels.
 text(40,635,'同一采集 · 同一帧号',24,accent)
 text(40,681,'输入：RGB + 深度 + 相机内参',22)
 text(40,719,'手部：MediaPipe 21 点二维骨架',22)
 text(40,757,'物体：FoundationPose 四目标跟踪',22)
 text(40,795,'监督：此片段没有交互真值标签',22)
 text(40,843,'原始时戳未核验；按 30 fps 回放',20,muted)
 text(40,878,'用于展示各模块的数据流与真实输出',20,muted)
 text(630,635,'物体位置 / 相机坐标系（米）',24,accent)
 for n in range(4):
  p=z['poses'][f,n,:3,3]
  text(630,679+n*40,f'物体 {n+1}   x {p[0]:+.3f}   y {p[1]:+.3f}   z {p[2]:.3f}',22,colors[n])
 hs=['有效' if v else '缺失' for v in z['hand_valid'][f]]
 text(630,851,f'二维手部：左 {hs[0]} / 右 {hs[1]}',20)
 text(630,886,'度量关节三维未提供，输入保留缺失掩码',18,muted)
 if f<31:
  text(1240,270,'历史窗口积累中',34,accent)
  text(1240,335,f'{f+1} / 32 帧 · 暂不输出预测',24)
 else:
  # Diagram displays actual contact scores for hand-to-object pairs only.
  hand_at=[(1265,277),(1265,435)]
  obj_at=[(1650,235+n*87) for n in range(4)]
  for h,(hx,hy) in enumerate(hand_at):
   for n,(ox,oy) in enumerate(obj_at):
    p=float(z['probabilities'][f,h,n+3,contact])
    if z['prediction_mask'][f,h,n+3] and p>=.5:
     d.line((hx+27,hy,ox-29,oy),fill=colors[n],width=2+int(p*3))
     dx,dy=ox-hx,oy-hy;length=(dx*dx+dy*dy)**.5;ux,uy=dx/length,dy/length
     tip=(ox-29,oy);d.polygon([tip,(tip[0]-12*ux-5*uy,tip[1]-12*uy+5*ux),(tip[0]-12*ux+5*uy,tip[1]-12*uy-5*ux)],fill=colors[n])
  for h,(x,y) in enumerate(hand_at):
   d.ellipse((x-27,y-27,x+27,y+27),fill=accent if z['entity_seen'][f,h] else '#526079')
   text(x-27,y+33,['左手','右手'][h],20)
  for n,(x,y) in enumerate(obj_at):
   d.rounded_rectangle((x-28,y-25,x+28,y+25),radius=7,fill=colors[n]);text(x-37,y+28,f'物体 {n+1}',20)
  text(1220,590,'上图：接触关系，p ≥ 0.50；无边 ≠ 真值为负',18,muted)
  scores=[]
  for h in range(3):
   for n in range(3,7):
    if z['prediction_mask'][f,h,n]:
     for k,t in enumerate(meta['event_types']):scores.append((float(z['probabilities'][f,h,n,k]),h,n,t['name_zh']))
  scores=sorted(scores,reverse=True)[:5]
  text(1220,636,'手 → 物：模型概率 Top 5',24,accent)
  for i,(p,h,n,name) in enumerate(scores):
   yy=680+i*39
   text(1220,yy,f'{node_names[h]} → {node_names[n]}  {name}',20)
   text(1680,yy,f'{p:.3f}',20,accent)
   d.rectangle((1220,yy+29,1220+int(500*p),yy+31),fill=accent)
  text(1220,893,'仅模型输出，未与真值计分',20,muted)
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
