import {groups,idKeys,statusNames,roleNames,clone,records,validate,applyEdit,csvText,mergeConfirmedReview} from './review-core.mjs';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
let bundle,base,data,key,group='relations',selected='control',frame=100,overlay=true,loop=false,dirty=false,editTimer,toastTimer,editorEvidence=[],imageFrame=-1,currentImage=null,requestId=0,revision=0,videoCallback=null,animationCallback=null;
const video=$('#video'),canvas=$('#scene'),ctx=canvas.getContext('2d'),colors=['#ffd071','#7ccfff','#d7aaff','#ff9aa9'];
const name=id=>data.entities.find(e=>e.entity_id===id)?.display_name||id||'';
const label=r=>bundle.predicates[r.predicate]||bundle.events[r.event_type]||bundle.modes[r.mode]||r.description||'手别复核';
const handReviewed=f=>(data.handedness_reviews||[]).some(r=>r.frame_id===f&&r.review_status==='accepted');
const current=()=>group==='hands'?(data.handedness_reviews||[]).find(r=>String(r.frame_id)===String(selected)):(data[group]||[]).find(r=>r[idKeys[group]]===selected);
function toast(message,error=false){clearTimeout(toastTimer);$('#toast').textContent=message;$('#toast').classList.toggle('error',error);$('#toast').hidden=false;toastTimer=setTimeout(()=>$('#toast').hidden=true,4200);}
function saveLocal(){
  try{revision++;localStorage.setItem(key,JSON.stringify({data,reviewer:$('#reviewer').value,revision,selection:{group,selected},frame}));$('#save-state').textContent='✓ 已保存到当前浏览器 · '+new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});return true;}
  catch(e){$('#save-state').textContent='浏览器保存失败，请立即导出备份';toast('浏览器存储不可用，当前修改仍在页面中，请导出备份。',true);return false;}
}
function rows(){if(group==='hands')return base.hand_review_frames.map(f=>({frame_id:f,...(data.handedness_reviews||[]).find(r=>r.frame_id===f),review_status:(data.handedness_reviews||[]).find(r=>r.frame_id===f)?.review_status||'pending'}));return data[group];}
function recordId(r){return group==='hands'?String(r.frame_id):r[idKeys[group]];}
function renderQueue(){
 const all=records(data),done=all.filter(x=>x.item.review_status==='accepted'||x.item.review_status==='rejected').length;
 $('#review-count').textContent=done+' / '+all.length;$('#progress-fill').style.width=(100*done/all.length)+'%';
 $('#progress-caption').textContent=`已完成 ${done} 条 · 待审 ${all.filter(x=>x.item.review_status==='pending').length} 条`;
 for(const b of $$('.tabs button')){b.setAttribute('aria-selected',String(b.dataset.group===group));b.querySelector('b').textContent=b.dataset.group==='hands'?base.hand_review_frames.length:data[b.dataset.group].length;}
 const visible=rows().filter(r=>$('#status-filter').value==='all'||r.review_status===$('#status-filter').value);
 $('#record-list').innerHTML=visible.map(r=>`<button class="record-card ${recordId(r)===String(selected)?'selected':''}" data-id="${esc(recordId(r))}"><span class="label">${esc(group==='hands'?'第 '+r.frame_id+' 帧 · 手别异常':label(r))}</span><span class="subtitle">${esc(group==='hands'?'模型输出右手，需核对解剖左右':r.source?name(r.source)+' → '+name(r.target):group==='cooperations'?r.participants.map(p=>name(p.entity_id)).join(' · '):'完整任务描述')}</span><span class="card-bottom"><span>${group==='hands'?'单帧复核':r.start_frame+'–'+(r.end_frame-1)+' 帧'}</span><span class="status ${r.review_status}">${statusNames[r.review_status]}</span></span></button>`).join('')||'<p class="empty">此筛选下没有标注</p>';
 $('#add-record').hidden=group==='hands';$('#add-record').textContent='＋ 新增'+(groups[group]||'标注');
}
function options(list,selectedValue){return Object.entries(list).map(([v,t])=>`<option value="${esc(v)}" ${v===selectedValue?'selected':''}>${esc(t)}</option>`).join('');}
const entityOptions=v=>options(Object.fromEntries(data.entities.map(e=>[e.entity_id,e.display_name+(e.kind==='object'?' · '+e.entity_id:'')])),v);
function renderEvidence(){
 $('#evidence-list').innerHTML=editorEvidence.map(f=>`<span class="evidence-tag"><button type="button" data-evidence="${f}" title="查看第 ${f} 帧">${f}</button><button type="button" data-remove-evidence="${f}" aria-label="移除证据帧 ${f}">×</button></span>`).join('')||'<span class="helper">请添加至少一个区间内的证据帧</span>';
}
function participantRow(p){return `<div class="role-row"><select aria-label="参与实体" class="participant-entity">${entityOptions(p.entity_id)}</select><select aria-label="协作角色" class="participant-role">${options({...roleNames,...(!(p.role in roleNames)?{[p.role]:p.role}:{})},p.role)}</select><button class="remove-participant" title="移除此参与者" type="button">×</button></div>`;}
function actionButtons(){return `<div id="form-error" class="error-message" hidden role="alert"></div><div class="review-actions"><button type="button" class="primary" data-review="accepted">✓ 确认通过</button><div class="secondary-actions"><button type="button" data-review="needs_revision">需修改</button><button type="button" class="reject" data-review="rejected">否决</button></div><button type="button" class="save-draft" data-review="pending">保存草稿</button><button type="button" class="text-button" id="next-pending">下一条待审核 →</button><button type="button" class="text-button" id="cancel-edits">撤销未保存的修改</button></div>`;}
function renderEditor(){
 dirty=false;clearTimeout(editTimer);const r=current();
 $('#editor-group').textContent=groups[group]||'手别异常';
 const status=r?.review_status||'pending';$('#editor-status').className='status '+status;$('#editor-status').textContent=statusNames[status];
 if(group==='hands'){
   $('#selection-heading').textContent='手别异常 · 第 '+selected+' 帧';
   $('#editor-body').innerHTML=`<div class="banner">这帧模型输出为右手，请结合前后帧核对。审核意见会单独保存，原姿态数据保持可追溯。</div><p class="editor-intro">“通过”表示本条手别复核意见已确认，不代表已修复姿态。</p><label class="form-field"><span>复核意见</span><textarea id="notes" rows="5" placeholder="例如：实际为左手，模型左右手槽位需要纠正。">${esc(r?.notes||'')}</textarea></label><div class="footer-actions"><button type="button" id="hand-before">前 3 帧</button><button type="button" id="hand-at">异常帧</button><button type="button" id="hand-after">后 3 帧</button></div>${actionButtons()}<p class="audit-stamp">${r?.reviewer?'审核人：'+esc(r.reviewer):'尚无人工审核记录'}</p>`;return;
 }
 if(!r){$('#editor-body').innerHTML='<p class="empty">从左侧选择一条标注，或新增标注</p>';return;}
 $('#selection-heading').textContent=label(r);editorEvidence=[...r.evidence_frames];
 let special='';
 if(group==='relations'||group==='transitions')special=`<div class="two-fields"><label class="form-field"><span>主体</span><select id="source">${entityOptions(r.source)}</select></label><label class="form-field"><span>客体</span><select id="target">${entityOptions(r.target)}</select></label></div><label class="form-field"><span>${group==='relations'?'关系类型':'转换事件'}</span><select id="label-type">${options(group==='relations'?bundle.predicates:bundle.events,r.predicate||r.event_type)}</select></label>`;
 if(group==='cooperations')special=`<label class="form-field"><span>协作方式</span><select id="label-type">${options(bundle.modes,r.mode)}</select></label><div class="field-block"><span class="field-label">参与实体与角色</span><div id="participants">${r.participants.map(participantRow).join('')}</div><button id="add-participant" class="text-button" type="button">＋ 添加参与者</button></div><div class="field-block"><span class="field-label">依据的持续关系</span><div id="relation-refs">${data.relations.map(e=>`<label class="check-row"><input type="checkbox" value="${esc(e.relation_id)}" ${r.relation_ids.includes(e.relation_id)?'checked':''}><span>${esc(name(e.source)+' · '+label(e)+' · '+name(e.target))}<br><span class="muted">${e.start_frame}–${e.end_frame-1} 帧 · ${statusNames[e.review_status]}</span></span></label>`).join('')}</div><p class="helper">先通过所依据的关系，再确认协作方式。</p></div>`;
 if(group==='task_semantics')special=`<label class="form-field"><span>任务描述</span><textarea id="description" rows="3">${esc(r.description)}</textarea></label>`;
 $('#editor-body').innerHTML=`<p class="editor-intro">${group==='transitions'?'缩小转换发生的范围；精确到单帧时起止帧填相同值。':'核对实体、关系和起止帧，再确认审核结论。'}</p>${special}<div class="two-fields"><label class="form-field"><span>开始帧</span><input id="start-frame" type="number" min="0" max="445" value="${r.start_frame}"><button class="frame-set" id="set-start" type="button">使用当前帧</button></label><label class="form-field"><span>结束帧（包含）</span><input id="end-frame" type="number" min="0" max="445" value="${r.end_frame-1}"><button class="frame-set" id="set-end" type="button">使用当前帧</button></label></div><p class="helper">起止帧均包含在区间内，最后一帧为 445。</p><div class="field-block"><span class="field-label">证据帧</span><div class="evidence-list" id="evidence-list"></div><div class="evidence-input"><input id="evidence-number" type="number" min="0" max="445" aria-label="添加证据帧" placeholder="帧号"><button id="add-evidence" type="button">添加</button><button id="add-current" type="button">＋ 当前帧</button></div></div><label class="form-field"><span>审核备注</span><textarea id="notes" rows="3" placeholder="记录判断依据、遮挡情况或修改原因">${esc(r.notes||'')}</textarea></label>${actionButtons()}<p class="audit-stamp">${r.reviewer?'审核人：'+esc(r.reviewer)+' · '+esc(new Date(r.reviewed_at).toLocaleString('zh-CN')):'尚无人工审核记录'}<br>有效修改自动保存为待审草稿；通过后再次修改需重新审核。</p>`;
 renderEvidence();
}
function readPatch(){
 if(group==='hands')return {notes:$('#notes').value};
 const number=id=>$(id).value.trim()===''?NaN:Number($(id).value);
 const patch={start_frame:number('#start-frame'),end_frame:number('#end-frame')+1,evidence_frames:[...editorEvidence].sort((a,b)=>a-b),notes:$('#notes').value};
 if(group==='relations'||group==='transitions'){patch.source=$('#source').value;patch.target=$('#target').value;patch[group==='relations'?'predicate':'event_type']=$('#label-type').value;}
 if(group==='cooperations'){patch.mode=$('#label-type').value;patch.participants=$$('#participants .role-row').map(el=>({entity_id:el.querySelector('.participant-entity').value,role:el.querySelector('.participant-role').value}));patch.relation_ids=$$('#relation-refs input:checked').map(el=>el.value);}
 if(group==='task_semantics')patch.description=$('#description').value;
 return patch;
}
function commit(status='pending',silent=false){
 clearTimeout(editTimer);if(group!=='hands'&&!current())return true;
 try{
   const patch=readPatch(),reviewer=$('#reviewer').value;let next;
   if(group==='hands'){
     if(status!=='pending'&&(!reviewer.trim()||!patch.notes.trim()))throw new Error('请填写审核人和手别复核意见。');
     next=clone(data);next.handedness_reviews ||= [];let r=next.handedness_reviews.find(r=>r.frame_id===Number(selected));
     if(!r){r={frame_id:Number(selected)};next.handedness_reviews.push(r);}Object.assign(r,patch,{review_status:status});
     if(status!=='pending')Object.assign(r,{reviewer:reviewer.trim(),reviewed_at:new Date().toISOString()});else{delete r.reviewer;delete r.reviewed_at;}
   }else next=applyEdit(data,group,selected,patch,status,reviewer);
   validate(next,base,bundle);const invalidated=next.cooperations.filter((r,i)=>r.review_status==='pending'&&data.cooperations[i]?.review_status==='accepted').length;
   data=next;dirty=false;saveLocal();$('#form-error').hidden=true;refresh(false);
   if(!silent){renderEditor();toast(status==='pending'?'草稿已保存':statusNames[status]+'，审核记录已保存');}
   if(invalidated)toast('关联协作已回到待审核，请重新核对。');return true;
 }catch(e){$('#form-error').textContent=e.message;$('#form-error').hidden=false;$('#save-state').textContent='当前修改尚未保存 · 请修正提示';if(!silent)toast(e.message,true);return false;}
}
function markDirty(){dirty=true;clearTimeout(editTimer);$('#save-state').textContent='正在保存修改…';editTimer=setTimeout(()=>commit('pending',true),700);}
function flush(){return !dirty||commit('pending',true);}
function selectRecord(nextGroup,id){if(!flush())return;group=nextGroup;selected=String(id);renderQueue();renderEditor();renderTimeline();const r=current();goFrame(group==='hands'?Number(id):r?.start_frame||0);saveLocal();}
function renderTimeline(){
 $('#timeline').innerHTML=Object.keys(groups).map(g=>`<div class="timeline-row"><span>${groups[g]}</span><div class="timeline-lanes">${data[g].map((r,i)=>`<button class="timeline-bar ${r.review_status} ${g===group&&r[idKeys[g]]===selected?'selected':''}" style="left:${100*r.start_frame/446}%;width:${100*(r.end_frame-r.start_frame)/446}%;top:${(i%3)*7+2}px" data-group="${g}" data-id="${esc(r[idKeys[g]])}" title="${esc(label(r))}：${r.start_frame}–${r.end_frame-1} 帧" aria-label="${esc(label(r))} ${r.start_frame}至${r.end_frame-1}帧"></button>`).join('')}<span class="timeline-cursor" style="left:${100*frame/445}%"></span></div></div>`).join('');
}
function renderEntities(){
 $('#entity-list').innerHTML=data.entities.map(e=>`<div class="entity-chip"><i style="background:${entityColor(e.entity_id)}"></i>${esc(e.display_name)}${e.kind==='object'?'<small>'+esc(e.entity_id)+'</small>':''}</div>`).join('');
}
function entityColor(id){return id==='left_hand'?'#62e8ad':id==='right_hand'?'#ffad81':colors[data.entities.filter(e=>e.kind==='object').findIndex(e=>e.entity_id===id)]||'#b5c3bc';}
function activeRelations(){return data.relations.filter(r=>r.start_frame<=frame&&frame<r.end_frame&&r.review_status!=='rejected');}
function updateFrameUI(){
 $('#frame-badge').textContent='第 '+frame+' / 445 帧';$('#frame-number').value=frame;$('#frame-slider').value=frame;
 $('#hand-warning').hidden=!base.hand_review_frames.includes(frame);
 $('#hand-warning').textContent=handReviewed(frame)?'本帧手别已复核 · 原估计保留':'本帧手别待复核';
 for(const el of $$('.timeline-cursor'))el.style.left=(100*frame/445)+'%';
 $('#live-relations').innerHTML=activeRelations().map(r=>`<div class="relation-chip ${r.review_status}">${esc(name(r.source))}<span class="arrow">→</span><strong>${esc(label(r))}</strong><span class="arrow">→</span>${esc(name(r.target))} <small>· ${statusNames[r.review_status]}</small></div>`).join('')||'<span class="small muted">当前帧没有已记录的关系，仍可补充标注。</span>';
}
function refresh(editor=true){renderQueue();renderTimeline();renderEntities();updateFrameUI();draw();if(editor)renderEditor();else{const s=current()?.review_status||'pending';$('#editor-status').textContent=statusNames[s];$('#editor-status').className='status '+s;}}
function tag(text,x,y,color){
 ctx.font='13px "Noto Sans CJK SC","Microsoft YaHei",sans-serif';const w=ctx.measureText(text).width+14;x=Math.max(5,Math.min(635-w,x));y=Math.max(22,Math.min(454,y));
 ctx.fillStyle='#13261fe8';ctx.fillRect(x,y,w,25);ctx.strokeStyle=color;ctx.lineWidth=1;ctx.strokeRect(x,y,w,25);ctx.fillStyle=color;ctx.fillText(text,x+7,y+17);
}
function arrow(a,b,status){
 const dx=b[0]-a[0],dy=b[1]-a[1],n=Math.hypot(dx,dy);if(n<2)return;const ux=dx/n,uy=dy/n;
 ctx.strokeStyle=ctx.fillStyle=status==='accepted'?'#62e8ad':'#ffcf75';ctx.lineWidth=2;ctx.setLineDash(status==='accepted'?[]:[5,4]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);
 ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-ux*9-uy*4,b[1]-uy*9+ux*4);ctx.lineTo(b[0]-ux*9+uy*4,b[1]-uy*9-ux*4);ctx.fill();
}
function draw(source){
 const bg=source||(!video.paused&&video.readyState>=2?video:imageFrame===frame?currentImage:null);if(!bg)return;
 ctx.drawImage(bg,0,0,640,480);canvas.dataset.frame=String(frame);if(!overlay)return;
 const p=bundle.poses[frame],anchors={};
 for(const [id,pts] of Object.entries(p.objects))if(pts){
   const color=entityColor(id);ctx.strokeStyle=color;ctx.lineWidth=1.5;
   for(let a=0;a<8;a++)for(const bit of [1,2,4]){const b=a^bit;if(a<b){ctx.beginPath();ctx.moveTo(...pts[a]);ctx.lineTo(...pts[b]);ctx.stroke();}}
   ['#ff6b70','#66e9a5','#73b6ff'].forEach((c,i)=>{ctx.strokeStyle=c;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...pts[8]);ctx.lineTo(...pts[9+i]);ctx.stroke();});
   anchors[id]=pts[8];const entity=data.entities.find(e=>e.entity_id===id);const at=entity.label_anchor||[.03,.15];const x=at[0]*640,y=at[1]*480;
   ctx.strokeStyle=color;ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(...pts[8]);ctx.lineTo(x,y+12);ctx.stroke();tag(name(id)+' '+id,x,y,color);
 }
 for(const [id,pts]of Object.entries(p.hands))if(pts){ctx.strokeStyle=ctx.fillStyle=entityColor(id);ctx.lineWidth=2;
   for(const [a,b] of bundle.hand_connections){ctx.beginPath();ctx.moveTo(...pts[a]);ctx.lineTo(...pts[b]);ctx.stroke();}
   for(const pt of pts){ctx.beginPath();ctx.arc(...pt,2,0,Math.PI*2);ctx.fill();}
   anchors[id]=[0,1].map(i=>[0,5,9,13,17].reduce((s,n)=>s+pts[n][i],0)/5);tag(name(id)+(base.hand_review_frames.includes(frame)?(handReviewed(frame)?' · 已复核':' · 手别待审'):''),pts[0][0]+12,pts[0][1]+20,entityColor(id));
 }
 for(const e of data.entities)if(e.kind==='context'&&e.image_anchor)anchors[e.entity_id]=[e.image_anchor[0]*640,e.image_anchor[1]*480];
 for(const r of activeRelations())if(anchors[r.source]&&anchors[r.target])arrow(anchors[r.source],anchors[r.target],r.review_status);
}
function pause(){video.pause();if(videoCallback!==null&&video.cancelVideoFrameCallback)video.cancelVideoFrameCallback(videoCallback);if(animationCallback!==null)cancelAnimationFrame(animationCallback);videoCallback=animationCallback=null;$('#play').textContent='▶';$('#play').setAttribute('aria-label','播放');}
async function goFrame(n){
 n=Number(n);if(!Number.isInteger(n)||n<0||n>=446){toast('请输入 0 到 445 之间的整数帧号。',true);return;}pause();frame=n;updateFrameUI();
 const request=++requestId;$('#frame-loading').hidden=false;
 video.currentTime=(n+.1)/30;
 const img=new Image();img.onload=()=>{if(request!==requestId)return;currentImage=img;imageFrame=n;$('#frame-loading').hidden=true;draw(img);};img.onerror=()=>{if(request===requestId){$('#frame-loading').textContent='画面载入失败，请刷新或重试';toast('第 '+n+' 帧载入失败。',true);}};
 img.src='frames/'+String(n).padStart(6,'0')+'.jpg';
}
function playingFrame(t,metadata){
 if(video.paused)return;const n=Math.min(445,Math.max(0,Math.floor((metadata?.mediaTime??video.currentTime)*30+1e-4)));
 const r=current();if(loop&&r?.end_frame&&n>=r.end_frame){video.currentTime=(r.start_frame+.1)/30;}
 else{frame=n;updateFrameUI();draw(video);}scheduleVideo();
}
function scheduleVideo(){if(video.requestVideoFrameCallback)videoCallback=video.requestVideoFrameCallback(playingFrame);else animationCallback=requestAnimationFrame(playingFrame);}
async function togglePlay(){if(!video.paused){pause();goFrame(frame);return;}if(!flush())return;
 try{const r=current();if(loop&&r?.end_frame&&(frame<r.start_frame||frame>=r.end_frame))frame=r.start_frame;else if(frame>=445)frame=loop&&r?.start_frame||0;video.currentTime=(frame+.1)/30;requestId++;$('#frame-loading').hidden=true;await video.play();$('#play').textContent='Ⅱ';$('#play').setAttribute('aria-label','暂停');scheduleVideo();}catch(e){toast('播放失败，可使用逐帧查看继续审核。',true);}
}
function modal(title,html){$('#dialog-title').textContent=title;$('#dialog-body').innerHTML=html;$('#dialog').showModal();}
function exportData(){
 if(!flush())return null;try{validate(data,base,bundle);}catch(e){toast(e.message,true);return null;}
 const out=clone(data);out.web_review={version:bundle.version,base_annotation_sha256:bundle.annotation_sha256,exported_at:new Date().toISOString(),revision};return out;
}
function download(filename,content,type){const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);}
function exportModal(){if(!flush())return;modal('保存与导出审核结果',`<p>已保存的草稿保留在当前浏览器。导出文件可用于跨设备备份或交给数据处理流程。</p><div class="dialog-buttons"><button id="download-json" class="primary">下载审核结果（JSON，完整可回读）</button><button id="download-csv">下载中文审核表（CSV）</button><button id="import-json">导入已有审核结果</button></div><p class="small">源姿态与原始候选保持不变。审核通过不会自动放行正式训练。</p>`);
 $('#download-json').onclick=()=>{const out=exportData();if(out){download('手物交互_审核结果_'+new Date().toISOString().slice(0,10)+'.json',JSON.stringify(out,null,2),'application/json');toast('审核结果已导出');}};
 $('#download-csv').onclick=()=>{const out=exportData();if(out)download('手物交互_中文审核表.csv',csvText(out,bundle),'text/csv;charset=utf-8');};
 $('#import-json').onclick=()=>$('#import-file').click();
}
function addRecord(){
 if(!flush())return;const id='manual_'+group+'_'+Date.now(), common={start_frame:frame,end_frame:Math.min(frame+31,446),evidence_frames:[frame],review_status:'pending',label_source:'human',geometry_status:'not_checked',notes:''};let r;
 if(group==='relations')r={...common,relation_id:id,source:'left_hand',target:'G04_1',predicate:'contact'};
 if(group==='transitions')r={...common,event_id:id,source:'left_hand',target:'G04_1',event_type:'contact_onset',end_frame:frame+1};
 if(group==='task_semantics')r={...common,task_id:id,description:'待补充任务描述'};
 if(group==='cooperations'){
  const edge=data.relations.find(r=>r.start_frame<=frame&&frame<r.end_frame&&r.source==='left_hand'&&r.review_status!=='rejected')||data.relations.find(r=>r.source==='left_hand'&&r.review_status!=='rejected');
  if(!edge){toast('请先添加一条手与物体的持续关系。',true);return;}
  r={...common,cooperation_id:id,mode:'single_hand_manipulation',start_frame:edge.start_frame,end_frame:edge.end_frame,evidence_frames:[edge.start_frame],participants:[{entity_id:edge.source,role:'manipulator'},{entity_id:edge.target,role:'manipulated_object'}],relation_ids:[edge.relation_id]};
 }
 data[group].push(r);selected=id;saveLocal();refresh();toast('已新增待审核标注，请补充内容。');
}
function nextPending(){if(!flush())return;const list=rows(),i=list.findIndex(r=>recordId(r)===selected);const next=[...list.slice(i+1),...list.slice(0,i)].find(r=>r.review_status==='pending'||r.review_status==='needs_revision');if(next)selectRecord(group,recordId(next));else toast('这一层没有其他待审核项。');}
function editNames(){if(!flush())return;modal('修改中文显示名称',`<p>只修改显示名称，物体编号与姿态绑定保持不变。</p>${data.entities.map(e=>`<div class="name-row"><label for="name-${esc(e.entity_id)}">${esc(e.kind==='hand'?name(e.entity_id):e.kind==='context'?'环境参照':e.entity_id)}</label><input id="name-${esc(e.entity_id)}" data-entity="${esc(e.entity_id)}" maxlength="30" value="${esc(e.display_name)}"></div>`).join('')}<div class="footer-actions"><button id="save-names" class="primary">保存名称</button></div>`);
 $('#save-names').onclick=()=>{const next=clone(data);for(const el of $$('#dialog-body [data-entity]')){if(!el.value.trim()){toast('名称不能为空。',true);return;}next.entities.find(e=>e.entity_id===el.dataset.entity).display_name=el.value.trim();}data=next;saveLocal();refresh();$('#dialog').close();toast('中文名称已更新');};
}
function bind(){
 $('#record-list').onclick=e=>{const b=e.target.closest('[data-id]');if(b)selectRecord(group,b.dataset.id);};
 for(const b of $$('.tabs button'))b.onclick=()=>{if(!flush())return;$('#status-filter').value='all';const g=b.dataset.group;selectRecord(g,g==='hands'?base.hand_review_frames[0]:data[g][0]?.[idKeys[g]]||'');};
 $('#status-filter').onchange=renderQueue;
 $('#timeline').onclick=e=>{const b=e.target.closest('[data-id]');if(b)selectRecord(b.dataset.group,b.dataset.id);};
 $('#prev-frame').onclick=()=>goFrame(Math.max(0,frame-1));$('#next-frame').onclick=()=>goFrame(Math.min(445,frame+1));$('#play').onclick=togglePlay;
 $('#frame-number').onchange=e=>goFrame(Number(e.target.value));$('#frame-slider').oninput=e=>goFrame(Number(e.target.value));
 $('#speed').onchange=e=>video.playbackRate=Number(e.target.value);$('#loop').onclick=()=>{loop=!loop;$('#loop').setAttribute('aria-pressed',String(loop));};
 $('#show-overlay').onclick=()=>{overlay=true;$('#show-overlay').classList.add('active');$('#show-raw').classList.remove('active');draw();};$('#show-raw').onclick=()=>{overlay=false;$('#show-raw').classList.add('active');$('#show-overlay').classList.remove('active');draw();};
 video.onended=()=>{const r=current();if(loop&&r?.end_frame){pause();frame=r.start_frame;togglePlay();}else goFrame(445);};
 $('#editor-body').oninput=e=>{if(e.target.id!=='evidence-number')markDirty();};
 $('#editor-body').onchange=e=>{if(e.target.tagName==='SELECT'||e.target.type==='checkbox')markDirty();};
 $('#editor-body').onclick=e=>{
 const b=e.target.closest('button');if(!b)return;
 if(b.dataset.review){commit(b.dataset.review);return;}
 if(b.dataset.evidence){goFrame(Number(b.dataset.evidence));return;}
 if(b.hasAttribute('data-remove-evidence')){editorEvidence=editorEvidence.filter(f=>f!==Number(b.dataset.removeEvidence));renderEvidence();markDirty();return;}
 if(b.id==='set-start'||b.id==='set-end'){$(b.id==='set-start'?'#start-frame':'#end-frame').value=frame;markDirty();}
 if(b.id==='add-current'||b.id==='add-evidence'){
   const raw=$('#evidence-number').value,n=b.id==='add-current'?frame:raw.trim()===''?NaN:Number(raw);
   if(!Number.isInteger(n)||n<Number($('#start-frame').value)||n>Number($('#end-frame').value)){toast('证据帧必须是当前区间内的整数帧号。',true);return;}
   editorEvidence=[...new Set([...editorEvidence,n])].sort((a,b)=>a-b);renderEvidence();markDirty();
 }
 if(b.id==='add-participant'){$('#participants').insertAdjacentHTML('beforeend',participantRow({entity_id:'right_hand',role:'stabilizer'}));markDirty();}
 if(b.classList.contains('remove-participant')){b.closest('.role-row').remove();markDirty();}
 if(b.id==='next-pending')nextPending();
 if(b.id==='cancel-edits'){renderEditor();$('#save-state').textContent='已恢复最近保存的草稿';toast('已撤销未保存的修改');}
 if(b.id==='hand-before')goFrame(Math.max(0,Number(selected)-3));if(b.id==='hand-after')goFrame(Math.min(445,Number(selected)+3));if(b.id==='hand-at')goFrame(Number(selected));
 };
 $('#reviewer').onchange=saveLocal;$('#add-record').onclick=addRecord;$('#export').onclick=exportModal;$('#close-dialog').onclick=()=>$('#dialog').close();$('#edit-names').onclick=editNames;
 $('#help').onclick=()=>modal('如何审核这一段标注',`<ol><li>填写右上角的审核人，选择左侧一条候选标注。</li><li>播放视频观察动作，使用左右方向键逐帧核对；可切换原始画面。</li><li>调整主体、客体、关系和起止帧，添加证据帧。有效修改会自动保存为待审草稿。</li><li>点击“确认通过”“需修改”或“否决”。已通过的关系修改后会回到待审核。</li><li>最后导出完整 JSON 备份，或下载中文审核表。跨浏览器、跨设备请导入备份。</li></ol><p>左右手按被摄者的解剖左右。结束帧包含在区间内；未知源时戳时，只按原生帧审核。手别异常意见不会直接覆盖原始骨架。</p>`);
 $('#source-details').onclick=()=>modal('数据来源与标注说明',`<p>HO-Cap：Jikai Wang、Qifan Zhang、Yu-Wei Chao、Bowen Wen、Xiaohu Guo、Yu Xiang。图像数据采用 <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>。</p><p>来源：<a href="https://irvlutd.github.io/HOCap/" target="_blank" rel="noreferrer">官方项目</a>；下载镜像版本 2b24836d5e51ad39e56ed4db3fc0c166e755332e。选取单采集446帧，保持原生帧序，另编码播放视频；新增模型姿态估计及交互候选，不表示官方背书。</p><p>手部由 MediaPipe、物体由 FoundationPose 估计。候选关系经助手看图提出，需你人工确认。手部相对深度与物体米制位置不直接混用。</p><p>草稿保存在当前浏览器，不自动同步到服务器。导出结果仍保持训练准入关闭。</p>`);
 $('#import-file').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  try{const next=JSON.parse(await file.text());if(next.web_review?.base_annotation_sha256!==bundle.annotation_sha256)throw new Error('文件对应的初始标注版本不一致。');validate(next,base,bundle);
   modal('导入审核结果',`<p>文件：${esc(file.name)}</p><p>导入将替换当前浏览器草稿。建议先备份当前结果。</p><div class="dialog-buttons"><button id="backup-current">先下载当前草稿</button><button id="confirm-import" class="primary">确认导入这份结果</button></div>`);
   $('#backup-current').onclick=()=>{const out=exportData();if(out)download('导入前_审核备份.json',JSON.stringify(out,null,2),'application/json');};
   $('#confirm-import').onclick=()=>{data=next;group='relations';selected=data.relations[0]?.relation_id||'';saveLocal();refresh();$('#dialog').close();toast('审核结果已导入');};
  }catch(err){toast('导入失败：'+err.message,true);}finally{e.target.value='';}
 };
 document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT','BUTTON'].includes(document.activeElement.tagName)||$('#dialog').open)return;
  if(e.key==='ArrowLeft'){e.preventDefault();goFrame(Math.max(0,frame-1));}if(e.key==='ArrowRight'){e.preventDefault();goFrame(Math.min(445,frame+1));}if(e.code==='Space'){e.preventDefault();togglePlay();}
 });
 window.addEventListener('beforeunload',e=>{if(dirty&&!commit('pending',true)){e.preventDefault();e.returnValue='';}});
}
async function init(){
 try{
  const response=await fetch('data.json');if(!response.ok)throw new Error('标注数据载入失败');bundle=await response.json();base=clone(bundle.annotations);data=clone(base);key='hoi-review:'+bundle.annotation_sha256;
  try{let saved=JSON.parse(localStorage.getItem(key)||'null');let migrated=false;
   if(!saved&&bundle.approval){const previous=JSON.parse(localStorage.getItem('hoi-review:'+bundle.approval.previous_annotation_sha256)||'null');if(previous){saved={...previous,data:mergeConfirmedReview(previous.data,base)};validate(saved.data,base,bundle);localStorage.setItem(key,JSON.stringify(saved));migrated=true;}}
   if(saved){validate(saved.data,base,bundle);data=saved.data;$('#reviewer').value=saved.reviewer||'';revision=saved.revision||0;group=saved.selection?.group||group;selected=saved.selection?.selected||selected;frame=Number.isInteger(saved.frame)?saved.frame:100;$('#save-state').textContent=migrated?'✓ 已同步本次通过结论，保留原草稿内容':'✓ 已恢复当前浏览器的审核草稿';}
   else $('#save-state').textContent=base.batch_review?'✓ 本批已按你的确认全部通过':'草稿保存在当前浏览器 · 可导出备份';}
  catch(e){toast('未载入旧草稿：'+e.message,true);$('#save-state').textContent='使用原始候选 · 请检查或导入备份';}
  bind();refresh();goFrame(frame);
 }catch(e){$('#save-state').textContent='载入失败';$('#record-list').innerHTML='<p class="error-message">'+esc(e.message)+'，请刷新重试。</p>';toast(e.message,true);}
}
init();
