export const groups = {relations:'持续关系',transitions:'转换事件',cooperations:'协作方式',task_semantics:'任务语义'};
export const idKeys = {relations:'relation_id',transitions:'event_id',cooperations:'cooperation_id',task_semantics:'task_id'};
export const statusNames = {pending:'待审核',accepted:'已通过',needs_revision:'需修改',rejected:'已否决'};
export const roleNames = {manipulator:'操作手',manipulated_object:'被操作物体',stabilizer:'固定手',workpiece:'工件',tool:'工具',giver:'交出方',receiver:'接收方',support:'支撑物',patient:'被作用物体'};
export const clone = value => structuredClone(value);
export function records(data) {return Object.keys(groups).flatMap(group=>(data[group]||[]).map(item=>({group,item,id:item[idKeys[group]]})));}
export function validate(data,base,vocab) {
  const fail=message=>{throw new Error(message);};
  if (data.schema_version!==base.schema_version || data.sequence_id!==base.sequence_id || data.frame_count!==base.frame_count ||
      data.source_bindings?.sequence?.sha256!==base.source_bindings.sequence.sha256 ||
      data.source_bindings?.scene_manifest?.sha256!==base.source_bindings.scene_manifest.sha256) fail('文件不属于当前采集或来源版本不一致。');
  if(data.training_allowed!==false || data.allowed_as_model_input!==false || data.annotation_mode!=='offline_noncausal') fail('审核文件不能改变训练准入或作为模型输入。');
  const entities=new Map((data.entities||[]).map(e=>[e.entity_id,e]));
  if(entities.size!==base.entities.length || data.entities.length!==base.entities.length)fail('实体数量或标识发生变化。');
  for(const e of base.entities){const v=entities.get(e.entity_id);if(!v || v.kind!==e.kind || v.side!==e.side || typeof v.display_name!=='string' || !v.display_name.trim())fail('实体标识、左右手归属不可变，名称不能为空。');}
  const seen=new Set();
  for(const {group,item:r,id} of records(data)){
    if(typeof id!=='string'||!id.trim()||seen.has(id))fail('标注编号必须唯一。');seen.add(id);
    if(!Number.isInteger(r.start_frame)||!Number.isInteger(r.end_frame)||r.start_frame<0||r.end_frame>data.frame_count||r.start_frame>=r.end_frame)fail('帧区间无效，请填写范围内的整数起止帧。');
    if(!Array.isArray(r.evidence_frames)||!r.evidence_frames.length||r.evidence_frames.some(f=>!Number.isInteger(f)||f<r.start_frame||f>=r.end_frame))fail('至少添加一个证据帧，且必须位于当前区间内。');
    if(!(r.review_status in statusNames)||!['human','assistant_visual_review','kinematic_candidate','vlm'].includes(r.label_source))fail('审核状态或标注来源无效。');
    if(r.review_status==='accepted' && (r.label_source!=='human'||!r.reviewer?.trim()||!r.reviewed_at||!Number.isFinite(Date.parse(r.reviewed_at))))fail('通过标注需要审核人及有效的审核时间。');
    if(group==='relations'||group==='transitions'){
      const s=entities.get(r.source),t=entities.get(r.target);
      if(!s||!t||r.source===r.target)fail('请选择不同且存在的主体和客体。');
      if(group==='relations'){
        if(!(r.predicate in vocab.predicates))fail('未知的持续关系类型。');
        const ok=r.predicate==='supported_by'?s.kind==='object'&&['object','context'].includes(t.kind):r.predicate==='acts_on'?['hand','object'].includes(s.kind)&&t.kind==='object':s.kind==='hand'&&t.kind==='object';
        if(!ok)fail('主体与客体不适用于此关系；支撑关系从物体指向支撑物。');
      }else if(!(r.event_type in vocab.events))fail('未知的转换事件。');
    }
    if(group==='cooperations'){
      if(!(r.mode in vocab.modes))fail('未知的协作方式。');
      const ids=(r.participants||[]).map(p=>p.entity_id);
      if(ids.length<2||new Set(ids).size!==ids.length||ids.some(id=>!entities.has(id))||r.participants.some(p=>!p.role?.trim()))fail('协作需要至少两个不同实体和明确角色。');
      const hands=ids.filter(id=>entities.get(id).kind==='hand').length;
      if(['bimanual_joint','bimanual_complementary','handover'].includes(r.mode)&&hands!==2)fail('双手协作或交接必须包含左右两只手。');
      if(r.mode==='single_hand_manipulation'&&hands!==1)fail('单手操作需要且只能有一只手。');
      if(!Array.isArray(r.relation_ids)||!r.relation_ids.length)fail('请选择协作所依据的持续关系。');
      for(const ref of r.relation_ids){const edge=data.relations.find(e=>e.relation_id===ref);
        if(!edge||!ids.includes(edge.source)||!ids.includes(edge.target)||edge.end_frame<=r.start_frame||edge.start_frame>=r.end_frame)fail('协作与所选关系的实体或时间区间不匹配。');
        if(r.review_status==='accepted'&&edge.review_status!=='accepted')fail('请先通过协作所依据的全部持续关系。');
      }
    }
    if(group==='task_semantics'&&(typeof r.description!=='string'||!r.description.trim()))fail('请填写任务描述。');
  }
  if(JSON.stringify(data.hand_review_frames)!==JSON.stringify(base.hand_review_frames))fail('原始手别异常帧列表不可改写。');
  const handIds=new Set();
  for(const h of data.handedness_reviews||[]){
    if(!base.hand_review_frames.includes(h.frame_id)||handIds.has(h.frame_id)||!(h.review_status in statusNames))fail('手别审核帧或状态无效。');handIds.add(h.frame_id);
    if(h.review_status!=='pending'&&(!h.reviewer?.trim()||!h.reviewed_at||!Number.isFinite(Date.parse(h.reviewed_at))||!h.notes?.trim()))fail('手别审核需要审核人、时间与中文意见。');
  }
  return true;
}
export function invalidateDependents(data,relationId){
  let count=0;for(const r of data.cooperations)if(r.relation_ids.includes(relationId)&&r.review_status==='accepted'){
    r.review_status='pending';r.review_invalidated_reason='依赖的持续关系已修改，请重新审核';delete r.reviewer;delete r.reviewed_at;count++;
  }return count;
}
export function mergeConfirmedReview(draft,approved){
  const next=clone(draft),time=Date.parse(approved.batch_review.reviewed_at);
  const auditKeys=['review_status','reviewer','reviewed_at','review_method','label_source','original_label_source'];
  for(const {group,item:r,id} of records(approved)){
    const local=next[group]?.find(v=>v[idKeys[group]]===id);
    if(!local || Date.parse(local.modified_at||'1970-01-01')>time)continue;
    for(const k of auditKeys)if(k in r)local[k]=r[k];
  }
  for(const r of next.cooperations){
    if(r.review_status==='accepted'&&r.relation_ids.some(id=>next.relations.find(v=>v.relation_id===id)?.review_status!=='accepted')){
      r.review_status='pending';delete r.reviewer;delete r.reviewed_at;
    }
  }
  next.handedness_reviews ||= [];
  for(const r of approved.handedness_reviews||[]){
    const index=next.handedness_reviews.findIndex(h=>h.frame_id===r.frame_id),old=next.handedness_reviews[index];
    if(old&&Date.parse(old.reviewed_at||old.modified_at||'1970-01-01')>time)continue;
    const merged={...r,...(old?.notes?{notes:old.notes}:{})};
    if(index<0)next.handedness_reviews.push(clone(merged));else next.handedness_reviews[index]=merged;
  }
  for(const e of approved.entities){const local=next.entities.find(v=>v.entity_id===e.entity_id);if(local&&e.review_status==='accepted')for(const k of auditKeys)if(k in e)local[k]=e[k];}
  next.batch_review=clone(approved.batch_review);
  return next;
}
export function applyEdit(data,group,id,patch,status,reviewer,now=new Date().toISOString()){
  const next=clone(data),r=next[group].find(v=>v[idKeys[group]]===id);
  if(!r)throw new Error('未找到当前标注。');
  const before=JSON.stringify(r);Object.assign(r,clone(patch));
  const changed=before!==JSON.stringify(r);
  if(status!=='pending'&&!reviewer.trim())throw new Error('请先填写右上角的审核人。');
  if(status!=='pending'){
    r.original_label_source ||= r.label_source;
    r.label_source='human';r.reviewer=reviewer.trim();r.reviewed_at=now;r.review_status=status;
  }else if(changed||r.review_status!=='pending'){
    r.review_status='pending';delete r.reviewer;delete r.reviewed_at;
  }
  if(changed){r.modified_at=now;r.modified_by=reviewer.trim()||'未署名草稿';}
  if(group==='relations'&&(changed||r.review_status!=='accepted'))invalidateDependents(next,id);
  return next;
}
export function csvText(data,vocab){
 const name=id=>data.entities.find(e=>e.entity_id===id)?.display_name||id;
 const rows=[['标注层','编号','主体','客体','标签','起始帧','结束帧（包含）','证据帧','审核状态','审核人','审核时间','备注']];
 for(const {group,item:r,id} of records(data))rows.push([groups[group],id,name(r.source||''),name(r.target||''),vocab.predicates[r.predicate]||vocab.events[r.event_type]||vocab.modes[r.mode]||r.description,r.start_frame,r.end_frame-1,r.evidence_frames.join('、'),statusNames[r.review_status],r.reviewer||'',r.reviewed_at||'',r.notes||'']);
 for(const r of data.handedness_reviews||[])rows.push(['手别异常',r.frame_id,'','','手别复核',r.frame_id,r.frame_id,r.frame_id,statusNames[r.review_status],r.reviewer||'',r.reviewed_at||'',r.notes||'']);
 const quote=v=>'"'+String(v??'').replace(/^[=+@\-]/,"'$&").replaceAll('"','""')+'"';
 return '\ufeff'+rows.map(row=>row.map(quote).join(',')).join('\r\n');
}
