import test from 'node:test';
import assert from 'node:assert/strict';
import {applyEdit,validate,invalidateDependents,csvText} from '../src/annotation/web_assets/review-core.mjs';
const common={start_frame:60,end_frame:395,evidence_frames:[60,100,380],review_status:'pending',label_source:'assistant_visual_review'};
const relation=(relation_id,predicate,start_frame,end_frame,evidence_frames,source='left_hand',target='obj')=>({...common,relation_id,predicate,start_frame,end_frame,evidence_frames,source,target});
const base={schema_version:'pose_relation_annotation_v1',sequence_id:'synthetic_sample',frame_count:446,
 annotation_mode:'offline_noncausal',allowed_as_model_input:false,training_allowed:false,
 source_bindings:{sequence:{sha256:'fixture_sequence'},scene_manifest:{sha256:'fixture_manifest'}},
 entities:[{entity_id:'left_hand',kind:'hand',side:'left',display_name:'左手'},{entity_id:'right_hand',kind:'hand',side:'right',display_name:'右手'},{entity_id:'obj',kind:'object',display_name:'物体'},{entity_id:'table',kind:'context',display_name:'桌面'}],
 relations:[relation('approach','approaching',40,60,[40]),relation('control','grasp_control',60,395,[60,100,380]),relation('lift','co_motion',95,160,[100]),relation('return','co_motion',300,365,[320]),relation('support_after','supported_by',380,446,[400,445],'obj','table')],
 transitions:[],cooperations:[{...common,cooperation_id:'single_hand_pick_return',mode:'single_hand_manipulation',participants:[{entity_id:'left_hand',role:'manipulator'},{entity_id:'obj',role:'manipulated_object'}],relation_ids:['control','lift','return']}],task_semantics:[],hand_review_frames:[363,382,383,396,398]};
const v={predicates:{approaching:1,contact:1,grasp_control:1,co_motion:1,supported_by:1,stabilizes:1,acts_on:1},events:{contact_onset:1,grasp_onset:1,motion_onset:1,support_gained:1,support_lost:1,release:1},modes:{single_hand_manipulation:1,bimanual_joint:1,bimanual_complementary:1,handover:1,tool_mediated:1}};
const now='2026-10-23T04:00:00.000Z';
test('original pending annotations remain valid without a human reviewer',()=>assert.ok(validate(base,base,v)));
test('approval requires a human name and preserves original provenance',()=>{
 assert.throws(()=>applyEdit(base,'relations','control',{},'accepted','',now),/审核人/);
 const d=applyEdit(base,'relations','control',{},'accepted','审核员甲',now);
 assert.equal(d.relations[1].label_source,'human');assert.equal(d.relations[1].original_label_source,'assistant_visual_review');assert.equal(d.relations[1].reviewed_at,now);assert.equal(base.relations[1].review_status,'pending');validate(d,base,v);
});
test('last included frame 445 round trips as exclusive end 446',()=>{
 const d=applyEdit(base,'relations','support_after',{end_frame:445+1,evidence_frames:[445]},'accepted','甲',now);validate(d,base,v);assert.equal(d.relations.at(-1).end_frame,446);
 const bad=structuredClone(d);bad.relations.at(-1).evidence_frames=[446];assert.throws(()=>validate(bad,base,v),/证据帧/);
});
test('invalid intervals, missing evidence and unknown endpoints are rejected',()=>{
 for(const patch of [{start_frame:-1},{start_frame:60.5},{end_frame:447},{end_frame:60},{evidence_frames:[]},{source:'unknown'}]){
 const d=applyEdit(base,'relations','control',patch,'pending','',now);assert.throws(()=>validate(d,base,v));}
});
test('cooperation needs accepted dependent relations',()=>{
 const d=applyEdit(base,'cooperations','single_hand_pick_return',{},'accepted','甲',now);assert.throws(()=>validate(d,base,v),/先通过/);
});
test('editing an approved relation invalidates its approved cooperation',()=>{
 let d=structuredClone(base);
 for(const id of ['control','lift','return'])d=applyEdit(d,'relations',id,{},'accepted','甲',now);
 d=applyEdit(d,'cooperations','single_hand_pick_return',{},'accepted','甲',now);validate(d,base,v);
 d=applyEdit(d,'relations','control',{notes:'修正遮挡说明'},'pending','甲',now);
 assert.equal(d.relations[1].review_status,'pending');assert.equal(d.cooperations[0].review_status,'pending');assert.equal(d.cooperations[0].reviewer,undefined);validate(d,base,v);
});
test('bimanual identity and support direction are checked',()=>{
 const d=structuredClone(base);d.cooperations[0].mode='bimanual_joint';assert.throws(()=>validate(d,base,v),/两只手/);
 const b=structuredClone(base);b.relations[0].predicate='supported_by';assert.throws(()=>validate(b,base,v),/支撑关系/);
});
test('imports cannot change source identity or training gates',()=>{
 for(const patch of [{sequence_id:'another'},{training_allowed:true},{allowed_as_model_input:true},{hand_review_frames:[]}])assert.throws(()=>validate({...structuredClone(base),...patch},base,v));
});
test('handedness decisions require a real flagged frame and review note',()=>{
 const d=structuredClone(base);d.handedness_reviews=[{frame_id:363,review_status:'accepted',reviewer:'甲',reviewed_at:now,notes:'实际左手，待修正输出槽位'}];validate(d,base,v);
 d.handedness_reviews[0].frame_id=1;assert.throws(()=>validate(d,base,v));
});
test('Chinese CSV escapes quotes and spreadsheet formulas',()=>{
 const d=structuredClone(base);d.relations[0].notes='=HYPERLINK("bad")';const s=csvText(d,v);assert.ok(s.startsWith('\ufeff"标注层"'));assert.ok(s.includes('"\'=HYPERLINK(""bad"")"'));
});
