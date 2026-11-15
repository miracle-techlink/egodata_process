"""Private review delivery: measurable outputs and explicit unresolved fields."""
from datetime import datetime,timezone
import argparse,csv,json
from pathlib import Path
from src.dataset.complete_egodemo import save,digest
from src.annotation.refine_semantics import REFINED_OPERATIONS as OPERATIONS
from src.dataset.report_egodemo_refine import report


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 summary=report(r);audit=json.loads((r/'engineering_audit.json').read_text());assert audit['engineering_audit_complete']
 rows=json.loads((r/'sequence_completion_matrix.json').read_text())['records'];evidence={x['sequence_id']:x for x in audit['records']};queue=[]
 for row in rows:
  uid=row['sequence_id'];proposal=json.loads((r/'joint_annotations'/uid/'proposal.json').read_text());issues=[]
  if row['references']==0:issues.append('没有通过双目一致性检查的三维参考')
  if row['pixel_calibration_status']=='inconsistent_on_sampled_frames':issues.append('原标定与图像匹配不一致，已禁用三维派生交互')
  elif row['pixel_calibration_status'] not in ['consistent_on_sampled_frames','image_refined_with_heldout_checks']:issues.append('像素级标定需复核')
  if not row['hand_near_point_object_frames']:issues.append('尚未获得手附近的物体表面轨迹')
  if any(e['rigidity']=='rigid' for e in proposal['entities']) and not row['hand_near_pose_object_frames']:issues.append('刚体候选缺少手附近的有效6D')
  if any(e['physical_target_ambiguous'] for e in proposal['entities']):issues.append('部分名称/掩码指向多个物体或透明表面')
  issues.extend(['确认实际操作对象、工具/工件角色及跨遮挡稳定身份','逐帧确认接触、抓持、支撑、工具作用和协作边界'])
  ev=evidence[uid]
  queue.append({'sequence_id':uid,'task_zh':row['task_zh'],'priority':0 if row['references']==0 else (1 if len(issues)>3 else 2),
                'issues_zh':issues,'frame_count':row['native_frames'],
                'surface_observed_frames':ev['unique_video_frames_with_8_surface_points'],
                'pose_observed_frames':ev['unique_video_frames_with_pose'],
                'entity_candidates':row['entity_hypotheses'],'review_status':'pending','training_allowed':False})
 queue.sort(key=lambda x:(x['priority'],x['task_zh'],x['sequence_id']))
 save(r/'逐段待审清单.json',{'records':queue,'count':len(queue),'human_approved':0})
 with (r/'逐段待审清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['采集ID','任务','优先级','原生帧数','至少8表面点的唯一帧数','有6D候选的唯一帧数','实体候选','待审事项'])
  for x in queue:w.writerow([x['sequence_id'],x['task_zh'],x['priority'],x['frame_count'],x['surface_observed_frames'],x['pose_observed_frames'],x['entity_candidates'],'；'.join(x['issues_zh'])])
 with (r/'操作候选分布.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['操作代码','中文名称','源语义候选数','已人工确认数'])
  for name,count in sorted(summary['source_operation_candidates'].items(),key=lambda x:-x[1]):w.writerow([name,OPERATIONS.get(name,name),count,0])
 source=Path(__file__).resolve().parents[2]
 paths=[source/p for p in ['src/pose6d/refine_backends.py','src/pose6d/refine_seeds.py','src/pose6d/refine_tracks.py',
  'src/pose6d/rigid_fit.py','src/pose6d/nonrigid_tracks.py','src/pose6d/stereo_reference.py','src/pose6d/observation_exclusions.py',
  'src/pose6d/audit_stereo_pixels.py','src/pose6d/refine_stereo_calibration.py','src/pose6d/recover_manual_seeds.py','src/annotation/refine_associations.py','src/annotation/quality_bindings.py','src/annotation/refine_semantics.py',
  'src/annotation/egodemo_review.py','src/annotation/egodemo_review.html','src/dataset/report_egodemo_refine.py',
  'src/dataset/audit_egodemo_refine.py','src/dataset/deliver_egodemo_refine.py']]
 save(r/'code_manifest.json',{'records':[{'path':str(p.relative_to(source)),'sha256':digest(p)} for p in paths]})
 models=r/'models';vendor=models/'Fast-FoundationStereo-vendor'
 save(r/'model_provenance.json',{'models':[
  {'name':'C-FastFoundationStereo','repository':'nvidia/c-fast-foundationstereo','revision':'53a84e28a368f37e699776250215fcb450d6a109','sha256':digest(models/'cfast_model_best_bp2_serialize.pth'),'use':'all reference frames; 768 pixels, manual small-target recovery 1280 pixels'},
  {'name':'SAM2.1 base plus','repository':'facebook/sam2.1-hiera-base-plus','revision':'b7320756a13354e7530a63935656d35b2f91a290','sha256':digest(models/'sam2-hf'/'model.safetensors'),'use':'three diagnostic video-mask pilots only, not the full native tracking batch'}],
  'foundation_stereo_source_files':[{'path':str(p.relative_to(vendor)),'sha256':digest(p)} for p in sorted(vendor.rglob('*')) if p.is_file() and (p.suffix in ['.py','.yaml','.txt'])],
  'tracking':'CPU bidirectional Lucas-Kanade, calibrated triangulation and rigid surface fitting',
  'calibration_recovery':'Image-derived relative extrinsic, original metric baseline preserved; two fit frames and four held-out frames; original calibration retained',
  'independent_accuracy':None})
 help={'版本':'egodemo_refine_300_20261023','服务器目录':str(r),'独立采集':300,'原始任务名':75,
  '入口':'http://127.0.0.1:18788','本机SSH转发':'ssh -p 30314 -N -L 18788:127.0.0.1:18788 root@120.209.70.195',
  '审核顺序':['按逐段待审清单优先核对缺几何和标定异常','确认对象名称、刚体/部件/非刚体、工具/工件角色','关联属于同一真实对象的轨迹，跨遮挡不要仅凭同名合并','逐帧检查持续关系、转换边界和协作；必要时圈选新表面','保存待审草稿并导出JSON；保存不等于审核通过'],
  '文件':{'双手':'sequences/<UUID>/hands.npz（原官方左右21点）','几何':'native_tracks/<UUID>/tracklet_*.npz（表面点，保留失败）','对齐物体姿态':'joint_annotations/<UUID>/objects.npz','交互与ID提案':'joint_annotations/<UUID>/proposal.json','逐段状态':'第二轮补全进度.csv','缺项和优先级':'逐段待审清单.csv','工程审计':'engineering_audit.json','像素级标定抽查':'pixel_calibration_audit_summary.json','标定补救':'calibration_recovery_summary.json（原基线尺度＋独立图像验证的外参估计）'},
  '坐标合同':{'hands_world':'(T,2,21,3), 米；左右为被摄者左右','hands_camera':'(T,2,21,3), OpenCV原左相机系','poses':'(N,T,4,4)，物体参考表面到原左相机，失败NaN','pose_valid':'(N,T)，几何估计有效，不代表人工确认','objects':'稳定ID提案仅限可见连续追踪及有重合证据的关联；跨遮挡待核对','object_frame':'种子帧相机轴及可见表面原点，非CAD规范坐标','points':'非刚体保存带point_id的原生帧3D表面轨迹，不强制赋整物体6D','interval':'所有交互区间[start_frame,end_frame)，转换边界另存不确定范围'},
  '训练限制':['本轮为离线双向估计，不能直接作为因果预测输入','旧编码器obj_seq=(B,W,N,13)只覆盖刚体状态；非刚体需要表面点分支及有效掩码',f"{summary['observed_source_operation_types']}种是原文语义命中的操作候选，不代表已审核物理关系类型数",'未标注/不可见不等于无交互；不能转成none负例','全量人工通过数为0，训练门禁仍关闭'],
  '状态':{'自动处理结束':summary['processing_complete'],'工程一致性审计':audit['engineering_audit_complete'],'完整联合标注验收':False,'允许训练':False}}
 save(r/'使用说明.json',help)
 outputs=['refinement_report.json','sequence_completion_matrix.json','第二轮补全进度.csv','逐段待审清单.json','逐段待审清单.csv','操作候选分布.csv','使用说明.json','engineering_audit.json','pixel_calibration_audit_summary.json','code_manifest.json','model_provenance.json','visual_quality_overrides.json','manual_recovery_receipt.json','calibration_recovery_summary.json','semantic_refinement_report.json']
 save(r/'delivery_manifest.json',{'generated_at':datetime.now(timezone.utc).isoformat(),'source_selection_sha256':summary['source_selection_sha256'],
  'files':[{'path':name,'sha256':digest(r/name),'bytes':(r/name).stat().st_size} for name in outputs],
  'full_annotation_complete':False,'training_allowed':False})
 print(json.dumps({'delivered':len(outputs),'sequences':summary['sequences'],'filtered_pose_sequences':summary['sequences_with_filtered_pose_candidates'],'human_approved':0}))

if __name__=='__main__':main()
