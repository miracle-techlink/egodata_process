"""Release the audited interaction subset; never certify the entire 300-clip pool."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tarfile

from src.mamba.dataset_v5 import EgoDemoGraphV5Dataset, read, sha


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def require(condition, reason='Training delivery evidence is inconsistent'):
    if not condition:
        raise ValueError(reason)


def check_sequence_bindings(root, index):
    for row in index['sequences']:
        for content_key, path_key in [
            ('review_sha256','review_path'),('observations_sha256','observations_path'),
            ('observation_metadata_sha256','observation_metadata_path'),('seeds_sha256','seeds_path'),
            ('tracking_review_sha256','tracking_review_path'),('sequence_sha256','sequence_path')]:
            require(sha(root/row[path_key]) == row[content_key], 'Audited sequence artifact changed: '+row[path_key])


def admit(root):
    root = Path(root)
    if (root/'training_release.json').exists():
        raise FileExistsError('Release already frozen')
    index = read(root/'training_index.json'); digest = sha(root/'training_index.json')
    evidence = ['review_seed_audit.json', 'training_index_audit.json',
        'delivery_integrity_checks_v5.json', 'model_checks/real_graph_mamba_smoke.json',
        'model_checks/real_graph_target_coverage.json', 'audit_sources/model_report_crosscheck_v2.json']
    audits = [read(root/name) for name in evidence]
    review, audit, receipt, smoke, coverage, crosscheck = audits
    require(review['failed'] == audit['failed'] == receipt['failed'] == 0)
    require(audit['training_index_sha256'] == receipt['training_index_sha256'] == smoke['training_index_sha256'] == digest)
    require(audit['review_seed_audit_sha256'] == sha(root/evidence[0]))
    require(receipt['training_index_audit_sha256'] == sha(root/evidence[1]))
    require(crosscheck['failed'] == 0 and crosscheck['training_index_sha256'] == digest)
    require(coverage['training_index_sha256'] == digest)
    require(crosscheck['independent_dataset_audit_sha256'] == sha(root/evidence[1]))
    require(crosscheck['coverage_report_sha256'] == sha(root/evidence[4]))
    require(crosscheck['model_smoke_report_sha256'] == sha(root/evidence[3]))
    require(crosscheck['all_split_type_positive_negative_and_unknown_counts_match'] is True)
    require(crosscheck['model_bound_code_and_example_sha_verified'] is True)
    require(receipt['test_failures'] == receipt['test_errors'] == 0)
    require(smoke['passed'] and smoke['optimizer_steps'] == 0)
    require(smoke['future_labels_metadata_invariant'] and smoke['object_slot_permutation_equivariant'])
    for check in smoke['precision_checks'].values():
        require(check['valid_count'] > 0 and check['positive_count'] > 0 and check['negative_count'] > 0)
        require(check['finite_gradients'] and check['backbone_gradient_l2'] > 0)
        require(check['unknown_logit_gradient_count'] == 0 and check['weights_unchanged'])
    require(sha(root/'model_checks/real_graph_batch.npz') == smoke['example_sha256'])
    for path, digest_source in smoke['source_sha256'].items():
        require(sha(path) == digest_source)
    require(index['counts'] == audit['counts'])
    check_sequence_bindings(root,index)
    loaded = {}
    for split in ('train','val','test'):
        ds = EgoDemoGraphV5Dataset(root, split, diagnostic=True)
        require(len(ds) == index['splits'][split]['windows'] > 0)
        require(ds[0]['graph_label_valid'].any())
        loaded[split] = len(ds)
    require(not index['full_annotation_complete'] and not index['whole_object_6d_certified'])
    release = {'schema_version':'v5_scoped_interaction_training_release',
        'created_at':datetime.now(timezone.utc).isoformat(), 'training_index_sha256':digest,
        'training_scope':index['training_scope'], 'interaction_training_allowed':True,
        'full_joint_training_allowed':False,'full_annotation_complete':False,
        'whole_object_6d_certified':False,'human_ground_truth_certified':False,
        'blind_evaluation':False,'original_pool_retained':300,
        'subset_sequence_count':len(index['sequences']),'subset_windows':loaded,
        'positive_types_absent_from_train':receipt['positive_types_absent_from_train'],
        'evidence_sha256':{name:sha(root/name) for name in evidence},
        'scope_zh':'允许以已审核二维实例观测和官方手姿态进行助手银标交互预测的小规模训练。只覆盖本索引5段1238窗；没有认证300段完整标注或整物体6D，不含训练结果。',
        'evaluation_limit_zh':'训练仅12种正例类型；验证/测试另9类没有训练正例，须单列未见类型。窗口高度重叠、采集少且标注已被查看，不用于声称无偏或盲测性能。'}
    save(root/'training_release.json',release)
    for split in loaded:
        require(EgoDemoGraphV5Dataset(root,split)[0]['graph_label_valid'].any())
    return release


def package(root, repo):
    root,repo = Path(root),Path(repo)
    require(read(root/'training_release.json')['interaction_training_allowed'] is True)
    check_sequence_bindings(root,read(root/'training_index.json'))
    for split in ('train','val','test'):
        require(EgoDemoGraphV5Dataset(root,split)[0]['graph_label_valid'].any())
    target = root.parent/(root.name+'_interaction_subset.tar.gz')
    if target.exists(): raise FileExistsError('Frozen package exists')
    snapshot=root/'code_snapshot'
    paths=['src/__init__.py','src/mamba/__init__.py','src/dataset/__init__.py',
        'src/mamba/egodemo_v2.py','src/mamba/dataset_v5.py','src/mamba/graph_v5.py',
        'src/mamba/graph_v5_smoke.py','src/dataset/export_training_v5.py',
        'src/dataset/deliver_training_v5.py','src/dataset/audit_training_v5.py',
        'src/dataset/report_training_v5.py',
        'src/annotation/track_visual_entities_v5.py','src/annotation/render_visual_tracking_v5.py',
        'src/annotation/build_training_review_web_v5.py',
        'tests/test_dataset_v5.py','tests/test_graph_v5.py','tests/test_visual_tracking_v5.py',
        'tests/test_audit_training_v5.py','docs/TRAINING_V5_HANDOFF.md',
        '.agent/memory/03_data_pipeline.md']
    for relative in paths:
        source=repo/relative
        if not source.exists():
            if relative.endswith('__init__.py'):continue
            raise FileNotFoundError(source)
        dst=snapshot/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
    # Preserve every file needed for loading, plus galleries and the frozen audit
    # evidence. Native per-frame crops/videos remain on the private source disk.
    included=[]
    for folder,dirs,files in os.walk(root,followlinks=True):
        dirs[:]=[d for d in dirs if d not in {'crops','__pycache__','pilots','logs'}]
        for name in sorted(files):
            path=Path(folder)/name;rel=path.relative_to(root).as_posix()
            if name in {'PACKAGE_SCOPE.json','delivery_manifest.json'}:continue
            if name.endswith(('.pyc','.log')):continue
            included.append({'path':rel,'bytes':path.stat().st_size,'sha256':sha(path)})
    included.sort(key=lambda x:x['path'])
    save(root/'PACKAGE_SCOPE.json',{'schema_version':'v5_portable_numeric_subset',
        'numeric_training_self_contained':True,'full_300_dataset':False,
        'complete_native_source_pixels_included':False,'selected_native_review_images_included':True,
        'source_evidence_reaudit_requires_server':True,
        'excluded':['raw source videos','dense_evidence/**/crops/*','pilots','logs'],
        'files':included,'usage_zh':'解压后把code_snapshot加入PYTHONPATH即可加载数字训练子集；tracking联系表和中文页面包含在包内。旧V4继承证据以普通文件打包，不保留绝对软链接。完整原像素重新审计须使用私有服务器原视频、裁剪及V2/V3/V4。'})
    save(root/'delivery_manifest.json',{'schema_version':'v5_private_delivery_manifest',
        'training_index_sha256':sha(root/'training_index.json'),
        'training_release_sha256':sha(root/'training_release.json'),
        'package_scope_sha256':sha(root/'PACKAGE_SCOPE.json'),
        'file_count':len(included),'file_bytes':sum(x['bytes'] for x in included),
        'full_annotation_complete':False,'interaction_subset_released':True})
    with tarfile.open(target,'w:gz',dereference=True) as archive:
        for row in included:
            archive.add(root/row['path'],arcname=root.name+'/'+row['path'],recursive=False)
        for name in ['PACKAGE_SCOPE.json','delivery_manifest.json']:
            archive.add(root/name,arcname=root.name+'/'+name,recursive=False)
    digest=sha(target)
    (target.parent/(target.name+'.sha256')).write_text(digest+'  '+target.name+'\n')
    return {'path':str(target),'bytes':target.stat().st_size,'sha256':digest}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--repo',default='.');p.add_argument('--phase',choices=['admit','package'],required=True)
    a=p.parse_args();result=admit(a.root) if a.phase=='admit' else package(a.root,a.repo)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
