"""Build the read-only Chinese V5 delivery browser from frozen artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def bound(root: Path, path: str, expected: str | None = None) -> Path:
    # Frozen evidence may be a deliberate local symlink into V4.
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Non-relative artifact path: {path}")
    result = root / relative
    if not result.is_file() or (expected and sha(result) != expected):
        raise ValueError(f"Missing or changed artifact: {path}")
    return result


def ranges_frames(records):
    return {f for r in records for f in range(r["start_frame"], r["end_frame"])}


def build(root: Path) -> dict:
    root = root.resolve()
    index_path = root / "training_index.json"
    index = read(index_path)
    index_sha = sha(index_path)
    sequences, bindings = [], []
    titles = {
        "2a797a4e-5011-400e-b573-e32d80cc8a54": "纸盒与扫码器",
        "2eb9dcc4-6a9a-4f51-b304-d01e0d9bdd6d": "电水壶与水龙头",
        "99190763-eb6b-43ef-b0ad-8489a9496cbb": "订书机收纳",
        "9be303ef-7491-43be-9915-ddfd51fde7b3": "封口贴与信封",
        "acc3c38d-9591-4eaa-9e6b-d98e385249c8": "游戏手柄与托盘",
    }
    for row in index["sequences"]:
        uid = row["sequence_id"]
        for key in ("review", "observations", "observation_metadata", "seeds", "tracking_review", "sequence"):
            bound(root, row[key + "_path"], row[key + "_sha256"])
        review = read(root / row["review_path"])
        quality = read(root / row["tracking_review_path"])
        manifest = read(bound(root, quality["evidence_manifest_path"], quality["evidence_manifest_sha256"]))
        if quality["observations_sha256"] != row["observations_sha256"]:
            raise ValueError(f"QA observation binding mismatch: {uid}")
        page_by_path = {p["path"]: p for p in manifest["pages"]}
        pages = []
        for path in quality["reviewed_page_paths"]:
            page = page_by_path[path]
            bound(root, path, page["sha256"])
            pages.append(page)
        page_frames = {f for p in pages for f in p["frame_indices"]}
        frames = []
        accepted, rejected = 0, 0
        for frame in manifest["frames"]:
            if frame["frame_index"] not in page_frames:
                continue
            f = frame["frame_index"]
            objects = []
            for obj in frame["objects"]:
                a = [r for r in quality["accepted_ranges"] if r["entity_id"] == obj["entity_id"] and r["start_frame"] <= f < r["end_frame"]]
                b = [r for r in quality["rejected_ranges"] if r["entity_id"] == obj["entity_id"] and r["start_frame"] <= f < r["end_frame"]]
                if len(a) + len(b) != 1:
                    raise ValueError(f"Ambiguous QA: {uid}/{f}/{obj['entity_id']}")
                accepted += bool(a)
                rejected += bool(b)
                objects.append({**obj, "accepted": bool(a), "reason_zh": (a or b)[0]["reason_zh"]})
            frames.append({"frame_index": f, "timestamp_s": frame.get("timestamp_s"), "objects": objects,
                           "page_path": manifest["pages"][frame["page_index"]]["path"]})
        if accepted != quality["accepted_object_observations"] or rejected != quality["rejected_object_observations"]:
            raise ValueError(f"QA count mismatch: {uid}")
        reviewed_count = len(ranges_frames(review["reviewed_ranges"]))
        if reviewed_count != row["continuously_reviewed_frames"] or accepted != row["observed_entity_frame_count"]:
            raise ValueError(f"Index count mismatch: {uid}")
        sequences.append({**row, "title_zh": titles.get(uid, "连续审核采集"), "pages": pages, "frames": frames,
                          "reviewed_ranges": review["reviewed_ranges"], "entities": review["entities"],
                          "states": review["state_segments"], "transitions": review["transition_events"],
                          "unresolved": review.get("unresolved_ranges", []), "legend": manifest["entity_legend"],
                          "quality_accepted": accepted, "quality_rejected": rejected,
                          "quality_reviewed_frames": len(page_frames), "quality_ranges": {
                              "accepted": quality["accepted_ranges"], "rejected": quality["rejected_ranges"]},
                          "full_sequence_continuous_review_complete": review.get("full_sequence_continuous_review_complete", False),
                          "human_label_valid": False})
        bindings.append({"sequence_id": uid, "review_sha256": row["review_sha256"],
                         "tracking_review_sha256": row["tracking_review_sha256"],
                         "evidence_manifest_sha256": quality["evidence_manifest_sha256"]})
    for split in index["splits"].values():
        bound(root, split["path"], split["sha256"])
    release_path = root / index.get("admission_path", "training_release.json")
    release = read(release_path) if release_path.is_file() else None
    if release and release.get("training_index_sha256") != index_sha:
        raise ValueError("Release does not bind current training index")
    if release:
        if release.get("training_scope") != index["training_scope"]:
            raise ValueError("Release training scope mismatch")
        for path, expected in release.get("evidence_sha256", {}).items():
            bound(root, path, expected)
    release_sha = sha(release_path) if release else None
    out = root / "review_web"
    out.mkdir(exist_ok=True)
    integration = out / "integration_source"
    integration.mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    source_files = []
    for relative, title in (("src/mamba/dataset_v5.py", "训练数据加载器"),
                            ("src/mamba/graph_v5.py", "有向图与真实 Mamba 模型"),
                            ("src/mamba/graph_v5_smoke.py", "前向反向验证入口")):
        source = repo / relative
        target = integration / source.name
        shutil.copy2(source, target)
        source_files.append({"title_zh": title, "path": str(target.relative_to(root)),
                             "sha256": sha(target), "source_path": relative})
    links = [{"title_zh": "训练索引", "path": "training_index.json", "sha256": index_sha},
             {"title_zh": "保留的全部 300 段目录", "path": "all_300_acquisitions.json",
              "sha256": sha(bound(root, "all_300_acquisitions.json"))}]
    handoff_path = repo / "docs" / "TRAINING_V5_HANDOFF.md"
    if handoff_path.is_file():
        handoff_copy = out / "training_handoff.md"
        shutil.copy2(handoff_path, handoff_copy)
        links.insert(0, {"title_zh": "中文训练交付说明", "path": str(handoff_copy.relative_to(root)),
                         "sha256": sha(handoff_copy), "source_path": "docs/TRAINING_V5_HANDOFF.md"})
    for split, record in index["splits"].items():
        links.append({"title_zh": {"train": "训练", "val": "验证", "test": "测试"}[split] + "窗口索引",
                      "path": record["path"], "sha256": record["sha256"]})
    for path, title in (("training_release.json", "训练范围放行记录"),
                        ("training_index_audit.json", "训练索引独立审计"),
                        ("review_seed_audit.json", "连续审核与种子溯源审计"),
                        ("model_checks/real_graph_mamba_smoke.json", "真实 Mamba 前向与反向验收"),
                        ("model_checks/real_graph_target_coverage.json", "全窗口实际监督统计"),
                        ("audit_sources/model_report_crosscheck_v2.json", "最终模型报告交叉核验")):
        if (root / path).is_file():
            links.append({"title_zh": title, "path": path, "sha256": sha(root / path)})
    payload = {"schema_version": "assistant_training_v5_readonly_web", "created_at": datetime.now(timezone.utc).isoformat(),
               "training_index_sha256": index_sha, "training_release_sha256": release_sha,
               "training_scope": index["training_scope"],
               "index_created_at": index["created_at"], "window_config": index["window_config"],
               "event_types": index["event_types"], "splits": index["splits"], "counts": index["counts"],
               "known_cells_by_type": index["known_cells_by_type"], "exclusion_counts": index["exclusion_counts"],
               "original_acquisition_count": index["original_acquisition_count"],
               "original_native_frame_count": index["original_native_frame_count"],
               "whole_object_6d_certified": index["whole_object_6d_certified"],
               "full_joint_training_allowed": index["full_joint_training_allowed"],
               "full_annotation_complete": index["full_annotation_complete"], "release": release,
               "sequences": sequences, "source_files": source_files, "links": links,
               "read_only": True, "annotation_writes": 0, "human_approval_created": False}
    (out / "data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    (out / "index.html").write_text(HTML)
    report = {"schema_version": "assistant_training_v5_web_build", "created_at": payload["created_at"],
              "training_index_sha256": index_sha, "training_release_sha256": release_sha,
              "builder_sha256": sha(Path(__file__)), "source_bindings": bindings,
              "files": {n: sha(out / n) for n in ("data.json", "index.html")}, "source_files": source_files,
              "linked_files": links,
              "sequence_count": len(sequences), "window_count": sum(r["window_count"] for r in sequences),
              "reviewed_native_frames": sum(r["continuously_reviewed_frames"] for r in sequences),
              "quality_accepted": sum(r["quality_accepted"] for r in sequences),
              "quality_rejected": sum(r["quality_rejected"] for r in sequences),
              "read_only": True, "annotation_changes": 0, "human_approval_created": False,
              "serve_zh": "仅从数据集根目录绑定 127.0.0.1:18791，访问 /review_web/index.html"}
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练数据交付查看 · EgoDemo V5</title><style>
:root{font-family:system-ui,"Noto Sans CJK SC",sans-serif;color:#172b41;background:#f1f5f9;color-scheme:light}*{box-sizing:border-box}body{margin:0}header{background:#142c43;color:#fff;padding:24px 28px}h1{font-size:24px;margin:0 0 8px}h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:15px 0 8px}p{line-height:1.6;margin:8px 0}button,input,select{font:inherit;padding:8px;border:1px solid #b9c7d6;border-radius:6px;background:white;color:#172b41}button{cursor:pointer}button:hover{background:#ebf5ff}button:disabled{opacity:.4;cursor:default}a{color:#1765a5;text-decoration:underline}code{font-size:11px;overflow-wrap:anywhere}.muted{color:#536c83;font-size:13px}.panel{background:#fff;border:1px solid #dae3ec;border-radius:10px;padding:16px;margin-bottom:15px;min-width:0}.outer{padding:18px 28px}.notice{padding:13px 16px;background:#fff3d9;border:1px solid #ebd5a9;border-radius:8px;line-height:1.6}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.stat{flex:1;min-width:150px;background:#fff;border:1px solid #dae3ec;border-radius:9px;padding:15px}.stat strong{font-size:28px;display:block}.layout{display:grid;grid-template-columns:260px minmax(420px,1fr) 350px;gap:16px}.seq{width:100%;text-align:left;margin:6px 0;padding:12px;line-height:1.6}.seq.active{background:#eaf4ff;border-color:#327ab8}.seq code{display:block}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.tag{display:inline-block;padding:3px 7px;border-radius:4px;font-size:12px;background:#eaf0f6}.yes{background:#def3e9;color:#175b3a}.no{background:#ffebe5;color:#923527}.unknown{background:#fff1d5;color:#78591c}.record{padding:10px 0;border-top:1px solid #e2e8ef;line-height:1.5;font-size:13px}.record code{display:block}.rows{max-height:500px;overflow:auto}.frame-btn{padding:5px 8px;font-size:12px}.frame-btn.active{background:#164e7d;color:#fff}#scene{width:100%;display:block;background:#152332}#scene[hidden],#missing[hidden]{display:none}#missing{padding:55px 15px;text-align:center;background:#edf2f6;border:1px dashed #bac9d5}#frames{margin:10px 0}.hash{overflow-wrap:anywhere;font-size:11px;color:#536c83}.links{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #e0e7ee;padding:9px;text-align:left}summary{cursor:pointer}#error{color:#a52424;padding:20px}#frame{width:100px}#jump{width:100%}#scope-meta{margin-top:10px}.caption{font-size:12px;line-height:1.7;margin:10px 0}.lower{display:grid;grid-template-columns:1fr 1fr;gap:16px}#all-states .record,#all-transitions .record{font-size:12px}.entity{margin:7px 0;font-size:13px}.entity code{display:block}.legend{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.swatch{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px}@media(max-width:1250px){.layout{grid-template-columns:230px minmax(400px,1fr)}.right{grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:760px){.outer{padding:12px}.layout,.lower,.right{display:block}.seq{width:100%}header{padding:18px}.stat{min-width:130px}.sidebar{max-height:350px;overflow:auto}}
.toolbar select{max-width:100%;min-width:0}#coverage{overflow-wrap:anywhere}
</style><header><h1>训练数据交付查看 · EgoDemo V5</h1><div>只读查看 · 原始划分保留 · 助手银标与人工金标准分开记录</div></header>
<div id="error" hidden></div><main class="outer"><div class="notice" id="scope">正在核对交付索引…</div><div id="scope-meta" class="muted"></div><div id="stats" class="cards"></div>
<div class="panel"><h2>固定划分与实际监督</h2><div style="overflow:auto"><table><thead><tr><th>划分</th><th>采集数</th><th>窗口数</th><th>明确正例单元</th><th>明确负例单元</th></tr></thead><tbody id="split-table"></tbody></table></div><p class="muted">单元数量按窗口及有向实体对统计，重叠窗口会重复计入；不是独立动作或交互类型数量。验证、测试均为助手审核子集，不作为盲测金标准。</p></div>
<div class="layout"><aside class="sidebar"><div class="panel"><h2>逐段结果</h2><div id="sequences"></div></div><div class="panel"><h2>接入与核验文件</h2><div id="files"></div><p class="muted">文件链接供查看和下载；此页面没有修改标注或批准训练的功能。</p><details><summary>接口尺寸与来源</summary><div id="shapes"></div></details></div></aside>
<section><div class="panel"><h2 id="title">选择采集</h2><div id="coverage" class="muted"></div><div id="ranges" class="links"></div><div class="toolbar"><button id="prev-page">← 上一证据页</button><select id="page" aria-label="证据页"></select><button id="next-page">下一证据页 →</button></div><div id="frames" class="toolbar"></div><div class="toolbar"><label>原生帧 <input id="frame" type="number" min="0" value="0"></label><button id="go-frame">查看该帧状态</button><span class="muted">← / → 切换证据页</span></div><div id="legend" class="legend"></div><img id="scene" alt="已由助手实际查看的原生帧跟踪联系表" hidden><div id="missing">正在加载真实证据…</div><div id="image-meta" class="caption"></div><div id="seq-links" class="links"></div></div>
<div class="lower"><div class="panel"><h2>连续状态片段</h2><p class="muted">明确正例、明确负例与未知分开显示。区间均为 [起始帧, 结束帧)，不包含结束帧。</p><div id="all-states" class="rows"></div></div><div class="panel"><h2>转换边界与视觉信号</h2><p class="muted">范围事件保留边界不确定性；视觉照明不等同成功扫描或物理接触。</p><div id="all-transitions" class="rows"></div></div></div></section>
<aside class="right"><div class="panel"><h2>当前帧框质量</h2><p id="frame-title" class="muted"></p><div id="quality"></div><h3>当前交互状态</h3><div id="active-states"></div><h3>当前未知范围</h3><div id="unknowns"></div></div><div class="panel"><h2>实体与质量范围</h2><div id="entities"></div><details><summary>全部拒绝范围与原因</summary><div id="rejections"></div></details><details><summary>原始来源 SHA</summary><div id="provenance"></div></details></div></aside></div></main>
<script>
'use strict';const $=x=>document.getElementById(x),SPLIT={train:'训练集',val:'验证集',test:'测试集'},N={left_hand:'左手',right_hand:'右手',both_hands:'双手',object:'物体',relation:'持续关系',operation:'操作',transition:'转换',cooperation:'协作',contact:'接触',grasp_control:'抓持控制',supported_by:'受到支撑',co_motion:'共同运动',stabilizes:'稳定工件',acts_on:'作用于',bimanual_complementary:'双手互补',complementary:'双手互补',support_gained:'建立支撑',support_lost:'失去支撑',contact_onset:'开始接触',grasp_onset:'建立抓持',release:'最终脱离',motion_onset:'开始运动',illumination_onset:'照明开启',illumination_offset:'照明关闭'};
let D,current,frame=0,pageIndex=0,generation=0;const name=x=>N[x]||x||'未指定';
function el(t,s,c){const n=document.createElement(t);if(s!==undefined)n.textContent=s;if(c)n.className=c;return n}function clear(id){$(id).replaceChildren();return $(id)}function url(p){return '../'+p.split('/').map(encodeURIComponent).join('/')}function link(title,p){const a=el('a',title);a.href=url(p);a.target='_blank';a.rel='noopener';return a}function range(a,b){return '['+a+', '+b+')'}function inside(r,f){return r.start_frame<=f&&f<r.end_frame}function entity(id){if(!id)return null;const e=current.entities.find(x=>(x.entity_id||x.id)===id);return e?.name_zh||name(id)}function refs(r){const ids=[r.subject,r.object_id,r.secondary_object_id].filter(Boolean);return ids.map(x=>entity(x)).join(' → ')}
function state(box,r){const d=el('div',undefined,'record'),known=r.silver_label_valid===true&&(r.value===0||r.value===1);d.append(el('span',known?(r.value?'明确存在':'明确不存在'):'未知','tag '+(known?(r.value?'yes':'no'):'unknown')),el('strong',' '+name(r.type)),el('div',refs(r)),el('p',r.reason_zh||'未提供中文依据'));return box.appendChild(d)}
function select(s,f){current=s;clear('sequences');for(const row of D.sequences){const b=el('button',undefined,'seq'+(row===s?' active':''));b.dataset.sequence=row.sequence_id;b.append(el('strong',row.title_zh),el('div',`${SPLIT[row.split]} · ${row.window_count} 个窗口`),el('code',row.sequence_id),el('div',`连续查看 ${row.continuously_reviewed_frames} / ${row.frame_count} 帧`),el('div',`框质量：接受 ${row.quality_accepted} · 拒绝 ${row.quality_rejected}`));b.onclick=()=>select(row);$('sequences').append(b)}$('title').textContent=s.title_zh+' · '+SPLIT[s.split];$('coverage').textContent=`${s.sequence_id}｜连续语义查看 ${s.continuously_reviewed_frames} / ${s.frame_count} 帧；跟踪框质量查看 ${s.quality_reviewed_frames} 帧、${s.pages.length} 页。`;const ranges=clear('ranges');for(const r of s.reviewed_ranges){const b=el('button','语义已看 '+range(r.start_frame,r.end_frame),'frame-btn');b.onclick=()=>go(r.start_frame);ranges.append(b)}const ps=clear('page');s.pages.forEach((p,i)=>{const o=el('option',`第 ${i+1} / ${s.pages.length} 页 · 帧 ${p.frame_indices.join('、')}`);o.value=i;ps.append(o)});const legend=clear('legend');for(const e of s.legend){const d=el('span',e.short_id+' '+e.name_zh,'tag');const sw=el('i',undefined,'swatch');sw.style.background='rgb('+e.color_rgb.join(',')+')';d.prepend(sw);legend.append(d)}const es=clear('entities');for(const e of s.entities){const d=el('div',undefined,'entity');d.append(el('strong',e.name_zh||e.entity_id),el('code',e.entity_id||e.id));es.append(d)}const rejected=clear('rejections');for(const r of s.quality_ranges.rejected){const d=el('div',undefined,'record'),b=el('button',range(r.start_frame,r.end_frame));b.onclick=()=>go(r.start_frame);d.append(el('strong',entity(r.entity_id)),b,el('p',r.reason_zh));rejected.append(d)}if(!rejected.children.length)rejected.append(el('p','该段没有拒绝范围。'));const all=clear('all-states');for(const r of s.states){const d=state(all,r),b=el('button',range(r.start_frame,r.end_frame),'frame-btn');b.onclick=()=>go(r.start_frame);d.prepend(b)}const ts=clear('all-transitions');for(const r of s.transitions){const d=el('div',undefined,'record'),signal=r.event_family==='visual_signal',a=r.boundary_frame_range,b=el('button',range(a[0],a[1]),'frame-btn');b.onclick=()=>go(a[0]);const tag=signal?'视觉信号':r.censored?'删失或待定边界':r.frame_resolved?'单帧边界':'有界不确定范围';d.append(el('span',tag,'tag '+(signal||r.censored?'unknown':'yes')),el('strong',' '+name(r.type)),el('div',refs(r)),b,el('p',r.reason_zh));if(r.occurrence_confirmed===true)d.append(el('div','已确认发生；范围整体落入预测时间段时才提供事件监督。','muted'));ts.append(d)}const ls=clear('seq-links');for(const [title,path] of [['连续审核原文件',s.review_path],['跟踪质量原文件',s.tracking_review_path],['二维观测数据',s.observations_path],['手部数值数据',s.sequence_path],['当前帧种子',s.seeds_path]])ls.append(link(title,path));const p=clear('provenance');for(const [title,value] of [['训练索引',D.training_index_sha256],['连续审核',s.review_sha256],['跟踪质量',s.tracking_review_sha256],['观测数据',s.observations_sha256]])p.append(el('p',title+'：'+value,'hash'));$('frame').max=s.frame_count-1;go(Number.isInteger(f)?f:(s.pages[0]?.frame_indices[0]||0))}
function go(f){f=Number(f);if(!Number.isFinite(f))return;frame=Math.min(current.frame_count-1,Math.max(0,Math.trunc(f)));$('frame').value=frame;history.replaceState(null,'','#'+current.sequence_id+':'+frame);pageIndex=current.pages.findIndex(p=>p.frame_indices.includes(frame));$('page').value=pageIndex<0?'':pageIndex;$('prev-page').disabled=pageIndex<=0;$('next-page').disabled=pageIndex<0||pageIndex>=current.pages.length-1;const p=current.pages[pageIndex],buttons=clear('frames');if(p)for(const n of p.frame_indices){const b=el('button',String(n),'frame-btn'+(n===frame?' active':''));b.setAttribute('aria-label','查看原生帧 '+n);b.onclick=()=>go(n);buttons.append(b)}renderImage(p);const q=clear('quality'),record=current.frames.find(r=>r.frame_index===frame);$('frame-title').textContent='原生帧 '+frame+' · '+(record?'该帧已查看跟踪证据':'该帧没有可展示的已审跟踪页');if(record)for(const o of record.objects){const d=el('div',undefined,'record');d.append(el('span',o.accepted?'框质量接受':'框质量拒绝','tag '+(o.accepted?'yes':'no')),el('strong',' '+entity(o.entity_id)),el('code',o.entity_id),el('p',o.reason_zh));q.append(d)}if(!record)q.append(el('p','缺测：不显示邻帧替代图，也不补框。','unknown'));const active=clear('active-states');for(const r of current.states.filter(r=>inside(r,frame)))state(active,r);if(!active.children.length)active.append(el('p','未知：没有当前帧明确状态记录。','unknown'));const unknown=clear('unknowns');for(const r of current.unresolved.filter(r=>inside(r,frame))){const d=el('div',undefined,'record');d.append(el('strong',name(r.type)),el('div',refs(r)),el('p',r.reason_zh));unknown.append(d)}if(!unknown.children.length)unknown.append(el('p','其他未记录关系仍为未知，不能当作负例。','muted'))}
function renderImage(p){const g=++generation,img=$('scene'),meta=clear('image-meta');img.hidden=true;img.removeAttribute('src');$('missing').hidden=false;if(!p){$('missing').textContent='当前帧没有已审核跟踪图像；保留缺测。';return}$('missing').textContent='正在读取真实证据页…';img.onload=()=>{if(g!==generation)return;img.hidden=false;$('missing').hidden=true};img.onerror=()=>{if(g!==generation)return;img.hidden=true;$('missing').textContent='证据文件无法读取，请检查交付目录；不以其他帧替代。'};img.src=url(p.path);img.dataset.page=p.path;meta.append(el('div','本页展示 '+p.frame_indices.length+' 个实际查看的原生帧，当前选择 '+frame+' 帧；图片上的“尚未视觉审核”为渲染时标记，后续质检结果以右侧绑定记录为准。'),link('打开原尺寸联系表',p.path),el('div','证据页 SHA：'+p.sha256,'hash'))}
async function init(){const response=await fetch('data.json',{cache:'no-store'});if(!response.ok)throw Error('数据文件无法读取');D=await response.json();for(const t of D.event_types)N[t.code]=t.name_zh;const release=D.release,allowed=release?.interaction_training_allowed===true&&release?.training_scope===D.training_scope;$('scope').textContent=`当前交付范围：使用已标注二维初始化的助手银标交互预测子集。${allowed?'范围放行记录已允许该交互子集接入训练。':'当前页面快照尚未确认训练范围放行；正式加载仍须核验独立放行文件。'} 整物体 6D 尚未认证，300 段完整标注与全联合训练尚未完成。`;$('scope-meta').textContent='二维框与在线重放跟踪不产生整物体 6D；当前帧人工式种子参与初始化，尚未验证自动端到端感知。页面数据生成于 '+D.created_at+'。';const reviewed=D.sequences.reduce((a,s)=>a+s.continuously_reviewed_frames,0),windows=D.sequences.reduce((a,s)=>a+s.window_count,0),accepted=D.sequences.reduce((a,s)=>a+s.quality_accepted,0),rejected=D.sequences.reduce((a,s)=>a+s.quality_rejected,0);for(const [title,value] of [['当前子集 / 原始采集',D.sequences.length+' / '+D.original_acquisition_count],['实际连续查看帧',reviewed.toLocaleString('zh-CN')],['可加载窗口',windows.toLocaleString('zh-CN')],['接受 / 拒绝对象观测',accepted+' / '+rejected]]){const d=el('div',undefined,'stat');d.append(el('strong',value),el('span',title));$('stats').append(d)}for(const [split,row] of Object.entries(D.splits)){const tr=el('tr');for(const v of [SPLIT[split],row.sequence_count,row.windows,D.counts[split].known_positive_cells,D.counts[split].known_negative_cells])tr.append(el('td',String(v)));$('split-table').append(tr)}for(const item of [...D.links,...D.source_files]){const d=el('p');d.append(link(item.title_zh,item.path));$('files').append(d)}$('shapes').append(el('p',`历史 ${D.window_config.window_frames} 帧；预测至少 ${D.window_config.horizon_seconds} 秒；最多 ${D.window_config.max_objects} 个物体。`),el('p','输入实体特征：[批次, 32, 11, 120]；图标签与预测：[批次, 11, 11, 66]。只有明确监督掩码参与损失；标签与角色文本不进入编码器。'),el('p','训练索引 SHA：'+D.training_index_sha256,'hash'));$('page').onchange=()=>go(current.pages[Number($('page').value)].frame_indices[0]);$('prev-page').onclick=()=>go(current.pages[pageIndex-1].frame_indices[0]);$('next-page').onclick=()=>go(current.pages[pageIndex+1].frame_indices[0]);$('go-frame').onclick=()=>go($('frame').value);$('frame').onchange=()=>go($('frame').value);document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'&&pageIndex>0){e.preventDefault();$('prev-page').click()}if(e.key==='ArrowRight'&&pageIndex>=0&&pageIndex<current.pages.length-1){e.preventDefault();$('next-page').click()}});const [uid,n]=location.hash.slice(1).split(':');select(D.sequences.find(s=>s.sequence_id===uid)||D.sequences[0],n===undefined?undefined:Number(n))}
init().catch(e=>{$('error').hidden=false;$('error').textContent='页面加载失败：'+e.message;console.error(e)});
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    print(json.dumps({k: result[k] for k in ("training_index_sha256", "sequence_count", "window_count", "quality_accepted", "quality_rejected")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
