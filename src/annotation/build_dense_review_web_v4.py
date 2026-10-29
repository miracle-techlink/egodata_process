"""Build an offline, read-only Chinese browser for native-frame V4 evidence.

Serve the dataset root, then open /review_web/index.html. No approval, annotation
promotion, image interpolation, or server writes happen in the browser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside dataset: {relative}")
    return path


def build(root: Path) -> dict:
    root = root.resolve()
    report_path = root / "dense_observables_report.json"
    source = read(report_path)
    descriptions = defaultdict(list)
    priority = root / "dense_annotation_priority.jsonl"
    if priority.exists():
        for line in priority.read_text().splitlines():
            row = json.loads(line)
            descriptions[row["sequence_id"]].append({k: row.get(k) for k in
                ("source_interval_id", "description_zh", "start_frame", "end_frame")})
    sequences, bindings = [], []
    for entry in source["records"]:
        uid = entry["sequence_id"]
        meta_path = root / "dense_sequences" / uid / "metadata.json"
        meta = read(meta_path)
        if sha256(meta_path) != entry["metadata_sha256"]:
            raise ValueError(f"Metadata SHA mismatch: {uid}")
        candidate_path = meta_path.with_name("boundary_candidates.json")
        candidate_sha = sha256(candidate_path)
        if candidate_sha != meta["sha256"]["boundary_candidates"]:
            raise ValueError(f"Candidate SHA mismatch: {uid}")
        candidates = read(candidate_path)["records"]
        if len(candidates) != entry["candidate_boundary_count"]:
            raise ValueError(f"Candidate count mismatch: {uid}")
        review_path = root / "dense_reviews" / f"{uid}.json"
        review = read(review_path) if review_path.exists() else {}
        reviewed = set()
        for interval in review.get("reviewed_ranges", []):
            reviewed.update(range(interval["start_frame"], interval["end_frame"]))
        viewed_pages = set()
        for binding in review.get("evidence_manifests", []):
            if sha256(inside(root, binding["path"])) != binding["sha256"]:
                raise ValueError(f"Evidence manifest SHA mismatch: {uid}")
            viewed_pages.update(binding.get("reviewed_page_paths", []))
        frames = defaultdict(list)
        for manifest_path in sorted((root / "dense_evidence" / uid).glob("*/manifest.json")):
            manifest = read(manifest_path)
            for frame in manifest["frames"]:
                page = manifest["pages"][frame["page_index"]]
                crop_path = inside(root, frame["crop_path"])
                # A missing crop is never replaced by another frame or a synthetic preview.
                if not crop_path.is_file():
                    continue
                frames[frame["frame_index"]].append({
                    "path": frame["crop_path"], "sha256": frame["crop_sha256"],
                    "native_rgb_sha256": frame["decoded_native_rgb_sha256"],
                    "crop_xyxy": frame["crop_xyxy_native"],
                    "timestamp_s": frame["timestamp_s"], "page_path": page["path"],
                    "manifest_path": str(manifest_path.relative_to(root)),
                    "assistant_reviewed": frame["frame_index"] in reviewed and page["path"] in viewed_pages,
                })
        available_reviewed = {f for f, variants in frames.items() if any(v["assistant_reviewed"] for v in variants)}
        if available_reviewed != reviewed:
            raise ValueError(f"Reviewed range lacks a real viewed crop: {uid}")
        state_records = review.get("state_segments", [])
        transitions = review.get("transition_events", [])
        sequences.append({
            "sequence_id": uid, "split": meta["split"],
            "frame_count": entry["frames_indexed"], "reviewed_count": len(reviewed),
            "rendered_count": len(frames), "frames": frames, "candidates": candidates,
            "source_intervals": descriptions[uid], "entities": review.get("entities", []),
            "reviewed_ranges": review.get("reviewed_ranges", []),
            "states": state_records, "transitions": transitions,
            "unresolved": review.get("unresolved_ranges", []),
            "review_sha256": sha256(review_path) if review else None,
            "review_path": str(review_path.relative_to(root)) if review else None,
            "continuous_visual_review_complete": bool(review.get("full_sequence_continuous_review_complete", False)),
        })
        bindings.append({"sequence_id": uid, "metadata_sha256": entry["metadata_sha256"],
                         "candidates_sha256": candidate_sha,
                         "review_sha256": sha256(review_path) if review else None})
    sequences.sort(key=lambda s: (-s["reviewed_count"], s["sequence_id"]))
    payload = {"schema_version": "assistant_annotation_v4_dense_review_web", "sequences": sequences,
               "totals": {"sequences": len(sequences),
                          "native_frames": sum(s["frame_count"] for s in sequences),
                          "reviewed_frames": sum(s["reviewed_count"] for s in sequences),
                          "rendered_frames": sum(s["rendered_count"] for s in sequences),
                          "candidate_boundaries": sum(len(s["candidates"]) for s in sequences)},
               "read_only": True, "human_label_valid": False, "training_allowed": False}
    out = root / "review_web"
    out.mkdir(exist_ok=True)
    # Escaping '<' also keeps source strings harmless if this is embedded later.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (out / "data.js").write_text("window.DENSE_REVIEW_DATA=" + encoded + ";\n")
    (out / "index.html").write_text(HTML)
    result = {"schema_version": payload["schema_version"], "created_at": datetime.now(timezone.utc).isoformat(),
              "source_report_sha256": sha256(report_path), "builder_sha256": sha256(Path(__file__)),
              "totals": payload["totals"], "source_bindings": bindings,
              "files": {n: sha256(out / n) for n in ("index.html", "data.js")},
              "read_only": True, "annotation_changes": 0, "training_allowed": False,
              "serve_zh": "在数据集根目录运行 python -m http.server 8764 --bind 127.0.0.1，打开 /review_web/index.html"}
    (out / "build_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>连续逐帧审核 · EgoDemo V4</title>
<style>
:root{color-scheme:light;font-family:system-ui,"Noto Sans CJK SC",sans-serif;color:#14283c;background:#f2f5f8}*{box-sizing:border-box}body{margin:0}header{padding:22px 28px;background:#14283c;color:white}h1{font-size:23px;margin:0 0 8px}h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:16px 0 8px}p{line-height:1.6}button,select,input{font:inherit;border:1px solid #bac8d4;border-radius:6px;padding:7px;background:white;color:#14283c}button{cursor:pointer}button:hover{background:#e5f0fc}button:disabled{cursor:default;opacity:.45}a{color:#1568ad}small,.muted{color:#577088}.cards{display:flex;gap:12px;flex-wrap:wrap;padding:18px 28px}.stat{background:white;border:1px solid #dae3eb;border-radius:9px;padding:14px 18px;min-width:180px}.stat strong{font-size:26px;display:block}.layout{display:grid;grid-template-columns:290px minmax(400px,1fr) 350px;gap:16px;padding:0 28px 28px}section,aside{min-width:0}.panel{background:white;border:1px solid #dae3eb;border-radius:9px;padding:15px;margin-bottom:14px}.list{max-height:67vh;overflow:auto}.sequence{display:block;width:100%;text-align:left;margin:6px 0;padding:11px;line-height:1.6}.sequence.active{border-color:#2378be;background:#edf6ff}.sequence code{font-size:11px;display:block;overflow-wrap:anywhere}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}#frame{width:95px}#scrub{width:100%;padding:0;margin:15px 0}#scene{display:block;width:100%;max-height:65vh;object-fit:contain;background:#14202b}#scene[hidden]{display:none}#missing{padding:60px 20px;text-align:center;background:#edf1f5;border:1px dashed #bac8d4;border-radius:8px}#missing[hidden]{display:none}#image-meta{font-size:12px;overflow-wrap:anywhere;line-height:1.7;margin-top:9px}.tag{display:inline-block;border-radius:4px;padding:3px 7px;font-size:12px;background:#edf1f5}.positive{background:#def5e8;color:#12623a}.negative{background:#ffebe8;color:#923827}.unknown{background:#fff1d6;color:#785717}.record{border-top:1px solid #e1e7ee;padding:10px 0;line-height:1.5;font-size:13px}.record code{font-size:11px;overflow-wrap:anywhere;display:block}.record p{margin:5px 0}.full{width:100%}#candidates{max-height:360px;overflow:auto}.candidate{display:block;text-align:left;width:100%;margin:6px 0;font-size:12px;line-height:1.5}.pill{font-size:12px;display:inline-block;margin:3px;padding:4px 8px;background:#edf6ff;border-radius:4px;cursor:pointer}details{font-size:12px;overflow-wrap:anywhere}summary{cursor:pointer}#error{padding:20px;color:#9c2424}#coverage{margin:10px 0;font-size:13px}.notice{font-size:13px;line-height:1.6;padding:9px;background:#fff5e3;border-radius:6px}.linkrow{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}@media(max-width:1250px){.layout{grid-template-columns:230px minmax(350px,1fr)}.right{grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:760px){.layout{display:block;padding:0 12px}.cards{padding:12px}.list{max-height:250px}.right{display:block}header{padding:18px}.stat{min-width:140px;flex:1}}
</style>
<header><h1>连续逐帧审核 · EgoDemo V4</h1><div>原生帧证据、明确正负状态、未知范围与转换边界。助手复核与人工金标准分开记录。</div></header>
<div id="error" hidden></div><div class="cards" id="stats"></div>
<main class="layout">
<aside><div class="panel"><h2>采集列表</h2><input id="search" class="full" aria-label="搜索采集" placeholder="搜索 UUID / 中文操作">
<select id="filter" class="full" aria-label="采集筛选" style="margin-top:8px"><option value="all">全部采集</option><option value="reviewed">已有连续复核</option><option value="pending">尚无连续复核</option><option value="candidates">有待审接近候选</option></select><div id="sequence-list" class="list"></div></div></aside>
<section><div class="panel"><h2 id="title">选择采集</h2><div id="coverage"></div><div id="ranges"></div>
<div class="toolbar"><button id="prev" aria-label="上一原生帧">← 上一帧</button><label>帧 <input type="number" id="frame" min="0" value="0"></label><button id="next" aria-label="下一原生帧">下一帧 →</button><label>裁剪 <select id="variant" aria-label="选择证据裁剪"></select></label></div>
<input id="scrub" type="range" min="0" value="0" aria-label="原生帧时间轴"><div class="muted" style="font-size:12px;margin-bottom:8px">键盘 ← / → 逐帧，Shift + ← / → 移动 10 帧。所有区间为 [起始帧, 结束帧)，结束帧不包含。</div>
<img id="scene" alt="当前原生帧的真实局部证据" hidden><div id="missing">当前帧尚未生成连续图像证据，不能在此确认状态。</div><div id="image-meta"></div></div>
<div class="panel"><h2>待审接近变化候选 <span id="candidate-count"></span></h2><div class="notice">来自三维距离阈值变化，仅用于定位审核范围；不等于接触、抓持或释放。候选状态仍待审，看到其他标注不自动将它通过。</div><div class="toolbar" style="margin-top:10px"><select id="candidate-filter" aria-label="候选类型"><option value="all">全部候选</option><option value="proximity_enter">接近阈值进入</option><option value="proximity_exit">接近阈值离开</option></select><button id="candidate-prev">上一页</button><span id="candidate-page"></span><button id="candidate-next">下一页</button></div><div id="candidates"></div></div>
<div class="panel"><h2>源文本区间</h2><div class="muted">用于任务语义导航；原文描述不直接作为当前帧的物理关系。</div><div id="source-intervals"></div></div></section>
<aside class="right"><div class="panel"><h2>当前帧状态</h2><div class="notice">只展示明确支持的正例与负例。没有记录、未见清楚或未审核均为未知，不当作负例。</div><div id="states"></div><h3>当前未知范围</h3><div id="unknowns"></div></div>
<div class="panel"><h2>已复核转换与视觉信号</h2><div id="transitions"></div><h3>实体与来源</h3><div id="entities"></div><div id="provenance"></div></div></aside>
</main><script src="data.js"></script><script>
'use strict';
const D=window.DENSE_REVIEW_DATA,$=id=>document.getElementById(id),N={left_hand:'左手',right_hand:'右手',both_hands:'双手',object:'物体',contact:'接触',grasp_control:'抓持控制',supported_by:'受到支撑',co_motion:'共同运动',stabilizes:'稳定工件',acts_on:'作用于',scan:'扫描（可见照明阶段）',tool_mediated:'工具介导协作',bimanual_joint:'双手共同操作',bimanual_complementary:'双手互补',handover:'交接',single_hand:'单手操作',contact_onset:'开始接触',grasp_onset:'建立抓持',motion_onset:'开始运动',support_gained:'建立支撑',support_lost:'失去支撑',release:'最终脱离',illumination_onset:'照明开启',illumination_offset:'照明关闭',proximity_enter:'接近阈值进入',proximity_exit:'接近阈值离开'};
let current=null,frame=0,variant=0,candidatePage=0,imageGeneration=0;
const label=x=>N[x]||x||'未指定';
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function clear(id){$(id).replaceChildren();return $(id)}
function asset(path){return '../'+path.split('/').map(encodeURIComponent).join('/')}
function entity(id){const e=current.entities.find(e=>(e.entity_id||e.id)===id);return e?.name_zh||e?.display_name||label(id)}
function refs(r){return [r.subject,r.object_id,r.secondary_object_id].filter(Boolean).map(x=>entity(x)+' ('+x+')').join(' → ')}
function range(r){return '['+r[0]+', '+r[1]+')'}
function inRange(r,f){return r.start_frame<=f&&f<r.end_frame}
function record(container,r,status){const box=el('div',undefined,'record');box.append(el('span',status.text,'tag '+status.cls),el('strong',' '+label(r.type)),el('code',refs(r)));if(r.reason_zh)box.append(el('p',r.reason_zh));container.append(box);return box}
function showList(){const box=clear('sequence-list'),q=$('search').value.toLowerCase(),filter=$('filter').value;for(const s of D.sequences){const desc=s.source_intervals.map(x=>x.description_zh||'').join(' ');if(q&&!(s.sequence_id+' '+desc).toLowerCase().includes(q))continue;if(filter==='reviewed'&&!s.reviewed_count||filter==='pending'&&s.reviewed_count||filter==='candidates'&&!s.candidates.length)continue;const b=el('button',undefined,'sequence'+(s===current?' active':''));b.dataset.sequence=s.sequence_id;b.append(el('strong',desc.slice(0,42)||'任务描述待补'),el('code',s.sequence_id),el('span',`${s.reviewed_count} / ${s.frame_count} 帧已连续查看 · ${s.candidates.length} 个待审候选`));b.onclick=()=>select(s);box.append(b)}}
function select(s,f){current=s;candidatePage=0;variant=0;frame=Number.isInteger(f)?f:(Number(Object.keys(s.frames)[0])||0);$('title').textContent=s.sequence_id;$('coverage').textContent=`${s.split} · 原生 ${s.frame_count} 帧 · 已连续查看 ${s.reviewed_count} 帧 · 已有图像 ${s.rendered_count} 帧；已看不等于所有标注层均完成。`;$('frame').max=$('scrub').max=s.frame_count-1;const ranges=clear('ranges');for(const r of s.reviewed_ranges){const b=el('button','已看 '+range([r.start_frame,r.end_frame]),'pill');b.onclick=()=>go(r.start_frame);ranges.append(b)}const ts=clear('transitions');for(const r of s.transitions){const b=record(ts,r,{text:r.event_family==='visual_signal'?'视觉信号':r.frame_resolved&&!r.censored?'物理精确帧':'物理范围边界',cls:r.frame_resolved&&!r.censored?'positive':'unknown'});const jump=el('button',range(r.boundary_frame_range));jump.onclick=()=>go(r.boundary_frame_range[0]);b.append(jump);if(r.event_family==='visual_signal')b.append(el('p','此信号不等于成功操作或物理接触。','muted'))}if(!s.transitions.length)ts.append(el('p','暂无已复核转换。','muted'));const es=clear('entities');for(const e of s.entities){const box=el('div',undefined,'record');box.append(el('strong',e.name_zh||e.display_name||e.entity_id),el('code',e.entity_id||e.id));if(e.geometry_status==='unavailable')box.append(el('span','该实体无对应整物体 6D','muted'));es.append(box)}if(!s.entities.length)es.append(el('p','尚无连续复核实体。候选中的 obj ID 不表示身份已经确认。','muted'));const p=clear('provenance');if(s.review_path){const a=el('a','查看连续审核原 JSON');a.href=asset(s.review_path);a.target='_blank';p.append(a,el('p','助手复核 SHA: '+s.review_sha256,'muted'))}const source=clear('source-intervals');for(const r of s.source_intervals){const b=el('button',range([r.start_frame,r.end_frame])+' '+r.description_zh,'candidate');b.onclick=()=>go(r.start_frame);source.append(b)}showList();showCandidates();go(frame)}
function go(f){if(!current)return;const n=Number(f);if(!Number.isFinite(n))return;frame=Math.max(0,Math.min(current.frame_count-1,Math.trunc(n)));$('frame').value=$('scrub').value=frame;history.replaceState(null,'','#'+current.sequence_id+':'+frame);const variants=current.frames[String(frame)]||[];const vsel=clear('variant');variants.forEach((v,i)=>{const o=el('option',(i+1)+' · '+v.crop_xyxy.join(','));o.value=i;vsel.append(o)});variant=Math.min(variant,Math.max(0,variants.length-1));vsel.value=variant;vsel.disabled=variants.length<2;renderImage();const state=clear('states'),unknown=clear('unknowns');const active=current.states.filter(r=>inRange(r,frame)&&r.silver_label_valid===true&&(r.value===0||r.value===1));for(const r of active)record(state,r,{text:r.value===1?'正例：存在':'负例：明确不存在',cls:r.value===1?'positive':'negative'});if(!active.length)state.append(el('p','未知：当前帧没有明确正负状态记录。','unknown'));for(const r of current.unresolved.filter(r=>inRange(r,frame)))record(unknown,r,{text:'未知',cls:'unknown'});if(!unknown.children.length)unknown.append(el('p','未单列未知范围；未记录的其他关系仍为未知。','muted'));$('prev').disabled=frame===0;$('next').disabled=frame===current.frame_count-1}
function renderImage(){const generation=++imageGeneration,v=(current.frames[String(frame)]||[])[variant],img=$('scene');img.hidden=true;img.removeAttribute('src');$('missing').hidden=false;clear('image-meta');if(!v){$('missing').textContent='第 '+frame+' 帧尚未生成连续图像证据，不能在此确认状态。';return}$('missing').textContent='正在读取第 '+frame+' 帧的真实证据…';img.onload=()=>{if(generation!==imageGeneration)return;img.hidden=false;$('missing').hidden=true};img.onerror=()=>{if(generation!==imageGeneration)return;img.hidden=true;$('missing').hidden=false;$('missing').textContent='当前帧证据文件无法读取；请从完整 V4 数据目录提供页面。'};img.dataset.frame=frame;img.src=asset(v.path);const meta=$('image-meta');meta.append(el('span',`第 ${frame} 帧 · 原生时间 ${Number(v.timestamp_s).toFixed(6)} 秒 · ${v.assistant_reviewed?'助手已连续查看':'仅已渲染，尚未复核'}`));meta.append(el('div','原生裁剪坐标 xyxy: '+v.crop_xyxy.join(', ')));const links=el('div',undefined,'linkrow');for(const [title,path] of [['原像素裁剪',v.path],['所在连续联系表',v.page_path],['证据清单',v.manifest_path]]){const a=el('a',title);a.href=asset(path);a.target='_blank';links.append(a)}meta.append(links);const detail=el('details');detail.append(el('summary','核验来源 SHA'),el('p','完整原生 RGB: '+v.native_rgb_sha256),el('p','裁剪文件: '+v.sha256));meta.append(detail)}
function showCandidates(){const box=clear('candidates'),kind=$('candidate-filter').value,rows=current.candidates.filter(r=>kind==='all'||r.candidate_type===kind),pages=Math.max(1,Math.ceil(rows.length/30));candidatePage=Math.min(candidatePage,pages-1);$('candidate-count').textContent='('+current.candidates.length+')';$('candidate-page').textContent=(candidatePage+1)+' / '+pages;$('candidate-prev').disabled=candidatePage===0;$('candidate-next').disabled=candidatePage===pages-1;for(const r of rows.slice(candidatePage*30,(candidatePage+1)*30)){const b=el('button',undefined,'candidate');b.dataset.candidateFrame=r.candidate_boundary_frame;b.append(el('strong',`帧 ${r.candidate_boundary_frame} · ${label(r.candidate_type)} · 待审`),el('div',`${entity(r.subject)} → ${r.object_id} · 建议 ${range(r.suggested_review_range)}`),el('div',current.frames[r.candidate_boundary_frame]?'该帧有真实证据':'该帧待生成证据','muted'));b.onclick=()=>go(r.candidate_boundary_frame);box.append(b)}if(!rows.length)box.append(el('p','此采集没有符合筛选的候选。','muted'))}
try{for(const [name,value] of [['独立采集',D.totals.sequences],['原生帧',D.totals.native_frames],['实际连续查看帧',D.totals.reviewed_frames],['待审接近变化候选',D.totals.candidate_boundaries]]){const s=el('div',undefined,'stat');s.append(el('strong',value.toLocaleString('zh-CN')),el('span',name));$('stats').append(s)}$('search').oninput=$('filter').onchange=showList;$('prev').onclick=()=>go(frame-1);$('next').onclick=()=>go(frame+1);$('frame').onchange=()=>go($('frame').value);$('scrub').oninput=()=>go($('scrub').value);$('variant').onchange=()=>{variant=Number($('variant').value);renderImage()};$('candidate-filter').onchange=()=>{candidatePage=0;showCandidates()};$('candidate-prev').onclick=()=>{candidatePage--;showCandidates()};$('candidate-next').onclick=()=>{candidatePage++;showCandidates()};document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();go(frame+(e.key==='ArrowLeft'?-1:1)*(e.shiftKey?10:1))}});const hash=location.hash.slice(1).split(':'),initial=D.sequences.find(s=>s.sequence_id===hash[0])||D.sequences[0];if(initial)select(initial,hash.length>1?Number(hash[1]):undefined)}catch(e){$('error').hidden=false;$('error').textContent='加载失败：'+e.message;console.error(e)}
</script></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    print(json.dumps({"output": str(args.root / "review_web"), "totals": result["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
