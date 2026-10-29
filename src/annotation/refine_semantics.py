"""Complete explicit source verbs without inventing physical interaction labels."""
import argparse,json,re
from pathlib import Path
from src.dataset.complete_egodemo import operation_candidates,OPERATIONS,save,digest

RULES=[
 ('tighten','旋紧',r'\bscrew\s+(?:(?:\w+\s+){0,8}(?:onto|into|on|in)|it\s+tight)\b'),
 ('insert','插入/装入',r'\bload(?:s|ed|ing)?\b'),
 ('close','合上',r'\bcover\s+(?:the|a|each)\b'),
 ('place','放置',r'\b(?:lay|return)\s+(?:the|a|each)\b'),
 ('extract','取出/拆出',r'\btake\s+(?:\w+\s+){1,8}(?:out\s+of|from)\b'),
 ('attach','贴附/连接',r'\b(?:stick(?:s|ing)?\s+(?:the|a|each)|reapply\s+(?:the\s+)?tape|(?:reinforce|secure)\b[^.;]{0,100}\btape)\b'),
 ('collect','收集/聚拢',r'\bsweep\b'),
 ('flip','翻转',r'\bturn\s+over\b'),
 ('lift','提起',r'\blift(?:s|ed|ing)?\b'),
 ('scoop','舀取',r'(?:^|\b(?:and|then|to)\s+)scoop(?:s|ed|ing)?\b'),
 ('portion','分装',r'\b(?:divide|distribute)\s+(?:the|a|each)\b'),
 ('clamp','夹固',r'(?:^|\b(?:and|then|to)\s+)clip\s+(?:the|a|each)\b'),
 ('disassemble','拆解',r'\bdisassembl\w*\b'),
 ('flatten','压平',r'\bflatten\w*\b'),
]
REFINED_OPERATIONS={**OPERATIONS,**{code:name for code,name,_ in RULES}}

def refined_operations(description):
    result={x['operation_type']:x for x in operation_candidates(description)}
    for code,name,pattern in RULES:
        matches=list(re.finditer(pattern,description,re.I))
        if matches and code not in result:
            result[code]={'operation_type':code,'operation_name_zh':name,
                          'text_evidence':[{'text':m.group(),'span':list(m.span())} for m in matches]}
    return [{'type':x['operation_type'],'name_zh':x['operation_name_zh'],
             'source_quote':x['text_evidence'][0]['text'],'text_evidence':x['text_evidence'],
             'label_source':'source_text_rule_v2','subject':None,'object_id':None,'label_valid':False}
             for x in result.values()]

def semantic_directory(root,uid):
    refined=root/'semantic_drafts_refined'/uid
    return refined if refined.exists() else root/'semantic_drafts'/uid

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();count=prior_empty=remaining=0;types={}
    for path in (a.root/'semantic_drafts').glob('*/chunk_*.json'):
        if '.error.' in path.name:continue
        data=json.loads(path.read_text())
        for row in data['records']:
            prior_empty+=not row['operations'];row['operations']=refined_operations(row['description_original'])
            row['operation_extraction_version']='explicit_source_verbs_v2';remaining+=not row['operations'];count+=1
            for op in row['operations']:types[op['type']]=types.get(op['type'],0)+1
        data['previous_semantic_chunk_sha256']=digest(path)
        save(a.root/'semantic_drafts_refined'/path.parent.name/path.name,data)
    report={'intervals':count,'previous_unmapped_intervals':prior_empty,'remaining_unmapped_intervals':remaining,
            'observed_operation_types':len(types),'operation_counts':types,'human_approved':0,'training_allowed':False}
    save(a.root/'semantic_refinement_report.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
