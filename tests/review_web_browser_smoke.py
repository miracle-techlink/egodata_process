"""Isolated browser QA; exported test decisions are never production annotations."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT=Path('runs/pose_relation_annotation_20261023/web_qa');OUT.mkdir(parents=True,exist_ok=True)
URL='http://127.0.0.1:8741'
checks=[]
def ok(name):
    checks.append(name);print('PASS',name,flush=True)

def state(page):
    return page.evaluate("JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k=>k.startsWith('hoi-review:')))).data")

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1050},accept_downloads=True)
    page=context.new_page();errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(URL,wait_until='networkidle')
    page.wait_for_function("document.querySelector('#scene').dataset.frame==='100'")
    page.locator('[data-review=accepted]').click()
    assert '审核人' in page.locator('#form-error').inner_text()
    ok('approval without reviewer is blocked')
    page.locator('#reviewer').fill('自动化测试（不是真实审核）')
    page.locator('[data-review=accepted]').click()
    assert state(page)['relations'][1]['review_status']=='accepted'
    page.locator('#notes').fill('自动化测试：修改后须重新审核')
    page.wait_for_timeout(900)
    assert state(page)['relations'][1]['review_status']=='pending'
    ok('explicit approval, provenance and automatic draft invalidation')
    page.locator('#end-frame').fill('59');page.locator('[data-review=pending]').click()
    assert page.locator('#form-error').is_visible()
    assert state(page)['relations'][1]['end_frame']==395
    page.locator('#cancel-edits').click()
    assert page.locator('#end-frame').input_value()=='394'
    ok('invalid interval rejected and unsaved edit can be cancelled')
    before=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
    page.locator('#show-raw').click()
    after=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
    assert before!=after
    page.locator('#show-overlay').click()
    page.locator('#frame-number').fill('445');page.locator('#frame-number').press('Enter')
    page.wait_for_function("document.querySelector('#scene').dataset.frame==='445'")
    page.locator('#prev-frame').click()
    page.wait_for_function("document.querySelector('#scene').dataset.frame==='444'")
    ok('raw toggle and exact native last/previous frames')
    page.locator('#frame-number').fill('100');page.locator('#frame-number').press('Enter')
    page.wait_for_function("document.querySelector('#scene').dataset.frame==='100'")
    page.locator('#play').click();page.wait_for_timeout(800)
    assert page.locator('#scene').get_attribute('data-frame')!='100'
    page.locator('#play').click()
    ok('continuous video playback with pose updates')
    page.locator('.tabs [data-group=transitions]').click()
    page.locator('.record-card[data-id=let_go]').click()
    page.locator('#start-frame').fill('405');page.locator('#end-frame').fill('405')
    page.locator('[data-remove-evidence="400"]').click();page.locator('[data-remove-evidence="420"]').click()
    page.locator('#frame-number').fill('405');page.locator('#frame-number').press('Enter')
    page.wait_for_function("document.querySelector('#scene').dataset.frame==='405'")
    page.locator('#add-current').click();page.locator('[data-review=accepted]').click()
    event=state(page)['transitions'][-1]
    assert (event['start_frame'],event['end_frame'],event['evidence_frames'])==(405,406,[405])
    ok('single-frame event inclusive UI to exclusive export boundary')
    page.locator('.tabs [data-group=hands]').click()
    page.wait_for_function("document.querySelector('#scene').dataset.frame==='363'")
    assert page.locator('#hand-warning').is_visible()
    page.locator('#notes').fill('自动化测试：左手槽位待复核；不改变原骨架')
    page.locator('[data-review=accepted]').click()
    assert state(page)['handedness_reviews'][0]['frame_id']==363
    ok('handedness review stored separately from source pose')
    page.locator('.tabs [data-group=cooperations]').click()
    page.locator('[data-review=accepted]').click()
    assert '先通过' in page.locator('#form-error').inner_text()
    page.locator('.tabs [data-group=relations]').click()
    for identity in ['control','lift','return']:
        page.locator(f'.record-card[data-id="{identity}"]').click()
        page.locator('[data-review=accepted]').click()
    page.locator('.tabs [data-group=cooperations]').click()
    page.locator('[data-review=accepted]').click()
    assert state(page)['cooperations'][0]['review_status']=='accepted'
    page.locator('.tabs [data-group=relations]').click()
    page.locator('.record-card[data-id=control]').click()
    page.locator('#notes').fill('自动化测试：第二次修改')
    page.wait_for_timeout(900)
    assert state(page)['cooperations'][0]['review_status']=='pending'
    ok('cooperation dependency approval and cascading invalidation')
    page.locator('#edit-names').click();page.locator('#name-G04_1').fill('松饼粉瓶（审核名）');page.locator('#save-names').click()
    assert state(page)['entities'][2]['entity_id']=='G04_1'
    assert state(page)['entities'][2]['display_name']=='松饼粉瓶（审核名）'
    ok('Chinese display-name edits preserve object identity')
    page.locator('#export').click()
    with page.expect_download() as dl:page.locator('#download-json').click()
    dl.value.save_as(OUT/'qa_export.json')
    exported=json.loads((OUT/'qa_export.json').read_text())
    assert exported['web_review']['base_annotation_sha256']
    with page.expect_download() as dl:page.locator('#download-csv').click()
    dl.value.save_as(OUT/'qa_export.csv')
    assert (OUT/'qa_export.csv').read_text().startswith('\ufeff"标注层"')
    ok('downloaded full JSON and Chinese CSV')
    page.locator('#close-dialog').click();page.reload(wait_until='networkidle')
    assert state(page)['relations'][1]['notes']=='自动化测试：第二次修改'
    ok('draft persistence after reload')
    clean=browser.new_context(viewport={'width':1512,'height':1100},accept_downloads=True)
    q=clean.new_page();q.on('pageerror',lambda e:errors.append(str(e)))
    q.goto(URL,wait_until='networkidle');q.wait_for_function("document.querySelector('#scene').dataset.frame==='100'")
    q.screenshot(path=str(OUT/'desktop_final.png'),full_page=True)
    q.locator('#export').click()
    bad=json.loads(json.dumps(exported));bad['source_bindings']['sequence']['sha256']='wrong'
    (OUT/'qa_bad_import.json').write_text(json.dumps(bad))
    q.locator('#import-file').set_input_files(str(OUT/'qa_bad_import.json'))
    q.wait_for_timeout(200)
    assert '导入失败' in q.locator('#toast').inner_text()
    q.locator('#import-file').set_input_files(str(OUT/'qa_export.json'))
    q.locator('#confirm-import').click()
    assert state(q)['handedness_reviews'][0]['review_status']=='accepted'
    assert state(q)['relations'][1]['notes']=='自动化测试：第二次修改'
    ok('source-mismatched import rejected, valid backup imported')
    mobile=browser.new_context(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
    m=mobile.new_page();m.goto(URL,wait_until='networkidle');m.wait_for_function("document.querySelector('#scene').dataset.frame==='100'")
    assert m.evaluate('document.documentElement.scrollWidth')==390
    m.screenshot(path=str(OUT/'mobile_final.png'),full_page=True)
    ok('mobile layout has no horizontal overflow')
    assert not errors,errors
    ok('no browser runtime errors')
    (OUT/'browser_report.json').write_text(json.dumps({'passed':True,'checks':checks,'runtime_errors':errors,'test_reviews_are_not_production':True},ensure_ascii=False,indent=2))
    browser.close()
