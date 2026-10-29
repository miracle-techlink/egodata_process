"""Real-data read-only browser checks; never produces annotation decisions."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18790/review_web/index.html")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    build_url = args.url.rsplit("/", 1)[0] + "/build_report.json"
    build = json.load(urlopen(build_url))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("document.querySelector('#scene').complete && document.querySelector('#scene').naturalWidth>0")
        assert page.locator(".sequence").count() == build["totals"]["sequences"] == 300
        assert "10,748" in page.locator("#stats").inner_text()
        assert "1,054" in page.locator("#stats").inner_text()
        checks.append("300段/1054实际连续查看帧/10748待审候选显示正确")
        assert page.locator("#scene").get_attribute("data-frame") == "0"
        assert "负例：明确不存在" in page.locator("#states").inner_text()
        page.locator("h1").click()
        page.keyboard.press("ArrowRight")
        page.wait_for_function("document.querySelector('#scene').dataset.frame==='1' && document.querySelector('#scene').complete")
        checks.append("键盘0→1按原生帧推进，真实crop解码成功")

        def goto(frame):
            page.locator("#frame").fill(str(frame))
            page.locator("#frame").dispatch_event("change")

        goto(68)
        assert "抓持控制" in page.locator("#unknowns").inner_text()
        goto(153)
        page.wait_for_function("document.querySelector('#scene').complete && document.querySelector('#scene').naturalWidth>0")
        assert "扫描" in page.locator("#states").inner_text()
        assert "照明开启" in page.locator("#transitions").inner_text()
        page.screenshot(path=str(args.out / "scanning_frame_153.png"))
        checks.append("68帧未知/153帧可见照明正例/独立视觉信号保持区分")
        goto(300)
        assert page.locator("#scene").is_hidden()
        assert page.locator("#scene").get_attribute("src") is None
        assert "第 300 帧尚未生成" in page.locator("#missing").inner_text()
        assert "未知" in page.locator("#states").inner_text()
        checks.append("缺图清空旧src与旧图，未记录状态为未知而非负例")
        page.locator("[data-sequence='acc3c38d-9591-4eaa-9e6b-d98e385249c8']").click()
        goto(319)
        page.wait_for_function("document.querySelector('#scene').complete && document.querySelector('#scene').naturalWidth>0")
        assert "已连续查看 320 帧" in page.locator("#coverage").inner_text()
        assert "助手已连续查看" in page.locator("#image-meta").inner_text()
        assert page.locator("#next").is_disabled()
        checks.append("第三段冻结320帧已显示，末帧319不越界")
        page.locator("#filter").select_option("pending")
        page.locator(".sequence").first.click()
        page.locator(".candidate[data-candidate-frame]").first.click()
        assert page.locator("#scene").is_hidden()
        assert "尚未生成" in page.locator("#missing").inner_text()
        page.screenshot(path=str(args.out / "pending_candidate.png"))
        checks.append("未渲染候选可定位但明确待审，不填替代图片")
        assert not errors, errors
        browser.close()
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "url": args.url,
              "checks_passed": checks, "browser_errors": errors,
              "source_web_build": build, "test_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "annotation_changes": 0, "optimizer_steps": 0,
              "screenshots": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob("*.png")}}
    (args.out / "browser_qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": len(checks), "browser_errors": errors, "totals": build["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
