"""Offline English report for current inputs, authentic outputs, and review status."""

from __future__ import annotations

from collections import Counter
from html import escape
import os
from pathlib import Path
from urllib.parse import quote

from .core import checked_path, load_tasks, read_json, sha, validate_result, write_json


def audit_run(root: Path) -> dict:
    root = Path(root).resolve()
    records = [validate_result(root, task["case_id"]) for task in load_tasks(root)]
    hashes = []
    for task in load_tasks(root):
        path = checked_path(root, task["output_dir"]) / "raw.png"
        if path.is_file():
            hashes.append(sha(path))
    duplicates = [digest for digest, count in Counter(hashes).items() if count > 1]
    report = {"case_count": len(records), "records_valid": all(r["records_valid"] for r in records) and not duplicates,
              "all_accepted": all(r["valid"] for r in records) and not duplicates,
              "duplicate_raw_hashes": duplicates, "cases": records,
              "interior_pixel_alignment_proven": False}
    write_json(root / "batch_validation.json", report)
    return report


def build_report(root: Path) -> Path:
    root = Path(root).resolve()
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_run(root)
    statuses = {row["case_id"]: row for row in audit["cases"]}

    def picture(path, label):
        if not path or not path.is_file():
            return f"<figure><figcaption>{escape(label)}</figcaption><p>Unavailable</p></figure>"
        relative = quote(os.path.relpath(path, report_dir).replace(os.sep, "/"), safe="/")
        return (f'<figure><figcaption>{escape(label)}</figcaption><a href="{relative}">'
                f'<img loading="lazy" src="{relative}" alt="{escape(label)}"></a></figure>')

    def record(path, label):
        if not path.is_file():
            return ""
        return f"<details><summary>{escape(label)}</summary><pre>{escape(path.read_text(encoding='utf-8'))}</pre></details>"

    cards = []
    for task in load_tasks(root):
        case_id = task["case_id"]
        status = statuses[case_id]
        output = checked_path(root, task["output_dir"])
        images = picture(checked_path(root, task["front_rgb"]), "Front RGB reference")
        images += picture(checked_path(root, task["back_normal"]), "Back camera normals")
        images += picture(output / "raw.png", "Actual generated raw")
        images += picture(output / "back_rgb.png" if status["records_valid"] else None, "Back RGB after mask constraint")
        images += picture(output / "overlay.png" if status["records_valid"] else None, "Pre-clipping alignment overlay")
        notes = "".join(f"<li>{escape(note)}</li>" for note in status["errors"] + status["warnings"])
        details = "".join(record(output / name, label) for name, label in (
            ("visual_review.json", "Visual review"), ("alignment.json", "Alignment and lighting diagnostics"),
            ("provenance.json", "Generation provenance"), ("worker_status.json", "Astra worker completion"),
            ("failure.json", "Generation failure")))
        details += record(checked_path(root, task["cameras"]), "Original camera metadata")
        accepted = "Accepted for inference" if status["valid"] else "Not accepted for inference"
        cards.append(f'<article data-case="{escape(case_id, quote=True)}"><h2>{escape(case_id)}</h2>'
                     f'<p><strong>{accepted}</strong> · Visual status: {escape(status["status"])}</p>'
                     f'<div class="images">{images}</div><ul>{notes}</ul>{details}</article>')
    html = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SupraTex back-view generation review</title>
<style>
body{font:15px/1.55 system-ui,sans-serif;background:#111820;color:#e5edf5;margin:0;padding:32px}
main{max-width:1500px;margin:auto}h1{font-size:30px}h2{font-size:21px}a{color:#8ad6d0}
article{padding:22px;background:#19242f;border:1px solid #344554;border-radius:10px;margin:24px 0}
.images{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}figure{margin:0;min-width:0}
figcaption{font-size:12px;margin:6px 0}img{width:100%;aspect-ratio:1;object-fit:contain;background:#e7e9ec}
details{margin-top:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0c1219;padding:16px}
.note{background:#22313f;padding:16px;border-left:3px solid #8ad6d0}input{padding:10px;width:280px;max-width:80%}
@media(max-width:850px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}body{padding:16px}}
</style><main><h1>SupraTex back-view generation review</h1>
<p>Compare the original inputs, actual generation, permitted postprocessing, and recorded Astra review.</p>
<p class="note">GPT-6 Astra orchestrates the built-in image tool; the underlying synthesis model is undisclosed.
Final alpha is constrained to the renderer mask. This does not prove internal pixel correspondence.
Stale or incomplete artifacts are excluded from final previews. Camera geometry is not inferred from placeholder metadata.</p>
"""
    html += (f'<p>Cases: {audit["case_count"]} · Accepted: {sum(row["valid"] for row in audit["cases"])}'
             f' · Audit: <a href="../batch_validation.json">full JSON</a></p>'
             '<label>Filter by case ID <input id="filter" type="search"></label>' + "".join(cards))
    html += """</main><script>
document.querySelector('#filter').addEventListener('input',function(){
  const query=this.value.toLowerCase();
  for(const item of document.querySelectorAll('article'))item.hidden=!item.dataset.case.toLowerCase().includes(query);
});</script></html>
"""
    destination = report_dir / "index.html"
    destination.write_text(html, encoding="utf-8")
    write_json(report_dir / "summary.json", {"case_count": audit["case_count"],
               "accepted": sum(row["valid"] for row in audit["cases"]), "records_valid": audit["records_valid"]})
    return destination
