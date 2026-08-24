#!/usr/bin/env python3
"""Render results/<sm>/*.yaml into a standalone HTML report (no deps, no CDN).

    python3 collector/facts/gen_report.py --sm sm100 --diff-sm sm90 \
        --out collector/facts/results/sm100/report.html

One file per SM dir; the diff view compares identity fields (verdict,
attention, moe, kv_allocated) for every (checkpoint, framework) present in
both SM dirs — the cross-platform upgrade-audit at a glance.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
IDENT_FIELDS = ("attention", "moe", "kv_allocated")


def load_sm(sm: str) -> dict:
    """{framework: {"_meta": ..., "results": {repo: cell}}} for one SM dir."""
    out = {}
    for p in sorted((ROOT / "results" / sm).glob("*.yaml")):
        doc = yaml.safe_load(p.read_text())
        if isinstance(doc, dict) and "results" in doc:
            out[doc["_meta"]["framework"]] = doc
    return out


def build_diff(cur: dict, prev: dict) -> list[dict]:
    rows = []
    for be, doc in cur.items():
        pdoc = prev.get(be)
        if not pdoc:
            continue
        for repo, cell in doc["results"].items():
            pcell = pdoc["results"].get(repo)
            if not pcell:
                continue
            changes = {}
            if cell.get("verdict") != pcell.get("verdict"):
                changes["verdict"] = [pcell.get("verdict"), cell.get("verdict")]
            for f in IDENT_FIELDS:
                a, b = pcell.get(f), cell.get(f)
                if a != b and (a or b):
                    changes[f] = [a, b]
            if changes:
                rows.append({"repo": repo, "framework": be, "changes": changes,
                             "prev": pcell, "cur": cell})
    return rows


HTML_TMPL = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>facts probe report — __SM__</title>
<style>
:root { color-scheme: light dark; }
body {
  margin: 0; padding: 24px; font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
  background: #fcfcfb; color: #0b0b0b;
  --surface-2: #f0efec; --border: #dcdbd6; --text-2: #52514e;
  --good: #0ca30c; --warn: #fab219; --crit: #d03b3b; --accent: #2a78d6;
  --good-bg: #0ca30c1a; --warn-bg: #fab2192a; --crit-bg: #d03b3b1a;
}
@media (prefers-color-scheme: dark) {
  body {
    background: #1a1a19; color: #ffffff;
    --surface-2: #262624; --border: #3a3a37; --text-2: #c3c2b7; --accent: #3987e5;
  }
}
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-2); margin-bottom: 20px; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
.tile {
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 18px; min-width: 150px;
}
.tile .fw { font-weight: 600; margin-bottom: 6px; }
.tile .n { font-size: 24px; font-weight: 700; }
.tile .row { color: var(--text-2); font-size: 12.5px; }
.tabs { display: flex; gap: 8px; margin-bottom: 14px; }
.tabs button {
  border: 1px solid var(--border); background: var(--surface-2); color: inherit;
  border-radius: 6px; padding: 6px 14px; cursor: pointer; font: inherit;
}
.tabs button.on { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.filters { display: flex; gap: 10px; margin-bottom: 12px; align-items: center; }
.filters input, .filters select {
  font: inherit; color: inherit; background: var(--surface-2);
  border: 1px solid var(--border); border-radius: 6px; padding: 5px 9px;
}
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid var(--border); padding: 6px 9px; text-align: left; vertical-align: top; }
th { background: var(--surface-2); position: sticky; top: 0; }
td.repo { font-weight: 600; white-space: nowrap; }
.badge {
  display: inline-block; border-radius: 5px; padding: 1px 7px; font-size: 12.5px;
  font-weight: 600; margin-bottom: 3px;
}
.b-pass  { background: var(--good-bg); color: var(--good); }
.b-cust  { background: var(--warn-bg); color: #8a6100; }
@media (prefers-color-scheme: dark) { .b-cust { color: var(--warn); } }
.b-fail  { background: var(--crit-bg); color: var(--crit); }
.ident { color: var(--text-2); font-size: 12.5px; }
.ident b { color: inherit; font-weight: 600; }
.err { color: var(--crit); font-size: 12px; word-break: break-word; }
.chg { font-size: 12.5px; }
.chg .from { color: var(--text-2); text-decoration: line-through; }
.chg .to { font-weight: 600; }
.arrow { color: var(--text-2); }
.hidden { display: none; }
.count { color: var(--text-2); font-size: 12.5px; margin-left: auto; }
</style></head><body>
<h1>facts probe report — __SM__</h1>
<div class="sub">__SUBTITLE__</div>
<div class="tiles" id="tiles"></div>
<div class="tabs">
  <button id="tab-matrix" class="on" onclick="show('matrix')">matrix (__SM__)</button>
  <button id="tab-diff" onclick="show('diff')">diff vs __DIFFSM__ (<span id="diffn"></span>)</button>
</div>
<div class="filters" id="matrix-filters">
  <input id="q" placeholder="filter checkpoint…" oninput="render()">
  <select id="fverdict" onchange="render()">
    <option value="">all verdicts</option><option>pass</option>
    <option>pass+custom</option><option>fail</option>
  </select>
  <select id="ffw" onchange="render()"><option value="">all frameworks</option></select>
  <span class="count" id="nrows"></span>
</div>
<div id="matrix"></div>
<div id="diff" class="hidden"></div>
<script>
const DATA = __DATA__;
const DIFF = __DIFF__;
const FWS = Object.keys(DATA);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const badge = v => v === "pass" ? '<span class="badge b-pass">✓ pass</span>'
  : v === "pass+custom" ? '<span class="badge b-cust">✓ pass+custom</span>'
  : '<span class="badge b-fail">✗ fail</span>';
function ident(c) {
  const bits = [];
  if (c.attention) bits.push("<b>attn</b> " + esc(c.attention));
  if (c.moe) bits.push("<b>moe</b> " + esc(c.moe));
  if (c.kv_allocated) bits.push("<b>kv</b> " + esc(c.kv_allocated));
  if (c.extra_args) bits.push("<b>extra</b> <code>" + esc(c.extra_args) + "</code>");
  let h = bits.length ? '<div class="ident">' + bits.join(" · ") + "</div>" : "";
  if (c.cause) h += '<div class="err">' + esc(c.cause) + (c.error ? " — " + esc(c.error) : "") + "</div>";
  return h;
}
function tiles() {
  document.getElementById("tiles").innerHTML = FWS.map(fw => {
    const m = DATA[fw]._meta, s = m.summary || {};
    const total = (s.pass||0)+(s["pass+custom"]||0)+(s.fail||0);
    return `<div class="tile"><div class="fw">${esc(fw)} ${esc(m.version||"")}</div>
      <div class="n">${(s.pass||0)+(s["pass+custom"]||0)}<span style="font-size:14px;color:var(--text-2)">/${total} run</span></div>
      <div class="row">✓ ${s.pass||0} pass · ✓ ${s["pass+custom"]||0} custom · ✗ ${s.fail||0} fail</div></div>`;
  }).join("");
}
function render() {
  const q = document.getElementById("q").value.toLowerCase();
  const fv = document.getElementById("fverdict").value;
  const ff = document.getElementById("ffw").value;
  const fws = ff ? [ff] : FWS;
  const repos = [...new Set(fws.flatMap(fw => Object.keys(DATA[fw].results)))].sort();
  let n = 0;
  let h = "<table><tr><th>checkpoint</th>" + fws.map(f => "<th>" + esc(f) + "</th>").join("") + "</tr>";
  for (const r of repos) {
    if (q && !r.toLowerCase().includes(q)) continue;
    const cells = fws.map(fw => DATA[fw].results[r]);
    if (fv && !cells.some(c => c && c.verdict === fv)) continue;
    n++;
    h += '<tr><td class="repo">' + esc(r) + "</td>" + cells.map(c =>
      "<td>" + (c ? badge(c.verdict) + ident(c) : '<span class="ident">—</span>') + "</td>").join("") + "</tr>";
  }
  document.getElementById("matrix").innerHTML = h + "</table>";
  document.getElementById("nrows").textContent = n + " checkpoints";
}
function renderDiff() {
  document.getElementById("diffn").textContent = DIFF.length;
  if (!DIFF.length) {
    document.getElementById("diff").innerHTML = '<p class="ident">no identity changes vs __DIFFSM__.</p>';
    return;
  }
  let h = "<table><tr><th>checkpoint</th><th>framework</th><th>field</th><th>__DIFFSM__ → __SM__</th></tr>";
  for (const d of DIFF)
    for (const [f, [a, b]] of Object.entries(d.changes))
      h += `<tr><td class="repo">${esc(d.repo)}</td><td>${esc(d.framework)}</td>
        <td>${esc(f)}</td><td class="chg"><span class="from">${esc(a ?? "∅")}</span>
        <span class="arrow">→</span> <span class="to">${esc(b ?? "∅")}</span></td></tr>`;
  document.getElementById("diff").innerHTML = h + "</table>";
}
function show(which) {
  for (const t of ["matrix", "diff"]) {
    document.getElementById(t).classList.toggle("hidden", t !== which);
    document.getElementById("tab-" + t).classList.toggle("on", t === which);
  }
  document.getElementById("matrix-filters").classList.toggle("hidden", which !== "matrix");
}
const sel = document.getElementById("ffw");
for (const f of FWS) sel.insertAdjacentHTML("beforeend", `<option>${esc(f)}</option>`);
tiles(); render(); renderDiff();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm", required=True, help="results subdir to report (e.g. sm100)")
    ap.add_argument("--diff-sm", help="results subdir to diff against (e.g. sm90)")
    ap.add_argument("--out", type=Path, help="default results/<sm>/report.html")
    args = ap.parse_args()

    cur = load_sm(args.sm)
    if not cur:
        raise SystemExit(f"no result yamls under results/{args.sm}")
    prev = load_sm(args.diff_sm) if args.diff_sm else {}
    diff = build_diff(cur, prev) if prev else []

    metas = [d["_meta"] for d in cur.values()]
    subtitle = " · ".join(f"{m['framework']} {m.get('version')}" for m in metas)
    subtitle += f" · platform {metas[0].get('platform')} · generated {date.today().isoformat()}"

    out = args.out or ROOT / "results" / args.sm / "report.html"
    page = (HTML_TMPL
            .replace("__DATA__", json.dumps(cur, ensure_ascii=False))
            .replace("__DIFF__", json.dumps(diff, ensure_ascii=False))
            .replace("__SUBTITLE__", html.escape(subtitle))
            .replace("__SM__", args.sm)
            .replace("__DIFFSM__", args.diff_sm or "n/a"))
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(diff)} diff rows)")


if __name__ == "__main__":
    main()
