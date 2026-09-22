"""Build experiments/dashboard.html from experiments/summary.json.

Self-contained: data embedded, no network, opens via file://.
Regenerated automatically by scripts/benchmark.py; run manually after
ad-hoc `make exp-run` if you want the page refreshed:

    .venv/bin/python scripts/exp_summary.py && .venv/bin/python scripts/build_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARC-AGI-3 실험 벤치마크</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb; --surface-2: #f1f0ee; --border: #dddcd8;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #74736e;
    --accent: #2a78d6; --grid: #e6e5e1;
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
    --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1:#1a1a19; --surface-2:#242422; --border:#3a3936;
      --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8d8c85;
      --accent:#3987e5; --grid:#33322f;
      --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
      --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#242422; --border:#3a3936;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8d8c85;
    --accent:#3987e5; --grid:#33322f;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface-1); color: var(--text-primary);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 24px 20px 60px; }
  header { display: flex; justify-content: space-between; align-items: baseline;
           gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  h1 { font-size: 20px; margin: 0; }
  h2 { font-size: 15px; margin: 32px 0 12px; }
  .sub { color: var(--text-secondary); font-size: 13px; }
  button.theme { background: var(--surface-2); color: var(--text-secondary);
    border: 1px solid var(--border); border-radius: 8px; padding: 4px 12px;
    cursor: pointer; font-size: 13px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 10px; }
  .tile { background: var(--surface-2); border: 1px solid var(--border);
          border-radius: 10px; padding: 12px 14px; }
  .tile .k { font-size: 12px; color: var(--text-secondary); }
  .tile .v { font-size: 22px; font-weight: 650; margin-top: 2px;
             font-variant-numeric: tabular-nums; }
  .tile .d { font-size: 11px; color: var(--text-muted); }
  .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
              margin: 6px 0 14px; }
  select { background: var(--surface-2); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 8px; padding: 5px 10px;
    font-size: 13px; }
  .panel { background: var(--surface-2); border: 1px solid var(--border);
           border-radius: 12px; padding: 16px; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 760px) { .charts { grid-template-columns: 1fr; } }
  .chart-title { font-size: 13px; color: var(--text-secondary); margin: 0 0 8px; }
  svg text { fill: var(--text-secondary); font-size: 11px; }
  svg .val { fill: var(--text-primary); font-weight: 600;
             font-variant-numeric: tabular-nums; }
  svg .gridline { stroke: var(--grid); stroke-width: 1; }
  svg .axis { stroke: var(--border); stroke-width: 1; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px;
            font-size: 12px; color: var(--text-secondary); }
  .legend .sw { display: inline-block; width: 10px; height: 10px;
                border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border);
           white-space: nowrap; }
  th { color: var(--text-secondary); font-weight: 600; font-size: 12px; }
  td.num, th.num { text-align: right; }
  .muted { color: var(--text-muted); }
  .tooltip { position: fixed; pointer-events: none; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 8px; padding: 7px 10px;
    font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.18); z-index: 10;
    display: none; max-width: 260px; }
  .tooltip b { font-variant-numeric: tabular-nums; }
  .empty { color: var(--text-muted); padding: 20px; text-align: center; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>ARC-AGI-3 실험 벤치마크</h1>
      <div class="sub" id="generated"></div>
    </div>
    <button class="theme" id="themeBtn">테마</button>
  </header>

  <div class="tiles" id="tiles"></div>

  <h2>벤치마크 비교</h2>
  <div class="controls">
    <label class="sub" for="benchSel">벤치마크:</label>
    <select id="benchSel"></select>
    <span class="sub" id="benchMeta"></span>
  </div>
  <div class="panel">
    <div class="charts">
      <div><p class="chart-title">종합 점수 (스코어카드, 0–100)</p><div id="scoreChart"></div></div>
      <div><p class="chart-title">완료 레벨 수 (전 게임 합)</p><div id="levelChart"></div></div>
    </div>
    <h2 style="margin-top:20px">게임별 상세</h2>
    <div class="tablewrap"><table id="gameTable"></table></div>
  </div>

  <h2>점수 추이 (실행 순서별)</h2>
  <div class="panel">
    <div id="trendChart"></div>
    <div class="legend" id="trendLegend"></div>
  </div>

  <h2>모든 실행 기록</h2>
  <div class="panel tablewrap"><table id="runTable"></table></div>
</div>
<div class="tooltip" id="tip"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const PALETTE = ['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8'];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const exps = DATA.experiments;
const colorOf = i => css(PALETTE[i % PALETTE.length]);
const fmt = x => (x == null ? '–' : (typeof x === 'number' ? x.toFixed(3) : x));

// ---- theme toggle (data-theme wins over media query) ----
document.getElementById('themeBtn').onclick = () => {
  const r = document.documentElement;
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const cur = r.dataset.theme || (dark ? 'dark' : 'light');
  r.dataset.theme = cur === 'dark' ? 'light' : 'dark';
  render();
};

// ---- tooltip ----
const tip = document.getElementById('tip');
function showTip(e, html) {
  tip.innerHTML = html; tip.style.display = 'block';
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + 'px'; tip.style.top = (e.clientY + 14) + 'px';
}
function hideTip() { tip.style.display = 'none'; }

// ---- benchmark groups ----
function benchGroups() {
  const groups = {};
  exps.forEach(e => (e.runs || []).forEach(r => {
    if (r.tag && r.tag.startsWith('bench-')) {
      (groups[r.tag] ??= {});
      groups[r.tag][e.name] = r;   // last run per experiment in this group
    }
  }));
  return Object.entries(groups).sort((a, b) => b[0].localeCompare(a[0]));
}
function latestPerExperiment() {
  const g = {};
  exps.forEach(e => { const rs = e.runs || []; if (rs.length) g[e.name] = rs[rs.length-1]; });
  return g;
}

// ---- SVG helpers ----
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

// Horizontal bar chart: rows = experiments, single accent hue
// (identity lives on the axis labels, so bars share one color).
function barChart(container, rows, valueKey, maxHint, tipHtml) {
  container.innerHTML = '';
  if (!rows.length) { container.innerHTML = '<div class="empty">데이터 없음</div>'; return; }
  const W = 480, ROW = 30, LBL = 150, PAD = 8;
  const H = rows.length * ROW + 24;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%' }, container);
  const maxV = Math.max(maxHint, ...rows.map(r => r[valueKey] || 0));
  const plotW = W - LBL - 60;
  // gridlines at 0/50/100%
  [0, .5, 1].forEach(t => {
    const x = LBL + t * plotW;
    el('line', { x1: x, y1: 4, x2: x, y2: H - 20, class: 'gridline' }, svg);
    const lbl = el('text', { x, y: H - 6, 'text-anchor': 'middle' }, svg);
    lbl.textContent = (maxV * t).toFixed(maxV >= 10 ? 0 : 1);
  });
  rows.forEach((r, i) => {
    const y = i * ROW + 6;
    const v = r[valueKey] || 0;
    const w = Math.max(v / maxV * plotW, 1.5);
    const name = el('text', { x: LBL - PAD, y: y + 15, 'text-anchor': 'end' }, svg);
    name.textContent = r.name.length > 22 ? r.name.slice(0, 21) + '…' : r.name;
    const bar = el('rect', { x: LBL, y, width: w, height: ROW - 12,
      rx: 4, fill: css('--accent') }, svg);
    const val = el('text', { x: LBL + w + 6, y: y + 15, class: 'val' }, svg);
    val.textContent = fmt(v);
    bar.addEventListener('mousemove', e => showTip(e, tipHtml(r)));
    bar.addEventListener('mouseleave', hideTip);
  });
  el('line', { x1: LBL, y1: 4, x2: LBL, y2: H - 20, class: 'axis' }, svg);
}

// Line chart: score over per-experiment run sequence.
function trendChart(container, legendBox) {
  container.innerHTML = ''; legendBox.innerHTML = '';
  const series = exps.map((e, i) => ({
    name: e.name, color: colorOf(i),
    pts: (e.runs || []).map((r, j) => ({ x: j + 1, y: r.aggregate?.score ?? 0, run: r })),
  })).filter(s => s.pts.length);
  if (!series.length) { container.innerHTML = '<div class="empty">데이터 없음</div>'; return; }
  const W = 960, H = 240, L = 46, R = 16, T = 12, B = 30;
  const maxX = Math.max(2, ...series.map(s => s.pts.length));
  const maxY = Math.max(1, ...series.flatMap(s => s.pts.map(p => p.y)));
  const sx = x => L + (x - 1) / (maxX - 1) * (W - L - R);
  const sy = y => T + (1 - y / maxY) * (H - T - B);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%' }, container);
  [0, .5, 1].forEach(t => {
    const y = sy(maxY * t);
    el('line', { x1: L, y1: y, x2: W - R, y2: y, class: 'gridline' }, svg);
    const lbl = el('text', { x: L - 8, y: y + 4, 'text-anchor': 'end' }, svg);
    lbl.textContent = (maxY * t).toFixed(maxY >= 10 ? 0 : 1);
  });
  for (let x = 1; x <= maxX; x++) {
    const lbl = el('text', { x: sx(x), y: H - 10, 'text-anchor': 'middle' }, svg);
    lbl.textContent = x;
  }
  series.forEach(s => {
    if (s.pts.length > 1) {
      const d = s.pts.map((p, i) => (i ? 'L' : 'M') + sx(p.x) + ' ' + sy(p.y)).join(' ');
      el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2,
                   'stroke-linejoin': 'round' }, svg);
    }
    s.pts.forEach(p => {
      const dot = el('circle', { cx: sx(p.x), cy: sy(p.y), r: 4.5, fill: s.color,
        stroke: css('--surface-2'), 'stroke-width': 2 }, svg);
      dot.addEventListener('mousemove', e => showTip(e,
        `<b>${s.name}</b><br>실행 #${p.x} (${p.run.run_id})<br>` +
        `점수 <b>${fmt(p.y)}</b> · 레벨 ${p.run.aggregate?.levels_completed ?? '–'} · ` +
        `액션 ${p.run.aggregate?.actions ?? '–'}` +
        (p.run.tag ? `<br><span class="muted">${p.run.tag}</span>` : '')));
      dot.addEventListener('mouseleave', hideTip);
    });
    legendBox.insertAdjacentHTML('beforeend',
      `<span><span class="sw" style="background:${s.color}"></span>${s.name}</span>`);
  });
}

function renderBench(tag) {
  const group = tag === '__latest__' ? latestPerExperiment()
    : Object.fromEntries(benchGroups().find(([t]) => t === tag)?.[1]
        ? Object.entries(benchGroups().find(([t]) => t === tag)[1]) : []);
  const rows = exps.filter(e => group[e.name]).map(e => {
    const r = group[e.name];
    return { name: e.name, run: r,
      score: r.aggregate?.score ?? 0,
      levels: r.aggregate?.levels_completed ?? 0,
      actions: r.aggregate?.actions ?? 0 };
  });
  document.getElementById('benchMeta').textContent = rows.length
    ? `${rows.length}개 버전 · max_steps=${rows[0].run.config?.max_steps ?? '–'} · 게임 ${rows[0].run.games?.length ?? '–'}개`
    : '';
  const tipHtml = r => `<b>${r.name}</b><br>점수 <b>${fmt(r.score)}</b> · ` +
    `레벨 ${r.levels} · 액션 ${r.actions}<br><span class="muted">${r.run.run_id}</span>`;
  barChart(document.getElementById('scoreChart'), rows, 'score', 1, tipHtml);
  barChart(document.getElementById('levelChart'), rows, 'levels', 1, tipHtml);

  // per-game table
  const games = [...new Set(rows.flatMap(r => (r.run.games || []).map(g => g.game_id)))];
  const t = document.getElementById('gameTable');
  t.innerHTML = '<tr><th>experiment</th>' +
    games.map(g => `<th class="num">${g}</th>`).join('') + '</tr>' +
    rows.map(r => '<tr><td>' + r.name + '</td>' + games.map(gid => {
      const g = (r.run.games || []).find(x => x.game_id === gid);
      if (!g) return '<td class="num muted">–</td>';
      return `<td class="num">${g.levels_completed ?? 0}/${g.win_levels ?? '?'}레벨 · ` +
             `${g.actions ?? '–'}액션 · ${fmt(g.score)}</td>`;
    }).join('') + '</tr>').join('');
}

function render() {
  document.getElementById('generated').textContent =
    `생성: ${DATA.generated_at} · 소스: experiments/summary.json`;

  // tiles
  const allRuns = exps.flatMap(e => e.runs || []);
  const best = allRuns.length ? Math.max(...allRuns.map(r => r.aggregate?.score ?? 0)) : null;
  const bestLv = allRuns.length ? Math.max(...allRuns.map(r => r.aggregate?.levels_completed ?? 0)) : null;
  const benches = benchGroups();
  document.getElementById('tiles').innerHTML = [
    ['실험 버전', exps.length, ''],
    ['총 실행 수', allRuns.length, ''],
    ['최고 점수', best == null ? '–' : best.toFixed(3), '전체 실행 기준'],
    ['최다 완료 레벨', bestLv ?? '–', '전체 실행 기준'],
    ['벤치마크 횟수', benches.length, benches[0]?.[0] ?? ''],
  ].map(([k, v, d]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`
  ).join('');

  // bench selector
  const sel = document.getElementById('benchSel');
  const keep = sel.value;
  sel.innerHTML = benches.map(([t]) => `<option value="${t}">${t}</option>`).join('') +
    '<option value="__latest__">버전별 최근 실행 (태그 무관)</option>';
  sel.value = keep && [...sel.options].some(o => o.value === keep)
    ? keep : (benches[0]?.[0] ?? '__latest__');
  sel.onchange = () => renderBench(sel.value);
  renderBench(sel.value);

  trendChart(document.getElementById('trendChart'), document.getElementById('trendLegend'));

  // all-runs table
  const rt = document.getElementById('runTable');
  rt.innerHTML = '<tr><th>experiment</th><th>run</th><th>tag</th>' +
    '<th class="num">score</th><th class="num">levels</th><th class="num">actions</th>' +
    '<th>games</th><th>git</th></tr>' +
    exps.flatMap((e) => (e.runs || []).slice().reverse().map(r =>
      `<tr><td>${e.name}</td><td>${r.run_id}</td>` +
      `<td>${r.tag ?? '<span class="muted">–</span>'}</td>` +
      `<td class="num">${fmt(r.aggregate?.score)}</td>` +
      `<td class="num">${r.aggregate?.levels_completed ?? '–'}</td>` +
      `<td class="num">${r.aggregate?.actions ?? '–'}</td>` +
      `<td>${(r.games || []).length}</td>` +
      `<td class="muted">${r.git?.commit ?? '–'}${r.git?.dirty ? '*' : ''}</td></tr>`
    )).join('');
}
render();
</script>
</body>
</html>
"""


def main() -> None:
    summary_path = EXPERIMENTS / "summary.json"
    if not summary_path.exists():
        raise SystemExit("experiments/summary.json not found. Run scripts/exp_summary.py first.")
    summary = json.loads(summary_path.read_text())
    from datetime import datetime, timezone
    summary["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    out = EXPERIMENTS / "dashboard.html"
    out.write_text(TEMPLATE.replace("__DATA__", data))
    print(f"[build_dashboard] Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
