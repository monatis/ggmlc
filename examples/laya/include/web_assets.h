#pragma once

#include <string>

namespace laya {

inline std::string get_index_html() {
    return R"rawliteral(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Laya — System 1 Decision Studio</title>
<style>
:root {
  --bg:#070b14; --card:#101827; --line:#1e293b; --text:#f1f5f9; --muted:#94a3b8;
  --accent:#22d3ee; --accent2:#a78bfa; --ok:#34d399; --warn:#fbbf24; --bad:#f87171;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}
.wrap{max-width:1280px;margin:0 auto;padding:22px 22px 48px}
header{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-end;
  padding-bottom:16px;margin-bottom:20px;border-bottom:1px solid var(--line)}
h1{font-size:22px;letter-spacing:-.4px}
h1 span{background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;color:transparent}
.sub{color:var(--muted);font-size:13px;margin-top:4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:10px 0 6px}
select,textarea,input,button{font:inherit;color:var(--text)}
select,textarea,input{width:100%;background:#0b1220;border:1px solid var(--line);border-radius:10px;padding:10px 12px}
textarea{min-height:140px;resize:vertical;font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
button{border:0;border-radius:10px;padding:10px 16px;cursor:pointer;font-weight:600}
.primary{background:linear-gradient(135deg,#0891b2,#7c3aed);color:#fff}
.ghost{background:#0b1220;border:1px solid var(--line);color:var(--text)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 4px}
.chip{font-size:12px;padding:5px 10px;border-radius:999px;border:1px solid var(--line);background:#0b1220;cursor:pointer;color:var(--muted)}
.chip.on{border-color:var(--accent);color:var(--accent)}
.meta{color:var(--muted);font-size:12px;margin-top:8px}
.ans{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.ans h3{font-size:13px;margin-bottom:8px}
.ans h3 em{font-style:normal;color:var(--accent);font-weight:700}
.bar{display:flex;align-items:center;gap:8px;margin:4px 0}
.track{flex:1;height:8px;background:#0b1220;border-radius:99px;overflow:hidden}
.fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.pct{width:52px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px;color:var(--muted)}
.lab{width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;margin-left:6px}
.b-choice{background:#164e63;color:#a5f3fc}
.b-score{background:#3b2d14;color:#fde68a}
.b-noul{background:#3b1d4a;color:#e9d5ff}
.status{min-height:18px;color:var(--muted);font-size:12px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1><span>Laya</span> System 1 Decision Studio</h1>
    <p class="sub">Non-autoregressive typed decisions — choice, score, noul — compiled with ggmlc. No token generation.</p>
  </div>
  <div class="status" id="health">connecting…</div>
</header>
<div class="grid">
  <div class="card">
    <label>Preset</label>
    <div class="chips" id="chips"></div>
    <p class="meta" id="blurb"></p>
    <label>State (JSON object or raw text)</label>
    <textarea id="state"></textarea>
    <label>Questions</label>
    <textarea id="questions" style="min-height:220px"></textarea>
    <div class="row">
      <button class="primary" id="go">Decide</button>
      <button class="ghost" id="reset">Reset preset</button>
    </div>
  </div>
  <div class="card" id="out">
    <p class="meta">Run a preset to see calibrated probabilities. Typical use: guard an LLM, route a ticket, score an invoice — one parallel pass per question.</p>
  </div>
</div>
</div>
<script>
let PRESETS = [];
let current = null;
async function boot(){
  try {
    const h = await fetch('/api/health').then(r=>r.json());
    document.getElementById('health').textContent = (h.model||'laya') + ' · ' + (h.device||'') + ' · ready';
    PRESETS = await fetch('/api/presets').then(r=>r.json());
    const chips = document.getElementById('chips');
    PRESETS.forEach((p,i)=>{
      const b = document.createElement('button');
      b.className = 'chip'+(i===0?' on':'');
      b.textContent = p.name;
      b.onclick = ()=>select(p.name);
      chips.appendChild(b);
    });
    if (PRESETS[0]) select(PRESETS[0].name);
  } catch(e) {
    document.getElementById('health').textContent = 'API unreachable';
  }
}
function select(name){
  current = PRESETS.find(p=>p.name===name) || current;
  document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on', c.textContent===name));
  if (!current) return;
  document.getElementById('blurb').textContent = current.title + ' — ' + current.blurb;
  document.getElementById('state').value = JSON.stringify(current.state, null, 2);
  document.getElementById('questions').value = JSON.stringify(current.questions, null, 2);
}
function parseMaybe(text){
  const t = text.trim();
  if (!t) return {};
  try { return JSON.parse(t); } catch { return t; }
}
function bar(p){
  const pct = Math.round(p*1000)/10;
  return `<div class="bar"><span class="lab"></span><div class="track"><div class="fill" style="width:${Math.max(1,pct)}%"></div></div><span class="pct">${pct.toFixed(1)}%</span></div>`;
}
function render(res){
  const box = document.getElementById('out');
  const u = res.usage||{};
  let html = `<p class="meta">${res.model||'laya'} · ${u.input_tokens||0} tokens · ${(u.latency_ms||0).toFixed(1)} ms · 0 output tokens</p>`;
  const answers = res.answers||{};
  for (const [id,a] of Object.entries(answers)){
    const badge = a.type==='choice'?'b-choice':a.type==='score'?'b-score':'b-noul';
    let head = id;
    if (a.type==='choice') head += ' → ' + (a.choice||'');
    if (a.type==='score') head += ' → ' + Number(a.score??0).toFixed(4);
    if (a.type==='noul') head += ' → P(true)=' + Number(a.noul??0).toFixed(4);
    html += `<div class="ans"><h3><em>${head}</em><span class="badge ${badge}">${a.type}</span> <span class="meta">conf ${Number(a.confidence).toFixed(4)} · act ${Number(a.action?.act_probability??0).toFixed(4)}</span></h3>`;
    const probs = a.probabilities||{};
    if (a.type==='noul' && !Object.keys(probs).length){
      html += barRow('false', 1-(a.noul||0)) + barRow('true', a.noul||0);
    } else {
      for (const [k,v] of Object.entries(probs)){
        let lab = k;
        if (a.legend && a.legend[k]) lab = k + ': ' + a.legend[k];
        html += barRow(lab, v);
      }
    }
    html += `</div>`;
  }
  box.innerHTML = html;
}
function barRow(lab,p){
  const pct = Math.round((p||0)*1000)/10;
  return `<div class="bar"><span class="lab" title="${lab}">${lab}</span><div class="track"><div class="fill" style="width:${Math.max(1,pct)}%"></div></div><span class="pct">${pct.toFixed(1)}%</span></div>`;
}
document.getElementById('go').onclick = async ()=>{
  const body = {
    state: parseMaybe(document.getElementById('state').value),
    questions: parseMaybe(document.getElementById('questions').value)
  };
  const out = document.getElementById('out');
  out.innerHTML = '<p class="meta">scoring…</p>';
  try {
    const res = await fetch('/api/decide', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const js = await res.json();
    if (js.error) { out.innerHTML = '<p class="meta">'+js.error+'</p>'; return; }
    render(js);
  } catch(e) {
    out.innerHTML = '<p class="meta">request failed</p>';
  }
};
document.getElementById('reset').onclick = ()=>{ if(current) select(current.name); };
boot();
</script>
</body>
</html>
)rawliteral";
}

}  // namespace laya
