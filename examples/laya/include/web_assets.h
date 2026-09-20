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
@media(max-width:960px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:10px 0 6px}
select,textarea,input,button{font:inherit;color:var(--text)}
select,textarea,input{width:100%;background:#0b1220;border:1px solid var(--line);border-radius:10px;padding:10px 12px}
textarea{min-height:120px;resize:vertical;font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
button{border:0;border-radius:10px;padding:10px 16px;cursor:pointer;font-weight:600}
.primary{background:linear-gradient(135deg,#0891b2,#7c3aed);color:#fff}
.ghost{background:#0b1220;border:1px solid var(--line);color:var(--text)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 4px}
.chip{font-size:12px;padding:5px 10px;border-radius:999px;border:1px solid var(--line);background:#0b1220;cursor:pointer;color:var(--muted)}
.chip.on{border-color:var(--accent);color:var(--accent)}
.tabs{display:flex;gap:6px;margin:8px 0}
.tab{font-size:12px;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:#0b1220;cursor:pointer;color:var(--muted)}
.tab.on{border-color:var(--accent);color:var(--accent)}
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
.qcard{border:1px solid var(--line);border-radius:12px;padding:10px;margin:8px 0;background:#0b1220}
.qhead{display:grid;grid-template-columns:1fr 140px auto;gap:8px;align-items:center}
.optrow{display:grid;grid-template-columns:1fr 1fr auto;gap:6px;margin-top:6px}
.tiny{padding:6px 10px;font-size:12px}
.toast{color:var(--ok);font-size:12px;margin-top:6px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1><span>Laya</span> System 1 Decision Studio</h1>
    <p class="sub">Typed decisions in one pass — build questions as a form, or paste a Jev/Laya schema.</p>
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
    <div class="tabs">
      <button class="tab on" id="tab-form" type="button">Question builder</button>
      <button class="tab" id="tab-json" type="button">Raw JSON schema</button>
    </div>
    <div id="form-pane">
      <div id="qlist"></div>
      <div class="row">
        <button class="ghost tiny" id="add-q" type="button">+ Add question</button>
      </div>
    </div>
    <div id="json-pane" style="display:none">
      <label>Questions (Laya / Jev schema)</label>
      <textarea id="questions" style="min-height:220px"></textarea>
    </div>
    <div class="row">
      <button class="primary" id="go" type="button">Decide</button>
      <button class="ghost" id="copy-schema" type="button">Copy Jev schema</button>
      <button class="ghost" id="reset" type="button">Reset preset</button>
    </div>
    <p class="toast" id="copied"></p>
  </div>
  <div class="card" id="out">
    <p class="meta">Build questions on the left (id, type, instructions, choices). Submit to score — or copy the generated schema into another Jev/Laya client.</p>
  </div>
</div>
</div>
<script>
let PRESETS = [];
let current = null;
let mode = 'form';
let questionsForm = [];

function uid(){ return 'q' + Math.random().toString(36).slice(2,7); }

async function boot(){
  try {
    const h = await fetch('/api/health').then(r=>r.json());
    const fam = h.families ? h.families.join(', ') : (h.family||'');
    document.getElementById('health').textContent =
      (h.model||'laya') + (fam? ' · '+fam : '') + ' · ' + (h.device||'') + ' · ready';
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
function schemaFromForm(){
  const o = {};
  for (const q of questionsForm){
    const id = (q.id||'').trim() || uid();
    const item = {type: q.type||'choice', instructions: q.instructions||''};
    if (q.type === 'choice'){
      const crit = {};
      for (const c of (q.criteria||[])){
        if (!c.key) continue;
        crit[c.key] = c.desc ? c.desc : null;
      }
      item.criteria = crit;
    } else if (q.type === 'score'){
      item.criteria = (q.criteria||[]).map(c => c.desc || c.key || '');
    } else if (q.criteria && q.criteria.length){
      const crit = {};
      for (const c of q.criteria) if (c.key) crit[c.key] = c.desc||'';
      if (Object.keys(crit).length) item.criteria = crit;
    }
    o[id] = item;
  }
  return o;
}
function formFromSchema(obj){
  questionsForm = [];
  if (!obj || typeof obj !== 'object') return;
  for (const [id, q] of Object.entries(obj)){
    const t = (q && q.type) || 'choice';
    const row = {id, type:t, instructions:(q&&q.instructions)||'', criteria:[]};
    const crit = q && q.criteria;
    if (t === 'score' && Array.isArray(crit)){
      row.criteria = crit.map((d,i)=>({key:String(i), desc: String(d??'')}));
    } else if (crit && typeof crit === 'object' && !Array.isArray(crit)){
      row.criteria = Object.entries(crit).map(([k,v])=>({key:k, desc: v==null?'':String(v)}));
    } else if (t === 'choice'){
      row.criteria = [{key:'option_a', desc:''},{key:'option_b', desc:''}];
    }
    questionsForm.push(row);
  }
}
function renderForm(){
  const box = document.getElementById('qlist');
  box.innerHTML = '';
  questionsForm.forEach((q, qi)=>{
    const card = document.createElement('div');
    card.className = 'qcard';
    const typeHint = q.type==='choice' ? 'label + description per choice'
      : q.type==='score' ? 'one row per score level (text)'
      : 'optional false/true descriptions';
    card.innerHTML = `
      <div class="qhead">
        <input data-k="id" placeholder="question id" value="${esc(q.id||'')}">
        <select data-k="type">
          <option value="choice"${q.type==='choice'?' selected':''}>choice</option>
          <option value="score"${q.type==='score'?' selected':''}>score</option>
          <option value="noul"${q.type==='noul'?' selected':''}>noul (yes/no)</option>
        </select>
        <button class="ghost tiny" data-rm type="button">Remove</button>
      </div>
      <label>Instructions</label>
      <input data-k="instructions" placeholder="What should this question decide?" value="${esc(q.instructions||'')}">
      <label>${typeHint}</label>
      <div data-opts></div>
      <button class="ghost tiny" data-add type="button" style="margin-top:8px">+ Option / level</button>
    `;
    const opts = card.querySelector('[data-opts]');
    (q.criteria||[]).forEach((c, ci)=>{
      const row = document.createElement('div');
      row.className = 'optrow';
      row.innerHTML = `
        <input data-ok="key" placeholder="${q.type==='score'?'level '+ci:'label'}" value="${esc(c.key||'')}">
        <input data-ok="desc" placeholder="description" value="${esc(c.desc||'')}">
        <button class="ghost tiny" data-del type="button">×</button>`;
      row.querySelector('[data-ok=key]').oninput = e=>{ q.criteria[ci].key = e.target.value; syncJson(); };
      row.querySelector('[data-ok=desc]').oninput = e=>{ q.criteria[ci].desc = e.target.value; syncJson(); };
      row.querySelector('[data-del]').onclick = ()=>{ q.criteria.splice(ci,1); renderForm(); syncJson(); };
      opts.appendChild(row);
    });
    card.querySelector('[data-k=id]').oninput = e=>{ q.id = e.target.value; syncJson(); };
    card.querySelector('[data-k=instructions]').oninput = e=>{ q.instructions = e.target.value; syncJson(); };
    card.querySelector('[data-k=type]').onchange = e=>{
      q.type = e.target.value;
      if (q.type==='noul' && (!q.criteria||!q.criteria.length))
        q.criteria = [{key:'false',desc:''},{key:'true',desc:''}];
      renderForm(); syncJson();
    };
    card.querySelector('[data-rm]').onclick = ()=>{ questionsForm.splice(qi,1); renderForm(); syncJson(); };
    card.querySelector('[data-add]').onclick = ()=>{
      q.criteria = q.criteria||[];
      q.criteria.push({key: q.type==='score'? String(q.criteria.length): '', desc:''});
      renderForm(); syncJson();
    };
    box.appendChild(card);
  });
}
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;'); }
function syncJson(){
  document.getElementById('questions').value = JSON.stringify(schemaFromForm(), null, 2);
}
function select(name){
  current = PRESETS.find(p=>p.name===name) || current;
  document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on', c.textContent===name));
  if (!current) return;
  document.getElementById('blurb').textContent = current.title + ' — ' + current.blurb;
  document.getElementById('state').value = JSON.stringify(current.state, null, 2);
  document.getElementById('questions').value = JSON.stringify(current.questions, null, 2);
  formFromSchema(current.questions);
  renderForm();
}
function parseMaybe(text){
  const t = text.trim();
  if (!t) return {};
  try { return JSON.parse(t); } catch { return t; }
}
function barRow(lab,p){
  const pct = Math.round((p||0)*1000)/10;
  return `<div class="bar"><span class="lab" title="${lab}">${lab}</span><div class="track"><div class="fill" style="width:${Math.max(1,pct)}%"></div></div><span class="pct">${pct.toFixed(1)}%</span></div>`;
}
function render(res){
  const box = document.getElementById('out');
  const u = res.usage||{};
  let html = `<p class="meta">${res.model||'laya'}${res.family?' · '+res.family:''}${res.route?' · '+res.route:''} · ${u.input_tokens||0} tokens · ${(u.latency_ms||0).toFixed(1)} ms · 0 output tokens</p>`;
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
function setMode(m){
  mode = m;
  document.getElementById('tab-form').classList.toggle('on', m==='form');
  document.getElementById('tab-json').classList.toggle('on', m==='json');
  document.getElementById('form-pane').style.display = m==='form'?'block':'none';
  document.getElementById('json-pane').style.display = m==='json'?'block':'none';
  if (m==='form'){
    try { formFromSchema(JSON.parse(document.getElementById('questions').value||'{}')); renderForm(); } catch(e){}
  } else {
    syncJson();
  }
}
document.getElementById('tab-form').onclick = ()=>setMode('form');
document.getElementById('tab-json').onclick = ()=>setMode('json');
document.getElementById('add-q').onclick = ()=>{
  questionsForm.push({id:'question_'+ (questionsForm.length+1), type:'choice', instructions:'',
    criteria:[{key:'option_a',desc:''},{key:'option_b',desc:''}]});
  renderForm(); syncJson();
};
document.getElementById('copy-schema').onclick = async ()=>{
  if (mode==='form') syncJson();
  const schema = document.getElementById('questions').value;
  try {
    await navigator.clipboard.writeText(schema);
    document.getElementById('copied').textContent = 'Copied Laya/Jev question schema to clipboard.';
  } catch(e) {
    document.getElementById('copied').textContent = 'Could not copy — select the JSON tab and copy manually.';
  }
  setTimeout(()=>document.getElementById('copied').textContent='', 2500);
};
document.getElementById('go').onclick = async ()=>{
  if (mode==='form') syncJson();
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
