var CATEGORIES = ['factual_knowledge','mathematical_reasoning','sentiment_classification','text_summarisation','named_entity_recognition','code_debugging','logical_deductive_reasoning','code_generation'];
var CAT_COLORS = ['#58a6ff','#d29922','#3fb950','#bc8cff','#f85149','#7ee787','#ffa657','#79c0ff'];
var CAT_SHORT = {factual_knowledge:'factual',mathematical_reasoning:'math',sentiment_classification:'sentiment',text_summarisation:'summary',named_entity_recognition:'ner',code_debugging:'debug',logical_deductive_reasoning:'logic',code_generation:'codegen'};
var runState = { status: 'idle', events: [], results: [], metrics: {}, tasks: [], categories: {}, resultsByTask: {}, judgeResults: {}, batches: [] };
var routingConfig = {};
var sseSource = null;
var activeTab = 'loop';
function escapeHtml(s) { var d = document.createElement('div'); d.textContent = s||''; return d.innerHTML; }
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(function(t) { t.classList.toggle('active', t.dataset.tab === tab); });
  ['loop','tasks','trace'].forEach(function(t) { var el = document.getElementById('tab-'+t); if (el) el.style.display = t === tab ? 'block' : 'none'; });
  if (tab === 'tasks') renderTasks();
  if (tab === 'trace') renderTrace();
}
async function init() { await loadTasks(); await loadModels(); await loadRouting(); }
async function loadTasks() {
  var res = await fetch('/api/tasks'); var sets = await res.json();
  var sel = document.getElementById('tasksFile');
  sel.innerHTML = sets.map(function(s) { return '<option value="'+s.file+'">'+s.name+' ('+s.count+' tasks'+(s.has_expected?', scored':'')+')</option>'; }).join('');
}
async function loadModels() {
  var res = await fetch('/api/models'); var data = await res.json();
  var sel = document.getElementById('model');
  sel.innerHTML = data.models.map(function(m) { return '<option value="'+m+'"'+(m===data.default?' selected':'')+'>'+m.split('/').pop()+'</option>'; }).join('');
  var jsel = document.getElementById('judgeModel');
  jsel.innerHTML = '<option value="">(same as agent)</option>' + data.models.map(function(m) { return '<option value="'+m+'">'+m.split('/').pop()+'</option>'; }).join('');
}
async function loadRouting() { var res = await fetch('/api/routing'); routingConfig = await res.json(); renderRoutingSidebar(); }
function renderRoutingSidebar() {
  var div = document.getElementById('routingSidebar'); var cats = routingConfig.categories || CATEGORIES;
  var efforts = routingConfig.effort_levels || ['low','medium','high','xhigh']; var models = routingConfig.models || [];
  var etMap = routingConfig.effort_tier_map || {}; var mrMap = routingConfig.model_route_map || {};
  var html = '';
  cats.forEach(function(cat) {
    var idx = cats.indexOf(cat); var color = idx < 8 ? CAT_COLORS[idx] : '#8b949e';
    var short = CAT_SHORT[cat] || cat.replace(/_/g,' '); var effort = etMap[cat] || ''; var model = mrMap[cat] || '';
    html += '<div class="route-card"><span class="route-cat" style="color:'+color+'">'+short+'</span><div class="route-controls">';
    html += '<select id="effort_'+cat+'" class="effort-sel" onchange="saveRouting()"><option value="">low</option>';
    efforts.forEach(function(e) { html += '<option value="'+e+'"'+(effort===e?' selected':'')+'>'+e+'</option>'; });
    html += '</select><select id="model_'+cat+'" class="model-sel" onchange="saveRouting()"><option value="">(default)</option>';
    models.forEach(function(m) { html += '<option value="'+m+'"'+(model===m?' selected':'')+'>'+m.split('/').pop()+'</option>'; });
    html += '</select></div></div>';
  });
  html += '<div style="margin-top:6px"><button class="btn btn-secondary" style="width:auto;font-size:11px;padding:3px 8px" onclick="resetRouting()">Reset</button></div>';
  div.innerHTML = html;
}
async function saveRouting() {
  var cats = routingConfig.categories || CATEGORIES; var etMap = {}, mrMap = {};
  cats.forEach(function(cat) { var e = document.getElementById('effort_'+cat); if (e && e.value) etMap[cat] = e.value; var m = document.getElementById('model_'+cat); if (m && m.value) mrMap[cat] = m.value; });
  await fetch('/api/routing', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({effort_tier_map: etMap, model_route_map: mrMap}) });
  routingConfig.effort_tier_map = etMap; routingConfig.model_route_map = mrMap;
}
function resetRouting() {
  routingConfig.effort_tier_map = {'logical_deductive_reasoning':'xhigh','code_debugging':'xhigh','code_generation':'xhigh'};
  var models = routingConfig.models || []; var hasKimi = models.indexOf('accounts/fireworks/models/kimi-k2p7-code') >= 0;
  routingConfig.model_route_map = hasKimi ? {'code_debugging':'accounts/fireworks/models/kimi-k2p7-code','code_generation':'accounts/fireworks/models/kimi-k2p7-code'} : {};
  saveRouting(); renderRoutingSidebar();
}
async function classifyPrompt() {
  var input = document.getElementById('classifyInput'); var prompt = input.value.trim(); if (!prompt) return;
  var res = await fetch('/api/classify', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({prompt: prompt}) });
  var data = await res.json(); var div = document.getElementById('classifyResults');
  if (data.error) { div.innerHTML = '<p style="color:#f85149;font-size:11px">'+data.error+'</p>'; return; }
  if (!data.predictions || data.predictions.length === 0) { div.innerHTML = '<p style="color:#8b949e;font-size:11px">No categories above threshold.</p>'; return; }
  var html = '';
  data.predictions.forEach(function(p) { var idx = CATEGORIES.indexOf(p.label); var color = idx >= 0 ? CAT_COLORS[idx] : '#8b949e'; var pct = Math.round(p.score * 100);
    html += '<div style="margin:3px 0"><span style="color:'+color+';font-size:10px">'+p.label+'</span> ('+pct+'%)<div class="pred-bar"><div class="pred-bar-fill" style="width:'+pct+'%;background:'+color+'"></div></div></div>'; });
  div.innerHTML = html;
}
async function startRun() {
  var btn = document.getElementById('runBtn'); btn.disabled = true; btn.textContent = 'Running...';
  runState = { status: 'running', events: [], results: [], metrics: {}, tasks: [], categories: {}, resultsByTask: {}, judgeResults: {}, batches: [] };
  renderMetrics(); renderLoop();
  var req = {
    tasks_file: document.getElementById('tasksFile').value, model: document.getElementById('model').value,
    batch_size: parseInt(document.getElementById('batchSize').value)||20, batch_tokens: parseInt(document.getElementById('batchTokens').value)||12000,
    max_concurrency: parseInt(document.getElementById('maxConcurrency').value)||3, reasoning_effort: document.getElementById('effort').value,
    disable_hints: document.getElementById('disableHints').checked, enable_classifier: document.getElementById('enableClassifier').checked,
    enable_judge: document.getElementById('enableJudge').checked, judge_model: document.getElementById('judgeModel').value, judge_effort: document.getElementById('judgeEffort').value,
  };
  try {
    var res = await fetch('/api/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(req) });
    var data = await res.json();
    if (data.error) { alert(data.error); btn.disabled = false; btn.textContent = 'Run Agent'; return; }
    connectSSE(data.run_id);
  } catch(e) { alert('Failed to start run: '+e.message); btn.disabled = false; btn.textContent = 'Run Agent'; }
}
function connectSSE(runID) {
  if (sseSource) sseSource.close();
  sseSource = new EventSource('/api/run/'+runID+'/events');
  sseSource.onmessage = function(e) {
    var ev = JSON.parse(e.data);
    if (ev.type === 'final') {
      runState.status = ev.status; runState.results = ev.results || []; runState.metrics = ev.metrics || {}; runState.judgeResults = ev.judge_results || {};
      runState.results.forEach(function(r) { runState.resultsByTask[r.task_id] = r; });
      renderMetrics(); renderLoop(); renderTasks(); updateStatusBadge();
      var btn = document.getElementById('runBtn'); btn.disabled = false; btn.textContent = 'Run Agent';
      sseSource.close();
    } else { runState.events.push(ev); handleEvent(ev); }
  };
  sseSource.onerror = function() { var btn = document.getElementById('runBtn'); btn.disabled = false; btn.textContent = 'Run Agent'; };
}
function handleEvent(ev) {
  switch(ev.type) {
    case 'classify_done': runState.categories = ev.data || {}; break;
    case 'batch_plan': runState.batches = (ev.data && ev.data.batches) || []; break;
    case 'batch_start':
      var bi = ev.data.index; if (!runState.batches[bi]) runState.batches[bi] = {};
      runState.batches[bi].status = 'running'; runState.batches[bi].turns = []; runState.batches[bi].task_ids = ev.data.task_ids; runState.batches[bi].effort = ev.data.effort; break;
    case 'call':
      var bi2 = batchOfEvent(ev); if (bi2 >= 0 && runState.batches[bi2]) { runState.batches[bi2].turns.push({ type: 'call', data: ev.data }); runState.batches[bi2].status = 'running'; } break;
    case 'tool':
      var bi3 = batchOfEvent(ev); if (bi3 >= 0 && runState.batches[bi3] && runState.batches[bi3].turns.length > 0) { var lt = runState.batches[bi3].turns[runState.batches[bi3].turns.length-1]; if (!lt.tools) lt.tools = []; lt.tools.push(ev.data); } break;
    case 'submit':
      var bi4 = batchOfEvent(ev); if (bi4 >= 0 && runState.batches[bi4]) { runState.batches[bi4].submit = ev.data; runState.batches[bi4].status = 'done'; if (ev.data.answers) { for (var k in ev.data.answers) { runState.resultsByTask[k] = { answer: ev.data.answers[k] }; } } } break;
    case 'batch_end': var bi5 = ev.data.index; if (runState.batches[bi5]) { runState.batches[bi5].status = 'done'; runState.batches[bi5].summary = ev.data; } break;
    case 'result': if (ev.data.task_id) { runState.resultsByTask[ev.data.task_id] = { answer: ev.data.answer, categories: ev.data.categories }; } break;
    case 'done': runState.metrics = ev.data || {}; break;
    case 'judge_progress': runState.judgeResults = ev.data.judge_results || {}; break;
  }
  renderMetrics(); renderLoop(); renderTasks(); updateStatusBadge();
}
function batchOfEvent(ev) {
  if (ev.data && ev.data.task_ids) { for (var i = 0; i < runState.batches.length; i++) { var bt = runState.batches[i]; if (bt && bt.task_ids) { var match = ev.data.task_ids.every(function(t) { return bt.task_ids.indexOf(t) >= 0; }); if (match) return i; } } }
  return -1;
}
function updateStatusBadge() { var badge = document.getElementById('runStatusBadge'); badge.innerHTML = '<span class="status-dot '+runState.status+'"></span>' + runState.status; }
function renderMetrics() {
  var m = runState.metrics; if (!m || Object.keys(m).length === 0) { document.getElementById('metricsRow').innerHTML = ''; return; }
  var html = ''; html += metricCard('Calls', m.calls||0); html += metricCard('Tokens', m.total_tokens||0, 'green'); html += metricCard('Cached', m.cached_tokens||0, 'purple');
  html += metricCard('Reasoning', m.reasoning_tokens||0, 'orange'); html += metricCard('Tools', m.tool_runs||0, 'purple'); html += metricCard('Batches', m.batch_count||0);
  html += metricCard('Fallbacks', m.fallbacks||0, 'orange'); var dur = m.duration_ms ? (m.duration_ms/1000).toFixed(1)+'s' : '-'; html += metricCard('Duration', dur);
  document.getElementById('metricsRow').innerHTML = html;
}
function metricCard(label, value, color) { return '<div class="metric"><div class="label">'+label+'</div><div class="value'+(color?' '+color:'')+'">'+value+'</div></div>'; }
function renderLoop() {
  var div = document.getElementById('loopViz');
  if (!runState.batches || runState.batches.length === 0) { if (runState.events.length === 0) { div.innerHTML = '<div class="empty">Start a run to see the agent loop.</div>'; return; } }
  var html = '';
  runState.batches.forEach(function(batch, i) { html += renderBatchLoop(i, batch); });
  if (runState.batches.length === 0 && runState.events.length > 0) { html = '<div class="empty">' + runState.events.length + ' events received...</div>'; }
  div.innerHTML = html;
}
function renderBatchLoop(batchIdx, batch) {
  var html = '<div style="margin-bottom:16px">';
  html += '<div style="font-size:12px;color:#d29922;font-weight:600;margin-bottom:6px">Batch '+batchIdx;
  if (batch.task_ids) html += ' <span style="color:#8b949e;font-weight:400">('+batch.task_ids.join(', ')+')</span>';
  if (batch.effort) html += ' <span style="background:#30363d;padding:1px 6px;border-radius:8px;font-size:10px;color:#d29922">'+batch.effort+'</span>';
  if (batch.status) html += ' <span class="status-dot '+batch.status+'"></span>';
  html += '</div>';
  if (batch.turns) {
    batch.turns.forEach(function(turn, ti) {
      var d = turn.data || {};
      html += '<div class="turn-group expanded" onclick="this.classList.toggle(\'expanded\')">';
      html += '<div class="turn-header"><span class="turn-num">Turn '+ti+'</span>';
      html += '<span class="turn-type">LLM call';
      if (d.effort) html += ' ('+d.effort+')';
      html += '</span>';
      if (d.total_tokens) html += '<span class="turn-tokens">'+d.total_tokens+' tok</span>';
      html += '<span class="turn-status done">done</span></div>';
      html += '<div class="turn-body">';
      if (d.output_preview) html += '<div class="llm-output">'+escapeHtml(d.output_preview)+'</div>';
      if (d.error) html += '<div style="color:#f85149;font-size:11px">Error: '+escapeHtml(d.error)+'</div>';
      if (turn.tools) {
        turn.tools.forEach(function(tool, toi) {
          html += '<div class="code-block"><span class="code-label">micropy #'+(toi+1)+'</span>'+escapeHtml(tool.code||'')+'</div>';
          if (tool.stdout) html += '<div class="observation"><div class="obs-label">stdout</div><div class="obs-stdout">'+escapeHtml(tool.stdout.substring(0,1000))+'</div></div>';
          if (tool.json) html += '<div class="observation"><div class="obs-label">json</div><div class="obs-stdout">'+escapeHtml(JSON.stringify(tool.json).substring(0,500))+'</div></div>';
          if (tool.error) html += '<div class="observation"><div class="obs-label">error</div><div class="obs-error">'+escapeHtml(tool.error)+'</div></div>';
        });
      }
      html += '</div></div>';
    });
  }
  if (batch.submit) {
    var s = batch.submit; var answers = s.answers || {};
    html += '<div class="submit-event"><div class="submit-label">submit() called';
    if (s.via) html += ' <span style="color:#8b949e;font-size:10px">via '+s.via+'</span>';
    html += '</div><div class="submit-answers">';
    for (var k in answers) {
      html += '<div class="submit-answer-row"><span class="submit-task-id">'+k+'</span><span class="submit-answer-val">'+escapeHtml(answers[k])+'</span></div>';
    }
    html += '</div></div>';
  }
  if (batch.summary) {
    var sm = batch.summary;
    html += '<div style="font-size:11px;color:#8b949e;margin-top:4px">'+sm.calls+' calls, '+sm.tools+' tools';
    if (sm.error) html += ' <span style="color:#f85149">'+escapeHtml(sm.error)+'</span>';
    html += '</div>';
  }
  html += '</div>';
  return html;
}
function renderTasks() {
  var grid = document.getElementById('taskGrid');
  var allTasks = [];
  var seen = {};
  if (runState.batches) {
    runState.batches.forEach(function(b, i) {
      if (b.task_ids) b.task_ids.forEach(function(id) { if (!seen[id]) { seen[id] = true; allTasks.push({ id: id, batchIndex: i }); } });
    });
  }
  if (runState.results) { runState.results.forEach(function(r) { if (!seen[r.task_id]) { seen[r.task_id] = true; allTasks.push({ id: r.task_id, batchIndex: -1 }); } }); }
  if (allTasks.length === 0) { grid.innerHTML = '<div class="empty">No tasks yet. Start a run.</div>'; return; }
  var html = '';
  allTasks.forEach(function(t) {
    var r = runState.resultsByTask[t.id] || {};
    var cats = (runState.categories[t.id] || (r.categories||[])).map(function(c) {
      var idx = CATEGORIES.indexOf(c); var color = idx >= 0 ? CAT_COLORS[idx] : '#8b949e';
      return '<span class="cat" style="background:'+color+'22;color:'+color+'">'+c+'</span>';
    }).join('');
    var jr = runState.judgeResults[t.id];
    var judge = jr ? (jr.pass ? '<span class="judge-pass">PASS</span>' : '<span class="judge-fail">FAIL</span>') + ' <span style="color:#8b949e;font-size:9px">'+(jr.via||'')+'</span>' : '';
    var answer = r.answer ? escapeHtml(r.answer) : '<span style="color:#484f58">pending...</span>';
    var cls = r.answer ? '' : 'pending';
    html += '<div class="task-card '+cls+'"><span class="id">'+t.id+'</span> '+cats+'<div class="answer">'+answer+'</div>'+judge+'</div>';
  });
  grid.innerHTML = html;
  document.getElementById('tasksBadge').textContent = allTasks.length;
}
function renderTrace() {
  var div = document.getElementById('traceList');
  if (!runState.events || runState.events.length === 0) { div.innerHTML = '<div class="empty">No events yet. Start a run.</div>'; return; }
  var html = '';
  runState.events.forEach(function(ev) {
    var ts = new Date(ev.timestamp).toLocaleTimeString();
    var detail = '';
    if (ev.type === 'call' && ev.data) {
      detail = '<div style="color:#8b949e;font-size:11px">tokens: '+ev.data.total_tokens+' (prompt='+ev.data.prompt_tokens+', output='+ev.data.completion_tokens+', cached='+ev.data.cached_tokens+', reasoning='+ev.data.reasoning_tokens+')</div>';
      if (ev.data.output_preview) detail += '<div class="code-block">'+escapeHtml(ev.data.output_preview)+'</div>';
    }
    if (ev.type === 'tool' && ev.data) {
      detail = '<div class="code-block">'+escapeHtml(ev.data.code||'')+'</div>';
      if (ev.data.stdout) detail += '<div style="color:#7ee787;font-size:11px">stdout: '+escapeHtml((ev.data.stdout||'').substring(0,500))+'</div>';
      if (ev.data.error) detail += '<div style="color:#f85149;font-size:11px">error: '+escapeHtml(ev.data.error)+'</div>';
    }
    if (ev.type === 'submit' && ev.data) {
      var answers = ev.data.answers || {};
      detail = '<div class="submit-event"><div class="submit-label">submit()</div><div class="submit-answers">';
      for (var k in answers) { detail += '<div class="submit-answer-row"><span class="submit-task-id">'+k+'</span><span class="submit-answer-val">'+escapeHtml(answers[k])+'</span></div>'; }
      detail += '</div></div>';
    }
    if (ev.type === 'result' && ev.data) {
      detail = '<div style="margin-top:3px;font-size:11px;color:#c9d1d9">'+escapeHtml(ev.data.answer||'')+'</div>';
    }
    var cls = ev.type.split('_')[0];
    html += '<div class="event-item '+cls+'"><span class="ts">'+ts+'</span> <span class="type">'+ev.type+'</span> '+formatEventBrief(ev)+detail+'</div>';
  });
  div.innerHTML = html;
  div.scrollTop = div.scrollHeight;
}
function formatEventBrief(ev) {
  switch (ev.type) {
    case 'batch_start': return 'Batch '+ev.data.index+' ('+ev.data.size+' tasks, effort='+ev.data.effort+(ev.data.model?', model='+ev.data.model.split('/').pop():'')+')';
    case 'batch_end': return 'Batch '+ev.data.index+' done ('+ev.data.calls+' calls, '+ev.data.tools+' tools)';
    case 'call': return 'LLM call turn='+ev.data.turn+' effort='+ev.data.effort+' tokens='+ev.data.total_tokens+(ev.data.error?' ERR':'');
    case 'tool': return 'MicroPython tool #'+ev.data.tool_index+(ev.data.error?' ERROR':'');
    case 'submit': return 'submit() called with '+Object.keys(ev.data.answers||{}).length+' answers';
    case 'result': return ev.data.task_id+' = '+(ev.data.answer||'').substring(0,80);
    case 'classify_done': return Object.keys(ev.data).length+' tasks classified';
    case 'batch_plan': return ev.data.batches.length+' batches';
    case 'done': return ev.data.total_tokens+' tokens, '+ev.data.calls+' calls';
    case 'error': return ev.data.message;
    default: return '';
  }
}
init();
