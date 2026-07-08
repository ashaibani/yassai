let currentRunId = null;
let eventSource = null;
let routingConfig = { effort_tier_map: {}, model_route_map: {}, categories: [], effort_levels: [], models: [] };
let runState = { events: [], results: [], metrics: {}, judgeResults: {}, tasks: [], batches: [], categories: {}, resultsByTask: {}, batchStatus: {} };
const CATEGORIES = ['factual_knowledge','mathematical_reasoning','sentiment_classification','text_summarisation','named_entity_recognition','code_debugging','logical_deductive_reasoning','code_generation'];
const CAT_COLORS = ['#58a6ff','#3fb950','#d29922','#bc8cff','#f85149','#7ee787','#ffa657','#79c0ff'];
const CAT_SHORT = { factual_knowledge:'Factual', mathematical_reasoning:'Maths', sentiment_classification:'Sentiment', text_summarisation:'Summary', named_entity_recognition:'NER', code_debugging:'Code Debug', logical_deductive_reasoning:'Logic', code_generation:'Code Gen' };
let activeTab = 'dashboard';

async function loadTasks() {
  const res = await fetch('/api/tasks');
  const sets = await res.json();
  const sel = document.getElementById('tasksFile');
  sel.innerHTML = sets.map(s => '<option value="'+s.file+'">'+s.name+' ('+s.count+')'+'</option>').join('');
}

function switchTab(name) {
  activeTab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  ['dashboard','batches','tasks','trace','tools'].forEach(t => { var el = document.getElementById('tab-'+t); if (el) el.style.display = t === name ? '' : 'none'; });
  rerenderActiveTab();
}

function rerenderActiveTab() {
  switch (activeTab) {
    case 'dashboard': renderMetrics(); renderBatchPlan(); renderEventStream(); break;
    case 'batches': renderBatches(); break;
    case 'tasks': renderTasks(); break;
    case 'trace': renderTrace(); break;
    case 'tools': renderTools(); break;
  }
  updateBadges();
}

async function startRun() {
  var btn = document.getElementById('runBtn');
  btn.disabled = true; btn.textContent = 'Running...';
  runState = { events: [], results: [], metrics: {}, liveMetrics: null, judgeResults: {}, tasks: [], batches: [], categories: {}, resultsByTask: {}, batchStatus: {} };
  document.getElementById('runStatusBadge').innerHTML = '<span class="status-dot running"></span>Running';
  // Status badge will be updated by SSE events
  rerenderActiveTab();
  var req = {
    tasks_file: document.getElementById('tasksFile').value,
    model: document.getElementById('model').value,
    batch_size: parseInt(document.getElementById('batchSize').value),
    batch_tokens: parseInt(document.getElementById('batchTokens').value),
    max_concurrency: parseInt(document.getElementById('maxConcurrency').value) || 1,
    reasoning_effort: document.getElementById('effort').value,
    disable_hints: document.getElementById('disableHints').checked,
    enable_classifier: document.getElementById('enableClassifier').checked,
    enable_judge: document.getElementById('enableJudge').checked,
    judge_model: document.getElementById('judgeModel').value,
    judge_effort: document.getElementById('judgeEffort').value,
    effort_tier_map: routingConfig.effort_tier_map,
    model_route_map: routingConfig.model_route_map,
  };
  var res = await fetch('/api/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(req) });
  var data = await res.json();
  if (data.error) { alert(data.error); btn.disabled = false; btn.textContent = 'Run Agent'; return; }
  currentRunId = data.run_id;
  connectSSE(data.run_id);
}

function connectSSE(runId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource('/api/run/'+runId+'/events');
  eventSource.onmessage = function(e) {
    var ev = JSON.parse(e.data);
    if (ev.type === 'judge_progress') {
      runState.judgeResults = ev.judge_results || {};
      runState.events.push(ev);
      rerenderActiveTab();
      return;
    }
    if (ev.type === 'final') {
      runState.results = ev.results || []; runState.metrics = ev.metrics || {}; runState.judgeResults = ev.judge_results || {};
      runState.results.forEach(function(r) { runState.resultsByTask[r.task_id] = r; });
      document.getElementById('runStatusBadge').innerHTML = '<span class="status-dot done"></span>Done';
      document.getElementById('runBtn').disabled = false; document.getElementById('runBtn').textContent = 'Run Agent';
      rerenderActiveTab(); eventSource.close(); return;
    }
    runState.events.push(ev); handleEvent(ev); rerenderActiveTab();
  };
  eventSource.onerror = function() {
    // Only show error if we don't already have final results (the SSE connection
    // closing after the final event is normal, not an error).
    if (runState.metrics && runState.metrics.total_tokens) {
      if (document.getElementById('runBtn').disabled) {
        document.getElementById('runStatusBadge').innerHTML = '<span class="status-dot done"></span>Done';
        document.getElementById('runBtn').disabled = false; document.getElementById('runBtn').textContent = 'Run Agent';
      }
      eventSource.close(); return;
    }
    document.getElementById('runStatusBadge').innerHTML = '<span class="status-dot error"></span>Error';
    document.getElementById('runBtn').disabled = false; document.getElementById('runBtn').textContent = 'Run Agent'; eventSource.close();
  };
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'classify_done': runState.categories = ev.data; break;
    case 'batch_plan':
      runState.batches = ev.data.batches;
      ev.data.batches.forEach(function(b) { runState.batchStatus[b.index] = { status: 'pending', effort: b.effort, model: b.model || '', taskIds: b.task_ids, size: b.size, calls: 0, tokens: 0 }; });
      break;
    case 'batch_start':
      if (!runState.batchStatus[ev.data.index]) runState.batchStatus[ev.data.index] = {};
      runState.batchStatus[ev.data.index].status = 'running';
      break;
    case 'batch_end':
      if (runState.batchStatus[ev.data.index]) {
        runState.batchStatus[ev.data.index].status = ev.data.error ? 'error' : 'done';
        runState.batchStatus[ev.data.index].calls = ev.data.calls || 0;
        runState.batchStatus[ev.data.index].tools = ev.data.tools || 0;
        runState.batchStatus[ev.data.index].error = ev.data.error || '';
      }
      break;
    case 'call':
      // Accumulate live metrics from call events
      if (!runState.liveMetrics) runState.liveMetrics = { total_tokens: 0, prompt_tokens: 0, output_tokens: 0, cached_tokens: 0, reasoning_tokens: 0, calls: 0 };
      runState.liveMetrics.total_tokens += ev.data.total_tokens || 0;
      runState.liveMetrics.prompt_tokens += ev.data.prompt_tokens || 0;
      runState.liveMetrics.output_tokens += ev.data.completion_tokens || 0;
      runState.liveMetrics.cached_tokens += ev.data.cached_tokens || 0;
      runState.liveMetrics.reasoning_tokens += ev.data.reasoning_tokens || 0;
      runState.liveMetrics.calls++;
      // Update batch-level stats
      var bi = batchOfEvent(ev);
      if (bi >= 0 && runState.batchStatus[bi]) {
        runState.batchStatus[bi].calls = (runState.batchStatus[bi].calls || 0) + 1;
        runState.batchStatus[bi].tokens = (runState.batchStatus[bi].tokens || 0) + (ev.data.total_tokens || 0);
      }
      break;
    case 'result':
      runState.resultsByTask[ev.data.task_id] = { task_id: ev.data.task_id, answer: ev.data.answer, categories: ev.data.categories || [], batchIndex: batchOf(ev.data.task_id) };
      break;
    case 'done': runState.metrics = ev.data; break;
  }
}

function batchOf(taskId) {
  for (var i = 0; i < runState.batches.length; i++) { if (runState.batches[i].task_ids.indexOf(taskId) >= 0) return i; }
  return -1;
}
function batchOfEvent(ev) {
  var d = ev.data || {};
  if (d.task_ids && d.task_ids.length > 0) return batchOf(d.task_ids[0]);
  if (d.task_id) return batchOf(d.task_id);
  return -1;
}

function updateBadges() {
  var tc = Object.keys(runState.resultsByTask).length;
  var wc = runState.events.filter(function(e) { return e.type === 'tool'; }).length;
  var tb = document.getElementById('tasksBadge'); if (tb) { tb.textContent = tc; if (tc > 0) tb.classList.add('live'); else tb.classList.remove('live'); }
  var tl = document.getElementById('toolsBadge'); if (tl) { tl.textContent = wc; if (wc > 0) tl.classList.add('live'); else tl.classList.remove('live'); }
}

function card(label, value, color) {
  return '<div class="metric-card"><div class="label">'+label+'</div><div class="value '+color+'">'+value+'</div></div>';
}

function renderMetrics() {
  // Use final metrics if available, otherwise live accumulated metrics
  var m = runState.metrics;
  if (!m || !m.total_tokens) { m = runState.liveMetrics || {}; }
  if (!m.total_tokens) { document.getElementById('metricsRow').innerHTML = ''; return; }
  var dur = m.duration_ms ? (m.duration_ms/1000).toFixed(1)+'s' : '-';
  var jr = runState.judgeResults || {};
  var pc = Object.values(jr).filter(function(j) { return j.pass; }).length;
  var tc2 = Object.keys(jr).length;
  var acc = tc2 > 0 ? pc+'/'+tc2 : '-';
  var accColor = tc2 > 0 ? (pc === tc2 ? 'green' : (pc < tc2/2 ? 'red' : 'orange')) : '';
  var toolCount = runState.events.filter(function(e) { return e.type === 'tool'; }).length;
  document.getElementById('metricsRow').innerHTML = [
    card('Total Tokens', m.total_tokens, 'orange'), card('Prompt', m.prompt_tokens || 0, ''),
    card('Output', m.output_tokens || 0, ''), card('Cached', m.cached_tokens || 0, 'green'),
    card('Reasoning', m.reasoning_tokens || 0, 'orange'), card('Calls', m.calls || 0, ''),
    card('Tools', toolCount, ''), card('Accuracy', acc, accColor),
    card('Duration', dur, '')
  ].join('');
}

function renderBatchPlan() {
  var div = document.getElementById('batchPlan');
  if (!runState.batches || runState.batches.length === 0) { div.innerHTML = ''; return; }
  var html = '<div style="font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Batch Plan</div>';
  html += '<div style="display:flex;flex-wrap:wrap;gap:4px">';
  runState.batches.forEach(function(b) {
    var st = runState.batchStatus[b.index] || {};
    var model = b.model ? b.model.split('/').pop() : '';
    var calls = st.calls || 0;
    var tokens = st.tokens || 0;
    var info = calls > 0 ? ' ('+calls+'c, '+tokens+'t)' : '';
    var errTag = st.error ? ' <span style="color:#f85149">ERR</span>' : '';
    html += '<div style="background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:3px 8px;font-size:11px">';
    html += '<span class="status-dot '+(st.status||'idle')+'"></span>';
    html += 'B'+b.index+': '+b.size+' <span class="effort">'+b.effort+'</span>';
    if (model) html += ' <span style="color:#58a6ff">'+model+'</span>';
    html += '<span style="color:#8b949e">'+info+'</span>';
    html += errTag;
    html += '</div>';
  });
  html += '</div>';
  // Judge progress section
  var jr = runState.judgeResults || {};
  var jrKeys = Object.keys(jr);
  if (jrKeys.length > 0) {
    var jrPass = jrKeys.filter(function(k) { return jr[k].pass; }).length;
    var jrFail = jrKeys.length - jrPass;
    html += '<div style="margin-top:8px;font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px">Judge Progress</div>';
    html += '<div style="display:flex;gap:6px;align-items:center">';
    html += '<div style="background:#0d1117;border:1px solid #30363d;border-radius:5px;padding:3px 10px;font-size:12px">';
    html += '<span style="color:#3fb950">'+jrPass+' PASS</span>';
    if (jrFail > 0) html += ' <span style="color:#f85149">'+jrFail+' FAIL</span>';
    html += ' <span style="color:#8b949e">/ '+jrKeys.length+'</span>';
    html += '</div>';
    // Progress bar
    var total = runState.results ? runState.results.length : (runState.batches.reduce(function(s,b){return s+b.size;},0));
    if (total > 0) {
      var pct = Math.round(jrKeys.length / total * 100);
      html += '<div style="flex:1;height:6px;background:#30363d;border-radius:3px;overflow:hidden"><div style="height:6px;width:'+pct+'%;background:#bc8cff;border-radius:3px"></div></div>';
      html += '<span style="color:#bc8cff;font-size:11px">'+pct+'%</span>';
    }
    html += '</div>';
  }
  div.innerHTML = html;
}

function formatEventItem(ev) {
  var cls = ev.type.split('_')[0];
  var ts = new Date(ev.timestamp).toLocaleTimeString();
  var bi = batchOfEvent(ev);
  var bb = bi >= 0 ? '<span class="batch-badge">B'+bi+'</span>' : '';
  var text = '';
  switch (ev.type) {
    case 'classify_start': text = 'Classifying '+ev.data.task_count+' tasks...'; break;
    case 'classify_done': text = 'Classification done ('+Object.keys(ev.data).length+' tasks)'; break;
    case 'classify_skip': text = 'Classifier skipped: '+ev.data.reason; break;
    case 'batch_plan': text = ev.data.batches.length+' batches planned'; break;
    case 'batch_start': text = 'Batch '+ev.data.index+' start ('+ev.data.size+' tasks, effort='+ev.data.effort+(ev.data.model?', model='+ev.data.model.split('/').pop():'')+')'; break;
    case 'batch_end': text = 'Batch '+ev.data.index+' done ('+ev.data.calls+' calls, '+ev.data.tools+' tools'+(ev.data.error?' ERR':'')+')'; break;
    case 'call': text = 'LLM call turn='+ev.data.turn+' effort='+ev.data.effort+' tok='+ev.data.total_tokens+' (cached='+(ev.data.cached_tokens||0)+', reason='+(ev.data.reasoning_tokens||0)+')'+(ev.data.error?' ERR':''); break;
    case 'tool': text = 'MicroPython tool #'+ev.data.tool_index+(ev.data.error?' ERROR':''); break;
    case 'result': text = ev.data.task_id+' = '+(ev.data.answer||'').substring(0,60)+(ev.data.answer&&ev.data.answer.length>60?'...':''); break;
    case 'done': text = 'Complete: '+ev.data.total_tokens+' tokens, '+ev.data.calls+' calls'; break;
    case 'judge_progress':
      var jp = (ev.data && ev.data.judge_results) ? ev.data.judge_results : (ev.judge_results || {});
      var jKeys = Object.keys(jp);
      var jPass = jKeys.filter(function(k) { return jp[k].pass; }).length;
      var jCount = (ev.data && ev.data.count) ? ev.data.count : (ev.count || jKeys.length);
      text = 'Judge progress: '+jPass+'/'+jKeys.length+' scored ('+jCount+' total)';
      break;
    case 'error': text = 'ERROR: '+ev.data.message; break;
    default: text = ev.type;
  }
  return '<div class="event-item '+cls+'"><span class="ts">'+ts+'</span> '+bb+'<span class="type">'+ev.type+'</span> '+text+'</div>';
}

function renderEventStream() {
  var list = document.getElementById('eventStream');
  if (!runState.events || runState.events.length === 0) { list.innerHTML = '<p style="color:#8b949e;font-size:12px">No events yet. Start a run.</p>'; return; }
  list.innerHTML = runState.events.map(formatEventItem).join('');
  list.scrollTop = list.scrollHeight;
}

function toggleBatchBody(header) {
  var body = header.nextElementSibling;
  body.style.display = body.style.display === 'none' ? '' : 'none';
}

function escapeHtml(s) { if (!s) return ''; return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

loadTasks();
loadModels();
loadRouting();

async function loadModels() {
  var res = await fetch('/api/models');
  var data = await res.json();
  var models = data.models || [];
  var defaultM = data.default || '';
  var modelSel = document.getElementById('model');
  modelSel.innerHTML = models.map(function(m) {
    var short = m.split('/').pop();
    return '<option value="'+m+'"'+(m===defaultM?' selected':'')+'>'+short+'</option>';
  }).join('');
  var judgeSel = document.getElementById('judgeModel');
  var judgeVal = judgeSel.value;
  judgeSel.innerHTML = '<option value="">(same as agent)</option>' + models.map(function(m) {
    var short = m.split('/').pop();
    return '<option value="'+m+'"'+(m===judgeVal?' selected':'')+'>'+short+'</option>';
  }).join('');
}
function renderBatches() {
  var div = document.getElementById('batchList');
  if (!runState.batches || runState.batches.length === 0) { div.innerHTML = '<p style="color:#8b949e;font-size:12px">No batches planned yet.</p>'; return; }
  var html = '';
  runState.batches.forEach(function(b) {
    var st = runState.batchStatus[b.index] || { status: 'pending' };
    var model = b.model ? b.model.split('/').pop() : 'default';
    var batchEvents = runState.events.filter(function(ev) { return batchOfEvent(ev) === b.index; });
    var answered = b.task_ids.filter(function(tid) { return runState.resultsByTask[tid]; }).length;
    html += '<div class="batch-group">';
    html += '<div class="batch-group-header" onclick="toggleBatchBody(this)">';
    html += '<span class="status-dot '+(st.status||'idle')+'"></span>';
    html += '<span class="batch-num">Batch '+b.index+'</span>';
    html += '<span class="task-count">'+answered+'/'+b.task_ids.length+' answered</span>';
    html += '<span class="effort-tag">'+b.effort+'</span>';
    html += '<span class="model-tag">'+model+'</span>';
    html += '<span class="status">'+batchEvents.length+' events</span>';
    html += '</div>';
    html += '<div class="batch-group-body" style="display:none">';
    html += '<div style="margin-bottom:6px;font-size:11px;color:#8b949e">Tasks: '+b.task_ids.join(', ')+'</div>';
    batchEvents.forEach(function(ev) { html += formatEventItem(ev); });
    b.task_ids.forEach(function(tid) {
      var r = runState.resultsByTask[tid];
      if (r) {
        var jr = runState.judgeResults[tid];
        var judge = jr ? (jr.pass ? '<span class="judge-pass">PASS</span>' : '<span class="judge-fail">FAIL</span>') : '';
        html += '<div style="margin-top:4px;padding:4px 8px;background:#0d1117;border-radius:4px;font-size:11px"><span style="color:#58a6ff;font-weight:600">'+tid+'</span>: '+escapeHtml((r.answer||'').substring(0,100))+' '+judge+'</div>';
      }
    });
    html += '</div></div>';
  });
  div.innerHTML = html;
}

function renderTasks() {
  var grid = document.getElementById('taskGrid');
  var allTasks = [];
  if (runState.batches && runState.batches.length > 0) {
    runState.batches.forEach(function(b) { b.task_ids.forEach(function(tid) { allTasks.push({ id: tid, batchIndex: b.index }); }); });
  } else if (runState.results && runState.results.length > 0) {
    runState.results.forEach(function(r) { allTasks.push({ id: r.task_id, batchIndex: -1 }); });
  }
  if (allTasks.length === 0) { grid.innerHTML = '<p style="color:#8b949e;font-size:12px">No tasks yet. Start a run.</p>'; return; }
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
    var batchTag = t.batchIndex >= 0 ? '<div class="batch-tag">Batch '+t.batchIndex+'</div>' : '';
    html += '<div class="task-card '+cls+'"><span class="id">'+t.id+'</span> '+cats+'<div class="answer">'+answer+'</div>'+judge+batchTag+'</div>';
  });
  grid.innerHTML = html;
}

function renderTrace() {
  var div = document.getElementById('traceList');
  if (!runState.events || runState.events.length === 0) { div.innerHTML = '<p style="color:#8b949e;font-size:12px">No events yet. Start a run.</p>'; return; }
  // Group events by batch for clarity
  var html = '';
  var currentBatch = -2;
  runState.events.forEach(function(ev) {
    var bi = batchOfEvent(ev);
    if (bi >= 0 && bi !== currentBatch) {
      currentBatch = bi;
      html += '<div style="margin:8px 0 4px;font-size:11px;color:#d29922;font-weight:600;border-bottom:1px solid #30363d;padding-bottom:2px">Batch '+bi+'</div>';
    } else if (bi < 0 && currentBatch !== -2) {
      currentBatch = -2;
    }
    var cls = ev.type.split('_')[0];
    var ts = new Date(ev.timestamp).toLocaleTimeString();
    var detail = '';
    if (ev.type === 'call' && ev.data) {
      detail = '<div style="margin-top:3px;color:#8b949e;font-size:11px">tokens: '+ev.data.total_tokens+' (prompt='+ev.data.prompt_tokens+', output='+ev.data.completion_tokens+', cached='+ev.data.cached_tokens+', reasoning='+ev.data.reasoning_tokens+')</div>';
      if (ev.data.output_preview) detail += '<div class="code-block">'+escapeHtml(ev.data.output_preview)+'</div>';
    }
    if (ev.type === 'tool' && ev.data) {
      detail = '<div class="code-block">'+escapeHtml(ev.data.code||'')+'</div>';
      if (ev.data.stdout) detail += '<div style="color:#7ee787;font-size:11px">stdout: '+escapeHtml((ev.data.stdout||'').substring(0,500))+'</div>';
      if (ev.data.json) detail += '<div style="color:#bc8cff;font-size:11px">json: '+escapeHtml(JSON.stringify(ev.data.json).substring(0,500))+'</div>';
      if (ev.data.error) detail += '<div style="color:#f85149;font-size:11px">error: '+escapeHtml(ev.data.error)+'</div>';
    }
    if (ev.type === 'result' && ev.data) {
      detail = '<div style="margin-top:3px;font-size:11px;color:#c9d1d9">'+escapeHtml(ev.data.answer||'')+'</div>';
    }
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
    case 'result': return ev.data.task_id+' = '+(ev.data.answer||'').substring(0,80);
    case 'classify_done': return Object.keys(ev.data).length+' tasks classified';
    case 'batch_plan': return ev.data.batches.length+' batches';
    case 'done': return ev.data.total_tokens+' tokens, '+ev.data.calls+' calls';
    case 'error': return ev.data.message;
    default: return '';
  }
}

function renderTools() {
  var div = document.getElementById('toolList');
  var tools = runState.events.filter(function(e) { return e.type === 'tool'; });
  if (tools.length === 0) { div.innerHTML = '<p style="color:#8b949e;font-size:12px">No tool calls yet.</p>'; return; }
  var html = '';
  var currentBatch = -2;
  tools.forEach(function(ev, i) {
    var bi = batchOfEvent(ev);
    if (bi >= 0 && bi !== currentBatch) {
      currentBatch = bi;
      html += '<div style="margin:8px 0 4px;font-size:11px;color:#d29922;font-weight:600;border-bottom:1px solid #30363d;padding-bottom:2px">Batch '+bi+'</div>';
    }
    html += '<div style="margin-bottom:10px">';
    html += '<div style="color:#bc8cff;font-weight:600;font-size:12px">Tool Call #'+(i+1)+' <span style="color:#8b949e;font-weight:400">(turn='+ev.data.turn+', tasks='+ev.data.task_ids.join(',').substring(0,40)+')</span></div>';
    html += '<div class="code-block">'+escapeHtml(ev.data.code||'')+'</div>';
    if (ev.data.stdout) html += '<div style="color:#7ee787;font-size:11px">stdout: '+escapeHtml((ev.data.stdout||'').substring(0,1000))+'</div>';
    if (ev.data.json) html += '<div style="color:#bc8cff;font-size:11px">json: '+escapeHtml(JSON.stringify(ev.data.json).substring(0,500))+'</div>';
    if (ev.data.error) html += '<div style="color:#f85149;font-size:11px">error: '+escapeHtml(ev.data.error)+'</div>';
    html += '</div>';
  });
  div.innerHTML = html;
}

async function loadRouting() {
  var res = await fetch('/api/routing');
  routingConfig = await res.json();
  renderRoutingSidebar();
}

function renderRoutingSidebar() {
  var div = document.getElementById('routingSidebar');
  var cats = routingConfig.categories || CATEGORIES;
  var efforts = routingConfig.effort_levels || ['low','medium','high','xhigh'];
  var models = routingConfig.models || [];
  var etMap = routingConfig.effort_tier_map || {};
  var mrMap = routingConfig.model_route_map || {};
  var html = '';
  cats.forEach(function(cat) {
    var idx = cats.indexOf(cat);
    var color = idx < 8 ? CAT_COLORS[idx] : '#8b949e';
    var short = CAT_SHORT[cat] || cat.replace(/_/g,' ');
    var effort = etMap[cat] || '';
    var model = mrMap[cat] || '';
    html += '<div class="route-card">';
    html += '<span class="route-cat" style="color:'+color+'">'+short+'</span>';
    html += '<div class="route-controls">';
    html += '<select id="effort_'+cat+'" class="effort-sel" onchange="saveRouting()">';
    html += '<option value="">low</option>';
    efforts.forEach(function(e) { html += '<option value="'+e+'"'+(effort===e?' selected':'')+'>'+e+'</option>'; });
    html += '</select>';
    html += '<select id="model_'+cat+'" class="model-sel" onchange="saveRouting()">';
    html += '<option value="">(default)</option>';
    models.forEach(function(m) { html += '<option value="'+m+'"'+(model===m?' selected':'')+'>'+m.split('/').pop()+'</option>'; });
    html += '</select>';
    html += '</div>';
    html += '</div>';
  });
  html += '<div style="margin-top:6px"><button class="btn btn-secondary" style="width:auto;font-size:11px;padding:3px 8px" onclick="resetRouting()">Reset</button></div>';
  div.innerHTML = html;
}

async function saveRouting() {
  var cats = routingConfig.categories || CATEGORIES;
  var etMap = {}, mrMap = {};
  cats.forEach(function(cat) {
    var e = document.getElementById('effort_'+cat); if (e && e.value) etMap[cat] = e.value;
    var m = document.getElementById('model_'+cat); if (m && m.value) mrMap[cat] = m.value;
  });
  await fetch('/api/routing', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({effort_tier_map: etMap, model_route_map: mrMap}) });
  routingConfig.effort_tier_map = etMap;
  routingConfig.model_route_map = mrMap;
}

function resetRouting() {
  routingConfig.effort_tier_map = {'logical_deductive_reasoning':'xhigh','code_debugging':'xhigh','code_generation':'xhigh'};
  var models = routingConfig.models || [];
  var hasKimi = models.indexOf('accounts/fireworks/models/kimi-k2p7-code') >= 0;
  routingConfig.model_route_map = hasKimi ? {'code_debugging':'accounts/fireworks/models/kimi-k2p7-code','code_generation':'accounts/fireworks/models/kimi-k2p7-code'} : {};
  saveRouting();
  renderRoutingSidebar();
}

async function classifyPrompt() {
  var input = document.getElementById('classifyInput');
  var prompt = input.value.trim();
  if (!prompt) return;
  var res = await fetch('/api/classify', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({prompt: prompt}) });
  var data = await res.json();
  var div = document.getElementById('classifyResults');
  if (data.error) { div.innerHTML = '<p style="color:#f85149;font-size:11px">'+data.error+'</p>'; return; }
  if (!data.predictions || data.predictions.length === 0) { div.innerHTML = '<p style="color:#8b949e;font-size:11px">No categories above threshold.</p>'; return; }
  var html = '';
  data.predictions.forEach(function(p) {
    var idx = CATEGORIES.indexOf(p.label); var color = idx >= 0 ? CAT_COLORS[idx] : '#8b949e';
    var pct = Math.round(p.score * 100);
    html += '<div style="margin:3px 0"><span style="color:'+color+';font-size:10px">'+p.label+'</span> ('+pct+'%)<div class="pred-bar"><div class="pred-bar-fill" style="width:'+pct+'%;background:'+color+'"></div></div></div>';
  });
  div.innerHTML = html;
}
