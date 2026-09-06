const form = document.querySelector('#training-form');
const statusText = document.querySelector('#status');
const sessionStatus = document.querySelector('#session-status');
const metricsText = document.querySelector('#metrics');
const startButton = document.querySelector('#start');
const stopButton = document.querySelector('#stop');
const charts = [...document.querySelectorAll('canvas[data-field]')];
let activeRunId = null;
let socket = null;
let metrics = [];

function setStatus(text) { statusText.textContent = text; }
function drawChart(canvas) {
  const field = canvas.dataset.field, label = canvas.dataset.label, context = canvas.getContext('2d');
  const { width, height } = canvas; context.clearRect(0, 0, width, height);
  context.strokeStyle = '#30394a'; context.lineWidth = 1;
  for (let y = 30; y < height - 18; y += 42) { context.beginPath(); context.moveTo(38, y); context.lineTo(width - 10, y); context.stroke(); }
  const values = metrics.map(record => Number(record[field])).filter(Number.isFinite);
  if (!values.length) { context.fillStyle = '#a9b4c6'; context.font = '13px system-ui'; context.fillText('No recorded data', 38, height / 2); return; }
  const low = Math.min(...values, 0), high = Math.max(...values, 0), span = high - low || 1;
  context.strokeStyle = '#00e5ff'; context.lineWidth = 3; context.beginPath();
  values.forEach((value, index) => { const x = 38 + index * (width - 50) / Math.max(values.length - 1, 1); const y = height - 22 - (value - low) / span * (height - 54); index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.stroke();
  context.fillStyle = '#a9b4c6'; context.font = '12px system-ui'; context.fillText(`${label}: ${high.toFixed(3)} to ${low.toFixed(3)}`, 38, 17);
}
function drawCharts() { charts.forEach(drawChart); }
function addMetric(record) { metrics.push(record); drawCharts(); const epsilon = Number(record.epsilon); const epsilonText = Number.isFinite(epsilon) ? `, ε ${epsilon.toFixed(3)}` : ''; metricsText.textContent = `${record.mode === 'evaluation' ? 'Headless evaluation' : 'Training'} · episode ${record.episode_num}: reward ${Number(record.total_reward).toFixed(2)}, steps ${record.steps}${epsilonText}`; }
function openSocket(runId) {
  if (socket) socket.close(); metrics = []; drawCharts();
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'; socket = new WebSocket(`${scheme}://${location.host}/ws/training/${runId}`);
  socket.onmessage = event => { const message = JSON.parse(event.data); if (message.type === 'metric') addMetric(message.data); if (message.type === 'status') { setStatus(`Run ${message.data.status}`); if (message.data.status !== 'running') { startButton.disabled = false; stopButton.disabled = true; loadRuns(); } } };
  socket.onerror = () => { setStatus('Live connection unavailable; checking saved metrics.'); pollRun(runId); };
}
async function pollRun(runId) { const status = await fetch(`/training/status/${runId}`); if (!status.ok) return; const data = await status.json(); const history = await fetch(`/runs/${data.history_dir.split(/[\\/]/).pop()}/metrics`); if (history.ok) { metrics = await history.json(); drawCharts(); } if (data.status === 'running') setTimeout(() => pollRun(runId), 1000); else { setStatus(`Run ${data.status}`); loadRuns(); } }
async function launchSession(mode) { const response = await fetch(`/sessions/${mode}/start`, {method:'POST'}); const data = await response.json(); if (!response.ok) { sessionStatus.textContent = data.detail || 'Could not start session.'; return; } if (data.hosted) { sessionStatus.textContent = 'Headless SARSA evaluation is running. Live results are shown below.'; activeRunId = data.run_id; metrics = []; drawCharts(); openSocket(activeRunId); return; } sessionStatus.textContent = `${mode === 'human' ? 'Human drive' : 'SARSA playback'} opened in a native PyGame window.`; }
document.querySelectorAll('[data-session]').forEach(button => button.addEventListener('click', () => launchSession(button.dataset.session)));
form.addEventListener('submit', async event => { event.preventDefault(); const data = Object.fromEntries(new FormData(form)); data.warm_start = form.warm_start.checked; data.resume = form.resume.checked; if (!data.seed) delete data.seed; for (const key of ['episodes','vehicles_count','duration','seed']) if (key in data) data[key] = Number(data[key]); for (const key of ['learning_rate','gamma','epsilon_start','epsilon_end','epsilon_decay']) data[key] = Number(data[key]); const response = await fetch('/training/start', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data) }); const result = await response.json(); if (!response.ok) { setStatus(result.detail || 'Could not start training'); return; } activeRunId = result.run_id; startButton.disabled = true; stopButton.disabled = false; setStatus(`Run ${result.run_id} running`); openSocket(activeRunId); });
stopButton.addEventListener('click', async () => { if (activeRunId) await fetch(`/training/stop/${activeRunId}`, {method:'POST'}); });
async function loadRuns() { const response = await fetch('/runs'); const runs = await response.json(); const target = document.querySelector('#runs'); target.replaceChildren(); if (!runs.length) { target.textContent = 'No completed runs yet.'; return; } runs.forEach(run => { const button = document.createElement('button'); button.className = 'run'; button.textContent = `${new Date(run.created_at * 1000).toLocaleString()} — ${run.args?.episodes ?? '?'} episodes`; button.onclick = async () => { const history = await fetch(`/runs/${run.artifact_run_id}/metrics`); metrics = await history.json(); drawCharts(); metricsText.textContent = `Loaded ${metrics.length} recorded episodes.`; }; target.append(button); }); }
async function loadDemonstrations() { const response = await fetch('/demonstrations/summary'); const summary = await response.json(); document.querySelector('#demo-summary').textContent = `${summary.num_episodes} saved demonstrations · ${summary.total_transitions} transitions · total reward ${Number(summary.total_reward).toFixed(2)}`; }
async function loadDocumentation() { const response = await fetch('/documentation'); const documents = await response.json(); const list = document.querySelector('#doc-list'); for (const doc of documents) { const button = document.createElement('button'); button.className = 'doc-button'; button.textContent = doc.filename.replace('.md','').replaceAll('_',' '); button.onclick = async () => { const page = await fetch(`/documentation/${doc.id}`); const data = await page.json(); document.querySelector('#doc-content').textContent = data.content || data.detail; document.querySelectorAll('.doc-button').forEach(item => item.classList.remove('active')); button.classList.add('active'); }; list.append(button); } list.querySelector('button')?.click(); }
loadRuns(); loadDemonstrations(); loadDocumentation(); drawCharts();
