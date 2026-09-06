const form = document.querySelector('#training-form');
const statusText = document.querySelector('#status');
const metricsText = document.querySelector('#metrics');
const startButton = document.querySelector('#start');
const stopButton = document.querySelector('#stop');
const canvas = document.querySelector('#chart');
const context = canvas.getContext('2d');
let activeRunId = null;
let socket = null;
let metrics = [];

function setStatus(text) { statusText.textContent = text; }
function drawChart() {
  const { width, height } = canvas; context.clearRect(0, 0, width, height);
  context.strokeStyle = '#343b49'; context.lineWidth = 1;
  for (let y = 35; y < height; y += 55) { context.beginPath(); context.moveTo(42, y); context.lineTo(width - 12, y); context.stroke(); }
  if (!metrics.length) return;
  const rewards = metrics.map(record => Number(record.total_reward));
  const low = Math.min(...rewards, 0), high = Math.max(...rewards, 0), span = high - low || 1;
  context.strokeStyle = '#00e5ff'; context.lineWidth = 3; context.beginPath();
  rewards.forEach((reward, index) => { const x = 42 + index * (width - 54) / Math.max(rewards.length - 1, 1); const y = height - 26 - (reward - low) / span * (height - 62); index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.stroke();
  context.fillStyle = '#a9b4c6'; context.font = '12px system-ui'; context.fillText(`reward ${high.toFixed(2)} to ${low.toFixed(2)}`, 42, 18);
}
function addMetric(record) { metrics.push(record); drawChart(); metricsText.textContent = `Episode ${record.episode_num}: reward ${Number(record.total_reward).toFixed(2)}, steps ${record.steps}`; }
function openSocket(runId) {
  if (socket) socket.close(); metrics = []; drawChart();
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'; socket = new WebSocket(`${scheme}://${location.host}/ws/training/${runId}`);
  socket.onmessage = event => { const message = JSON.parse(event.data); if (message.type === 'metric') addMetric(message.data); if (message.type === 'status') { setStatus(`Run ${message.data.status}`); if (message.data.status !== 'running') { startButton.disabled = false; stopButton.disabled = true; loadRuns(); } } };
}
form.addEventListener('submit', async event => { event.preventDefault(); const data = Object.fromEntries(new FormData(form)); data.warm_start = form.warm_start.checked; if (!data.seed) delete data.seed; for (const key of ['episodes','vehicles_count','duration','seed']) if (key in data) data[key] = Number(data[key]); for (const key of ['learning_rate','gamma','epsilon_start','epsilon_end','epsilon_decay']) data[key] = Number(data[key]); const response = await fetch('/training/start', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data) }); if (!response.ok) { const error = await response.json(); setStatus(error.detail || 'Could not start training'); return; } const job = await response.json(); activeRunId = job.run_id; startButton.disabled = true; stopButton.disabled = false; setStatus(`Run ${job.run_id} running`); openSocket(activeRunId); });
stopButton.addEventListener('click', async () => { if (!activeRunId) return; await fetch(`/training/stop/${activeRunId}`, {method:'POST'}); });
async function loadRuns() { const response = await fetch('/runs'); const runs = await response.json(); const target = document.querySelector('#runs'); target.replaceChildren(); if (!runs.length) { target.textContent = 'No completed runs yet.'; return; } runs.forEach(run => { const button = document.createElement('button'); button.className = 'run'; button.textContent = `${new Date(run.created_at * 1000).toLocaleString()} — ${run.args?.episodes ?? '?'} episodes`; button.onclick = async () => { const response = await fetch(`/runs/${run.artifact_run_id}/metrics`); metrics = await response.json(); drawChart(); metricsText.textContent = `Loaded ${metrics.length} recorded episodes.`; }; target.append(button); }); }
loadRuns(); drawChart();
