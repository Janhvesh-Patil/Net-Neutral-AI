/* ================================================================
   Net-Neutral AI — Frontend Logic  (app.js)
   Built upon: existing screen/navigation structure
   Extends: coordinator setup, client flow, live training, leaderboard

   ────────────────────────────────────────────────────────────────
   BACKEND INTEGRATION REFERENCE
   ────────────────────────────────────────────────────────────────
   CURRENT ENDPOINTS (v1.0 / scope_creep branch):
     POST /register             → register client
     GET  /get_clients          → list connected clients
     POST /upload_dataset       → upload training CSV
     POST /start_training       → begin training session
     GET  /status               → coordinator training status
     GET  /api/client_status/:id → per-client training status
     GET  /api/lobby            → session discovery (Phase 2)
     GET  /api/session_info     → this coordinator's session info
     GET  /leaderboard          → public leaderboard

   TRD v2.1 TARGET ENDPOINTS (migrate to these):
     POST /register_node        → replaces /register (JWT required)
     POST /jobs                 → replaces /upload_dataset + /start_training
     GET  /jobs/:id             → job details and status
     GET  /jobs/:id/model       → download trained model .pt
     POST /validate_checkpoint  → validate uploaded .pt
     POST /submit               → client weight submission (JWT)
     GET  /status/:job_id       → replaces /status
     GET  /stats                → public stats for landing page
     GET  /leaderboard          → same endpoint, now queries Supabase

   ALL API CALLS: use api() helper below.
   JWT AUTH:  uncomment Supabase blocks when credentials are set.
   ================================================================ */


// ── GLOBAL STATE ─────────────────────────────────────────────────
const S = {
  coordURL:        window.location.origin, // Set on coordinator setup
  role:            null,                   // 'coordinator' | 'client'
  clientId:        null,                   // e.g. 'client_A'
  sessionName:     '',
  expectedClients: 3,
  jobId:           null,                   // TRD v2.1: active job_id
  accuracyHistory: [],                     // [{round, acc}] for charts
  selectedSession: null,                   // Session picked from lobby
  signupRole:      null,                   // Role chosen during sign-up
  datasetFile:     null,
  checkpointFile:  null,

  pollCoord:       null,                   // Coordinator polling handle
  pollClient:      null,                   // Client polling handle

  // Supabase session — TRD v2.1
  // sbSession: null,
};

// Chart.js instances
let accChart     = null;
let resultsChart = null;

// Canvas network graph instance
let netGraph = null;
let _netResizeListener = null;


// ════════════════════════════════════════════════════════════════
//  SCREEN NAVIGATION
// ════════════════════════════════════════════════════════════════

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById(id);
  if (el) el.classList.add('active');

  // Per-screen init hooks
  if (id === 'screen-landing')       loadStats();
  if (id === 'screen-leaderboard')   loadLeaderboard();
  if (id === 'screen-live-training') initLiveView();
  if (id === 'screen-agent-setup')   selectOS('windows');
}

function selectRoleAndGo(role) {
  S.role = role;
  showScreen(role === 'coordinator' ? 'screen-coord-setup' : 'screen-client-setup');
}

function goToLive() {
  showScreen('screen-live-training');
}

function exitDashboard() {
  stopAllPolling();
  showScreen('screen-landing');
}

function stopAllPolling() {
  clearInterval(S.pollCoord);
  clearInterval(S.pollClient);
  S.pollCoord = S.pollClient = null;
  if (netGraph) { netGraph.stop(); netGraph = null; }
}


// ════════════════════════════════════════════════════════════════
//  API WRAPPER
// ════════════════════════════════════════════════════════════════

/*
 * Centralized fetch wrapper.
 *
 * TODO (TRD v2.1): Attach JWT to all protected requests.
 *   const sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
 *   const { data: { session } } = await sb.auth.getSession();
 *   if (session) headers['Authorization'] = `Bearer ${session.access_token}`;
 */
async function api(path, opts = {}) {
  const url  = S.coordURL + path;
  const hdrs = { ...(opts.headers || {}) };

  // TODO (TRD v2.1): uncomment when Supabase is configured
  // if (S.sbSession) hdrs['Authorization'] = `Bearer ${S.sbSession.access_token}`;

  if (opts.body && !(opts.body instanceof FormData)) {
    hdrs['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }

  const res = await fetch(url, { ...opts, headers: hdrs });
  if (!res.ok) {
    let msg = res.statusText;
    try { const d = await res.clone().json(); msg = d.error || msg; } catch {}
    throw new Error(msg);
  }

  const ct = res.headers.get('Content-Type') || '';
  return ct.includes('application/json') ? res.json() : res;
}


// ════════════════════════════════════════════════════════════════
//  AUTH  (Supabase integration points)
// ════════════════════════════════════════════════════════════════

/*
 * TODO (TRD v2.1): on DOMContentLoaded, check for existing session:
 *
 *   const sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
 *   const { data: { session } } = await sb.auth.getSession();
 *   if (session) {
 *     S.sbSession = session;
 *     const { data: profile } = await sb.from('profiles')
 *       .select('role').eq('id', session.user.id).single();
 *     if (profile.role === 'coordinator') showScreen('screen-coord-setup');
 *     else showScreen('screen-client-setup');
 *   }
 */

async function signIn() {
  const email = el('signin-email').value.trim();
  const pass  = el('signin-password').value;
  if (!email || !pass) return msg('signin-msg', 'Enter email and password.', 'err');

  msg('signin-msg', 'Signing in…', 'wait');

  /*
   * TODO (TRD v2.1):
   *   const { data, error } = await sb.auth.signInWithPassword({ email, password: pass });
   *   if (error) return msg('signin-msg', error.message, 'err');
   *   S.sbSession = data.session;
   *   const { data: profile } = await sb.from('profiles').select('role').eq('id', data.user.id).single();
   *   S.role = profile.role;
   *   redirectByRole(S.role);
   */

  // HACKATHON fallback — simulate auth
  await sleep(500);
  msg('signin-msg', 'Signed in (demo mode)', 'ok');
  await sleep(400);
  S.role = 'coordinator';
  showScreen('screen-coord-setup');
}

async function signUp() {
  const email = el('signup-email').value.trim();
  const pass  = el('signup-password').value;
  if (!email || !pass) return msg('signup-msg', 'Enter email and password.', 'err');
  if (pass.trim().length < 8) return msg('signup-msg', 'Password must be ≥ 8 characters.', 'err');
  if (!S.signupRole)    return msg('signup-msg', 'Please select a role first.', 'err');

  msg('signup-msg', 'Creating account…', 'wait');

  /*
   * TODO (TRD v2.1):
   *   const { data, error } = await sb.auth.signUp({ email, password: pass });
   *   if (error) return msg('signup-msg', error.message, 'err');
   *   await sb.from('profiles').upsert({ id: data.user.id, role: S.signupRole });
   *   S.role = S.signupRole;
   *   redirectByRole(S.role);
   */

  await sleep(500);
  msg('signup-msg', 'Account created (demo mode)', 'ok');
  await sleep(400);
  S.role = S.signupRole;
  showScreen(S.role === 'coordinator' ? 'screen-coord-setup' : 'screen-client-setup');
}

function quickAccess(role) {
  S.role = role;
  showScreen(role === 'coordinator' ? 'screen-coord-setup' : 'screen-client-setup');
}

// Sign-up step logic
function pickRole(role) {
  S.signupRole = role;
  document.querySelectorAll('.role-pick-opt').forEach(o => o.classList.remove('selected'));
  el(`pick-${role}`).classList.add('selected');
  el('signup-next').disabled = false;
}

function toSignupStep2() {
  if (!S.signupRole) return;
  el('signup-step1').style.display = 'none';
  el('signup-step2').style.display = 'block';
  el('signup-role-label').textContent = `Joining as: ${S.signupRole}`;
}

function backToStep1() {
  el('signup-step1').style.display = 'block';
  el('signup-step2').style.display = 'none';
}


// ════════════════════════════════════════════════════════════════
//  LANDING PAGE — Stats Bar
// ════════════════════════════════════════════════════════════════

async function loadStats() {
  /*
   * INTEGRATION: GET /stats  (public, no auth)
   * Returns: { nodes_connected, total_compute, active_jobs }
   *
   * TRD v2.1 spec:
   *   nodes_connected = COUNT(DISTINCT client_id) FROM global_credits
   *                     WHERE last_session_at > NOW() - INTERVAL '7 days'
   *   total_compute   = SUM(total_samples) FROM global_credits
   *   active_jobs     = COUNT(*) FROM jobs WHERE status = 'active'
   *
   * Per PRD §5.2: show dashes if endpoint unreachable — never fabricate.
   */
  try {
    const d = await api('/stats');
    setText('stat-nodes',   fmt(d.nodes_connected ?? d.nodes   ?? 0));
    setText('stat-compute', fmtLarge(d.total_compute ?? d.total_samples ?? 0));
    setText('stat-jobs',    String(d.active_jobs ?? d.active   ?? 0));
  } catch {
    // Leave dashes as-is; backend may not be running on landing
  }
}


// ════════════════════════════════════════════════════════════════
//  COORDINATOR SETUP  →  LAUNCH DASHBOARD
// ════════════════════════════════════════════════════════════════

function launchCoordinator() {
  const ip   = el('coord-ip').value.trim()   || 'localhost';
  const port = el('coord-port').value.trim() || '5000';
  S.expectedClients = parseInt(el('coord-clients').value) || 3;
  S.sessionName     = el('coord-session-name').value.trim() || 'Training Session';

  S.coordURL = ip.startsWith('http') ? ip : `http://${ip}:${port}`;

  setText('coord-dash-url', S.coordURL.replace(/^https?:\/\//, ''));
  setText('coord-session-badge', S.sessionName || 'Local Session');

  log('coord-log', `Coordinator URL: ${S.coordURL}`, 'info');
  log('coord-log', `Expecting ${S.expectedClients} client(s)`, 'info');
  log('coord-log', `Session: ${S.sessionName}`, 'info');

  showScreen('screen-coord-dashboard');
  startDashboardPolling();
}


// ════════════════════════════════════════════════════════════════
//  COORDINATOR DASHBOARD  —  Polling
// ════════════════════════════════════════════════════════════════

function startDashboardPolling() {
  clearInterval(S.pollCoord);
  refreshClients();
  refreshCoordStatus();
  S.pollCoord = setInterval(() => { refreshClients(); refreshCoordStatus(); }, 4000);
}

async function refreshClients() {
  /*
   * INTEGRATION: GET /get_clients
   * Returns: { clients: [{ id, ip, data_received, last_seen? }] }
   *
   * TRD v2.1: client list will come from GET /status/{job_id}
   * as part of the connected_nodes array in the job status object.
   */
  try {
    const d = await api('/get_clients');
    const clients = d.clients || [];

    const tbody = el('clients-tbody');
    tbody.innerHTML = '';

    if (!clients.length) {
      tbody.innerHTML = `<tr class="empty-row"><td colspan="5">No clients connected yet — waiting…</td></tr>`;
    } else {
      clients.forEach(c => {
        const tr = document.createElement('tr');
        const hasData = c.data_received;
        const lastSeen = c.last_seen
          ? new Date(c.last_seen).toLocaleTimeString()
          : '—';
        tr.innerHTML = `
          <td class="mono">${c.id || c.client_id || '—'}</td>
          <td class="mono">${c.ip || c.ip_address || 'unknown'}</td>
          <td style="color:var(--coord)">● Active</td>
          <td style="color:${hasData ? 'var(--coord)' : 'var(--muted)'}">${hasData ? '✓ Yes' : '○ No'}</td>
          <td class="mono" style="color:var(--muted)">${lastSeen}</td>`;
        tbody.appendChild(tr);
      });
    }

    setText('client-count-label', `Connected: ${clients.length} / ${S.expectedClients} client(s)`);
    setText('coord-stat-clients', clients.length);

  } catch { /* silent */ }
}

async function refreshCoordStatus() {
  /*
   * INTEGRATION: GET /status
   * Returns: { round, round_status, global_accuracy, clients_submitted }
   *
   * TRD v2.1: GET /status/{job_id}
   * Returns: { round, round_status, accuracy, clients_submitted, leaderboard }
   */
  try {
    const path = S.jobId ? `/status/${S.jobId}` : '/status';
    const d    = await api(path);

    setText('coord-stat-round',  d.round ?? '—');
    setText('coord-stat-status', fmtStatus(d.round_status));

    if (d.global_accuracy > 0) {
      const pct = normPct(d.global_accuracy);
      setText('coord-stat-acc', pct.toFixed(1) + '%');
    }
  } catch { /* backend may not be running yet */ }
}


// ════════════════════════════════════════════════════════════════
//  NEW JOB  —  File Upload + Submit
// ════════════════════════════════════════════════════════════════

function onDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('over');
}
function onDragLeave(e) {
  e.currentTarget.classList.remove('over');
}
function onDrop(e, type) {
  e.preventDefault();
  e.currentTarget.classList.remove('over');
  const file = e.dataTransfer.files[0];
  if (file) processFile(file, type);
}
function onFileSelect(e, type) {
  const file = e.target.files[0];
  if (file) processFile(file, type);
}

function processFile(file, type) {
  if (type === 'dataset') {
    if (!file.name.endsWith('.csv')) {
      uploadMsg('dataset-msg', '✗ Must be a .csv file', 'err');
      return;
    }
    S.datasetFile = file;
    el('dataset-hint').textContent = `${file.name} — ${fmtBytes(file.size)}`;
    uploadMsg('dataset-msg', `✓ ${file.name} ready to upload`, 'ok');

  } else if (type === 'checkpoint') {
    if (!file.name.endsWith('.pt')) {
      uploadMsg('ckpt-msg', '✗ Must be a .pt file', 'err');
      return;
    }
    S.checkpointFile = file;
    el('ckpt-hint').textContent = `${file.name} — ${fmtBytes(file.size)}`;
    uploadMsg('ckpt-msg', `✓ ${file.name} — will validate on submit`, 'ok');

    // With checkpoint, 5 rounds is recommended (TRD v2.1 §3.3)
    el('job-rounds').value = 5;
    el('rounds-hint').textContent = '— 5 recommended with checkpoint';
  }
}

async function submitJob() {
  if (!S.datasetFile) {
    msg('job-msg', 'Please upload a dataset CSV first.', 'err');
    return;
  }

  const jobName = el('job-name').value.trim() || 'Unnamed Job';
  const rounds  = parseInt(el('job-rounds').value) || 5;

  // Reset state from any previous job
  S.accuracyHistory = [];

  msg('job-msg', 'Uploading dataset…', 'wait');
  el('btn-launch').disabled = true;

  try {
    /*
     * INTEGRATION — Current (v1.0):
     *   Step 1: POST /upload_dataset — FormData { file: CSV }
     *           Returns: { rows: number, filename: string }
     *   Step 2: POST /start_training — { client_count: N }
     *           Returns: { shards_prepared: N, message: string }
     *
     * TRD v2.1 (single call):
     *   POST /jobs — multipart FormData {
     *     dataset:    CSV file,
     *     checkpoint: .pt file (optional),
     *     job_name:   string,
     *     epochs:     number,
     *     rounds:     number
     *   }
     *   Returns: { job_id, status: 'queued',
     *              source: 'checkpoint'|'random_init', recommended_rounds }
     *   Then poll GET /jobs/{job_id} for live status.
     */

    // Step 1: Upload dataset
    const fd = new FormData();
    fd.append('file', S.datasetFile);
    // TRD v2.1 extras (will be part of POST /jobs body):
    // fd.append('job_name', jobName);
    // fd.append('rounds',   rounds);
    // fd.append('epochs',   el('job-epochs').value);
    // if (S.checkpointFile) fd.append('checkpoint', S.checkpointFile);

    const up = await api('/upload_dataset', { method: 'POST', body: fd });
    log('coord-log', `Dataset uploaded: ${up.rows ?? '?'} rows`, 'ok');

    msg('job-msg', `Uploaded (${up.rows ?? '?'} rows). Starting training…`, 'wait');

    // Step 2: Start training
    const start = await api('/start_training', {
      method: 'POST',
      body: {
        client_count: S.expectedClients,
        // TRD v2.1: epochs, rounds, job_name added here
      },
    });

    if (start.job_id) S.jobId = start.job_id; // capture if returned early

    const shards = start.shards_prepared ?? S.expectedClients;
    msg('job-msg', `Training started — ${shards} shards distributed`, 'ok');
    log('coord-log', `Training started: ${shards} data shards prepared`, 'ok');

    await sleep(700);

    // Clean up file references for next job submission
    S.datasetFile = null;
    S.checkpointFile = null;

    // Navigate to live view
    setText('live-job-name', jobName);
    setText('live-total',    rounds);
    showScreen('screen-live-training');

  } catch (e) {
    msg('job-msg', `Error: ${e.message}`, 'err');
    log('coord-log', `Job failed: ${e.message}`, 'err');
  } finally {
    el('btn-launch').disabled = false;
  }
}


// ════════════════════════════════════════════════════════════════
//  LIVE TRAINING VIEW
// ════════════════════════════════════════════════════════════════

function initLiveView() {
  initAccuracyChart();
  initNetworkCanvas();
  startLivePolling();
}

// ── Chart.js Accuracy Chart ───────────────────────────────────

function initAccuracyChart() {
  const canvas = el('accuracy-chart');
  if (!canvas) return;
  if (accChart) { accChart.destroy(); accChart = null; }

  /*
   * Chart config: dark-themed line chart, teal accent.
   * X axis = round number labels.  Y axis = accuracy 0–100%.
   * Updated incrementally via pushAccuracyPoint().
   */
  accChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Global Accuracy',
        data: [],
        borderColor:     '#00e5a0',
        backgroundColor: 'rgba(0,229,160,0.07)',
        borderWidth: 2,
        pointBackgroundColor: '#00e5a0',
        pointBorderColor:     '#080c10',
        pointBorderWidth: 2,
        pointRadius: 4,
        fill:    true,
        tension: 0.4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      scales: {
        x: {
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#6b7a8d', font: { family: 'Space Mono', size: 10 } },
        },
        y: {
          min: 0, max: 100,
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color: '#6b7a8d',
            font:  { family: 'Space Mono', size: 10 },
            callback: v => v + '%',
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0d1117',
          borderColor: 'rgba(0,229,160,.3)',
          borderWidth: 1,
          titleFont: { family: 'Space Mono', size: 11 },
          bodyFont:  { family: 'Space Mono', size: 11 },
          callbacks: { label: ctx => `Acc: ${ctx.parsed.y.toFixed(2)}%` },
        },
      },
    },
  });
}

function pushAccuracyPoint(round, rawAcc) {
  if (!accChart) return;
  const pct = parseFloat(normPct(rawAcc).toFixed(2));
  S.accuracyHistory.push({ round, acc: pct });
  accChart.data.labels.push(`R${round}`);
  accChart.data.datasets[0].data.push(pct);
  accChart.update();
}


// ── Network Canvas Graph ──────────────────────────────────────

/*
 * Draws coordinator (teal) at centre, client nodes (blue) arranged
 * radially. Animated dashed edges with travelling packet dots.
 * Packets flow inbound during 'active', outbound during 'aggregating'.
 */
class NetworkGraph {
  constructor(canvas) {
    this.cv  = canvas;
    this.ctx = canvas.getContext('2d');
    this.clients  = [];    // { id, x, y, tx, ty, op }  (tx/ty = target pos)
    this.packets  = [];    // { ci, dir, t, spd }
    this.time     = 0;
    this.running  = false;
    this.raf      = null;
  }

  resize() {
    const p = this.cv.parentElement;
    const r = p.getBoundingClientRect();
    // Leave 42px for the panel header
    this.cv.width  = Math.floor(r.width);
    this.cv.height = Math.max(Math.floor(r.height) - 42, 120);
    this.cx = this.cv.width  / 2;
    this.cy = this.cv.height / 2;
    this._reposition();
  }

  _reposition() {
    const n = this.clients.length;
    if (!n) return;
    const r = Math.min(this.cx, this.cy) * 0.58;
    this.clients.forEach((c, i) => {
      const a = (2 * Math.PI * i / n) - Math.PI / 2;
      c.tx = this.cx + Math.cos(a) * r;
      c.ty = this.cy + Math.sin(a) * r;
    });
  }

  setClients(ids) {
    const existing = new Set(this.clients.map(c => c.id));
    ids.forEach(id => {
      if (!existing.has(id)) {
        this.clients.push({ id, label: id, x: this.cx, y: this.cy, tx: 0, ty: 0, op: 0 });
      }
    });
    this._reposition();
    setText('node-count-label', `${ids.length} NODE${ids.length !== 1 ? 'S' : ''}`);
  }

  spawnPackets(dir) {
    // dir: 'in' = client→coord, 'out' = coord→client
    this.clients.forEach((_, i) => {
      if (Math.random() < 0.45) {
        this.packets.push({ ci: i, dir, t: 0, spd: 0.011 + Math.random() * 0.007 });
      }
    });
  }

  draw() {
    const { ctx, cv, clients, packets, time, cx, cy } = this;
    ctx.clearRect(0, 0, cv.width, cv.height);

    // Smoothly interpolate client positions & fade in
    clients.forEach(c => {
      c.x  += (c.tx - c.x)  * 0.06;
      c.y  += (c.ty - c.y)  * 0.06;
      c.op  = Math.min(1, c.op + 0.025);
    });

    // Dashed edges (animated dash offset for data-flow feel)
    clients.forEach(c => {
      if (c.op < 0.05) return;
      ctx.save();
      ctx.globalAlpha = c.op * 0.35;
      ctx.beginPath();
      ctx.setLineDash([3, 6]);
      ctx.lineDashOffset = -((time * 22) % 9);
      ctx.moveTo(cx, cy);
      ctx.lineTo(c.x, c.y);
      ctx.strokeStyle = 'rgba(0,119,255,0.55)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    });

    // Move + draw packets
    this.packets = packets.filter(p => {
      const c = clients[p.ci];
      if (!c) return false;
      const [fx, fy, tx, ty] =
        p.dir === 'in'
          ? [c.x, c.y, cx, cy]
          : [cx,  cy,  c.x, c.y];
      const px = fx + (tx - fx) * p.t;
      const py = fy + (ty - fy) * p.t;
      const col = p.dir === 'in' ? '#0077ff' : '#00e5a0';

      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.shadowBlur = 8;
      ctx.shadowColor = col;
      ctx.fill();
      ctx.shadowBlur = 0;

      p.t += p.spd;
      return p.t < 1;
    });

    // Client nodes
    clients.forEach(c => {
      if (c.op < 0.02) return;
      ctx.save();
      ctx.globalAlpha = c.op;

      // Glow halo
      const g = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, 22);
      g.addColorStop(0, 'rgba(0,119,255,.13)');
      g.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(c.x, c.y, 22, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();

      // Circle
      ctx.beginPath();
      ctx.arc(c.x, c.y, 13, 0, Math.PI * 2);
      ctx.fillStyle   = '#0d1117';
      ctx.strokeStyle = '#0077ff';
      ctx.lineWidth   = 1.5;
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = '#6b7a8d';
      ctx.font      = '9px Space Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(c.label, c.x, c.y + 26);
      ctx.restore();
    });

    // Coordinator node (centre)
    const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, 34);
    cg.addColorStop(0, 'rgba(0,229,160,.11)');
    cg.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.arc(cx, cy, 34, 0, Math.PI * 2);
    ctx.fillStyle = cg;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(cx, cy, 19, 0, Math.PI * 2);
    ctx.fillStyle   = '#0d1117';
    ctx.strokeStyle = '#00e5a0';
    ctx.lineWidth   = 2;
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#00e5a0';
    ctx.font      = 'bold 8px Space Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('COORD', cx, cy + 33);
  }

  _loop() {
    this.time += 0.016;
    this.draw();
    this.raf = requestAnimationFrame(() => this._loop());
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.resize();
    this._loop();
  }

  stop() {
    this.running = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
  }
}

function initNetworkCanvas() {
  const canvas = el('network-canvas');
  if (!canvas) return;
  if (netGraph) { netGraph.stop(); netGraph = null; }
  netGraph = new NetworkGraph(canvas);
  netGraph.start();
  if (_netResizeListener) window.removeEventListener('resize', _netResizeListener);
  _netResizeListener = () => netGraph && netGraph.resize();
  window.addEventListener('resize', _netResizeListener);
}


// ── Live Polling ──────────────────────────────────────────────

function startLivePolling() {
  clearInterval(S.pollCoord);

  let lastRound  = -1;
  let lastStatus = '';

  const tick = async () => {
    try {
      /*
       * INTEGRATION:
       * Current: GET /status
       * TRD v2.1: GET /status/{job_id}
       * Returns: {
       *   round, round_status, global_accuracy, clients_submitted,
       *   total_rounds?,
       *   leaderboard?: [{ client_id, points_earned, samples_trained }],
       *   clients?:     [{ id, ip }]
       * }
       */
      const path = S.jobId ? `/status/${S.jobId}` : '/status';
      const d    = await api(path);

      // Header readouts
      setText('live-round', d.round ?? '—');
      if (d.total_rounds) setText('live-total', d.total_rounds);
      el('live-status-pill').textContent = fmtStatus(d.round_status);

      if (d.global_accuracy > 0) {
        const pct = normPct(d.global_accuracy);
        setText('live-acc-readout', pct.toFixed(1) + '%');
      }

      // Log & chart on state change
      if (d.round !== lastRound || d.round_status !== lastStatus) {
        const r = d.round;
        const st = d.round_status;

        if (st === 'active' && r > 0)
          liveLog(`Round ${r} — clients training locally…`, 'round');
        else if (st === 'aggregating')
          liveLog(`Round ${r} — aggregating weight updates via FedAvg…`, 'info');
        else if (st === 'data_distributing')
          liveLog('Distributing data shards to connected clients…', 'info');
        else if (st === 'done') {
          liveLog('All rounds complete. Training finished.', 'done');
          liveLog(`Final accuracy: ${normPct(d.global_accuracy).toFixed(2)}%`, 'done');
          clearInterval(S.pollCoord);
          S.pollCoord = null;
          onTrainingDone(d);
        }

        // Push new accuracy point to chart
        if (d.global_accuracy > 0 && r !== lastRound && r > 0) {
          pushAccuracyPoint(r, d.global_accuracy);
        }

        lastRound  = d.round;
        lastStatus = d.round_status;
      }

      // Update leaderboard panel
      if (d.leaderboard && d.leaderboard.length) updateLiveLeaderboard(d.leaderboard);

      // Update network graph clients + packets
      if (netGraph) {
        const ids = (d.clients || []).map(c => c.id || c.client_id);
        if (ids.length) {
          netGraph.setClients(ids);
          if (d.round_status === 'active')      netGraph.spawnPackets('in');
          if (d.round_status === 'aggregating') netGraph.spawnPackets('out');
        }
      }

    } catch { /* silent — backend busy or not yet running */ }
  };

  tick();
  S.pollCoord = setInterval(tick, 5000);
}

function liveLog(msg, type = 'info') {
  log('live-log', msg, type);
}

function updateLiveLeaderboard(lb) {
  const tbody = el('live-lb-body');
  if (!tbody) return;
  tbody.innerHTML = '';
  lb.forEach((e, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono" style="color:var(--muted)">#${i + 1}</td>
      <td class="mono">${e.client_id}</td>
      <td class="mono">${(e.samples_trained || 0).toLocaleString()}</td>
      <td class="mono" style="color:var(--client)">${e.points_earned || 0}</td>`;
    tbody.appendChild(tr);
  });
}

function onTrainingDone(d) {
  // Populate results screen
  const acc = d.global_accuracy
    ? normPct(d.global_accuracy).toFixed(1) + '%'
    : '—';
  setText('res-acc',     acc);
  setText('res-rounds',  d.round ?? '—');
  setText('res-clients', d.clients_submitted ?? '—');
  setText('res-samples', '—');

  // Auto-navigate after short pause
  setTimeout(() => {
    showScreen('screen-job-results');
    buildResultsChart();
  }, 2200);
}

function buildResultsChart() {
  const canvas = el('results-chart');
  if (!canvas || !S.accuracyHistory.length) return;
  if (resultsChart) { resultsChart.destroy(); resultsChart = null; }

  resultsChart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels:   S.accuracyHistory.map(p => `R${p.round}`),
      datasets: [{
        data:            S.accuracyHistory.map(p => p.acc),
        borderColor:     '#00e5a0',
        backgroundColor: 'rgba(0,229,160,0.07)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#00e5a0',
        pointRadius: 3,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#6b7a8d', callback: v => v + '%' } },
        x: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#6b7a8d' } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function downloadModel() {
  /*
   * INTEGRATION (TRD v2.1): GET /jobs/{job_id}/model → .pt binary
   * Current: no download endpoint yet — model lives at ./global_model.pt
   * on coordinator server. Implement a simple /download_model endpoint
   * in Flask that sends the file:
   *   @app.route('/download_model')
   *   def download_model():
   *       return send_file('global_model.pt', as_attachment=True)
   */
  const path = S.jobId ? `/jobs/${S.jobId}/model` : '/download_model';
  window.location.href = S.coordURL + path;
}

function downloadReport() {
  // Client-side JSON report from in-memory accuracy history
  const report = {
    generated:        new Date().toISOString(),
    job_id:           S.jobId,
    coordinator_url:  S.coordURL,
    accuracy_history: S.accuracyHistory,
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const a = Object.assign(document.createElement('a'), {
    href:     URL.createObjectURL(blob),
    download: `netneutral_report_${Date.now()}.json`,
  });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 100);
}


// ════════════════════════════════════════════════════════════════
//  CLIENT SETUP  —  Session Browser + Connect
// ════════════════════════════════════════════════════════════════

function switchTab(name) {
  ['browse', 'manual'].forEach(n => {
    el(`tab-${n}`).classList.toggle('active', n === name);
    el(`tab-pane-${n}`).style.display = n === name ? 'block' : 'none';
  });
}

async function scanSessions() {
  const listEl = el('sessions-list');
  listEl.innerHTML = `<div class="sessions-empty mono text-muted">Scanning network…</div>`;

  try {
    /*
     * INTEGRATION (Phase 2):
     * GET /api/lobby?subnet={subnet}
     * Discovery host = current page origin OR the URL in manual connect input.
     * Returns: [{
     *   session_id, session_name, coordinator_ip, coordinator_port,
     *   round_status, connected_clients, max_clients, base_url, last_heartbeat
     * }]
     * Stale sessions (last_heartbeat > SESSION_STALE_SECS ago) are excluded server-side.
     * For LAN-only testing without Supabase, add ?subnet=192.168.1
     *
     * Note: The lobby is served by the coordinator's Flask server.
     * The "discovery host" is any coordinator URL the client already knows about.
     * In production, a central Supabase registry removes this bootstrapping problem.
     */
    const discoveryBase = S.coordURL !== window.location.origin
      ? S.coordURL
      : window.location.origin;

    const data = await fetch(`${discoveryBase}/api/lobby`).then(r => r.json());
    const sessions = data.sessions || (Array.isArray(data) ? data : []);
    renderSessions(sessions);

  } catch {
    listEl.innerHTML = `
      <div class="sessions-empty mono text-muted">
        No sessions found. Use Manual Connect → to enter a URL directly.
      </div>`;
  }
}

function renderSessions(sessions) {
  const listEl = el('sessions-list');
  if (!sessions.length) {
    listEl.innerHTML = `<div class="sessions-empty mono text-muted">No active sessions on this network.</div>`;
    return;
  }
  listEl.innerHTML = '';
  sessions.forEach(s => {
    const full   = s.connected_clients >= s.max_clients;
    const active = s.round_status !== 'waiting_for_clients';
    const card   = document.createElement('div');
    card.className = 'session-card';
    card.innerHTML = `
      <div class="sc-info">
        <span class="sc-name">${s.session_name || 'Unnamed Session'}</span>
        <span class="sc-meta">${s.coordinator_ip}:${s.coordinator_port || 5000} · ${s.connected_clients}/${s.max_clients} clients</span>
      </div>
      <span class="sc-status ${active ? 'active' : 'waiting'}">${fmtStatus(s.round_status)}</span>`;
    if (!full) {
      card.onclick = () => pickSession(card, s);
    } else {
      card.style.opacity = '.45';
      card.style.cursor  = 'not-allowed';
    }
    listEl.appendChild(card);
  });
}

function pickSession(cardEl, session) {
  document.querySelectorAll('.session-card').forEach(c => c.classList.remove('picked'));
  cardEl.classList.add('picked');
  const url = session.base_url
    || `http://${session.coordinator_ip}:${session.coordinator_port || 5000}`;
  S.selectedSession = url;
  el('manual-coord-url').value = url;
}

async function connectClient() {
  // Resolve coordinator URL from active tab
  const tabEl = document.querySelector('.tab-btn.active');
  if (!tabEl) { msg('client-setup-msg', 'Select a connection method.', 'err'); return; }
  const tab = tabEl.id;
  let coordURL = '';

  if (tab === 'tab-browse' && S.selectedSession) {
    coordURL = S.selectedSession;
  } else {
    coordURL = el('manual-coord-url').value.trim();
    if (!coordURL) {
      msg('client-setup-msg', 'Enter a coordinator URL or browse sessions.', 'err');
      return;
    }
  }

  if (!coordURL.startsWith('http')) coordURL = 'http://' + coordURL;
  S.coordURL  = coordURL;
  S.clientId  = el('client-id-select').value || 'client_A';

  msg('client-setup-msg', '', '');

  // Show discovery animation
  setText('disc-url', coordURL);
  showScreen('screen-client-discovery');
  await runDiscovery(coordURL);
}

async function runDiscovery(coordURL) {
  const statusEl = el('disc-status');
  const steps    = [
    'Contacting coordinator…',
    'Validating session…',
    'Registering node…',
    'Confirming handshake…',
  ];

  for (let i = 0; i < steps.length - 1; i++) {
    statusEl.textContent = steps[i];
    await sleep(550);
  }

  try {
    /*
     * INTEGRATION:
     * Current (v1.0): POST /register
     *   Body: { client_id: string, ip_address: string }
     *   Returns: { status: 'registered' }
     *
     * TRD v2.1: POST /register_node (JWT required)
     *   Body: { machine_id, os, ram_gb, has_gpu, gpu_name }
     *   Returns: { node_id, status: 'registered' }
     *   machine_id for web client = Supabase user.id or session UUID.
     *
     * Note: In TRD v2.1, credits.py register_client() is called server-side
     * which inserts into the SQLite clients table (TRD §3.1).
     */
    await api('/register', {
      method: 'POST',
      body: {
        client_id:  S.clientId,
        ip_address: 'web-frontend',
        // TRD v2.1 extras:
        // machine_id: S.clientId,
        // os:         navigator.platform,
        // ram_gb:     null,
        // has_gpu:    false,
      },
    });

    statusEl.textContent = steps[steps.length - 1];
    await sleep(450);

    // Transition to client dashboard
    setText('client-dash-id', S.clientId);
    log('client-log', `Registered as ${S.clientId}`, 'ok');
    log('client-log', `Coordinator: ${coordURL}`, 'info');
    log('client-log', 'Waiting for training rounds to begin…', 'info');
    showScreen('screen-client-dashboard');
    startClientPolling();

  } catch (e) {
    statusEl.textContent = `Connection failed: ${e.message}`;
    const dotsEl = document.querySelector('.disc-dots');
    if (dotsEl) dotsEl.style.display = 'none';
    await sleep(2000);
    showScreen('screen-client-setup');
    msg('client-setup-msg', `Connection failed: ${e.message}`, 'err');
  }
}


// ════════════════════════════════════════════════════════════════
//  CLIENT DASHBOARD  —  Polling
// ════════════════════════════════════════════════════════════════

function startClientPolling() {
  clearInterval(S.pollClient);
  let lastRound = -1;
  let totalCredits = 0;

  const tick = async () => {
    try {
      /*
       * INTEGRATION: GET /api/client_status/{client_id}
       * Returns: {
       *   current_round: number,
       *   total_rounds:  number,
       *   round_status:  string,
       *   total_credits: number,
       *   global_accuracy: number,
       *   round_history?: [{ round, credits, samples, accuracy }]
       * }
       *
       * TRD v2.1: per-client data will be embedded in GET /status/{job_id}
       * response — query that instead and extract the matching client entry.
       */
      const d = await api(`/api/client_status/${S.clientId}`);

      setText('client-stat-credits', (d.total_credits || 0).toLocaleString());
      setText('client-stat-round',   `${d.current_round || '—'} / ${d.total_rounds || '—'}`);
      setText('client-stat-status',  fmtStatus(d.round_status));
      setText('client-rounds-label', `${d.current_round || 0} / ${d.total_rounds || '?'} rounds`);

      if (d.global_accuracy > 0) {
        setText('client-stat-acc', normPct(d.global_accuracy).toFixed(1) + '%');
      }

      const badge = el('client-conn-badge');
      if (d.round_status === 'active') {
        badge.textContent = 'Training';
        badge.className   = 'badge badge-coord';
      } else if (d.round_status === 'done') {
        badge.textContent = 'Complete';
        badge.className   = 'badge badge-client';
        log('client-log', `Training complete! Final credits: ${d.total_credits}`, 'done');
        clearInterval(S.pollClient);
        S.pollClient = null;
      }

      // Add timeline entry when round changes
      if (d.current_round !== lastRound && d.current_round > 0) {
        const earned = d.total_credits - totalCredits;
        addTimelineItem(d.current_round, d.round_status, earned);
        if (d.round_status === 'active')
          log('client-log', `Round ${d.current_round} — training locally…`, 'round');
        lastRound    = d.current_round;
        totalCredits = d.total_credits;
      }

    } catch { /* silent */ }
  };

  tick();
  S.pollClient = setInterval(tick, 3000);
}

function addTimelineItem(round, status, creditsEarned) {
  const tl = el('epoch-timeline');
  // Clear empty state
  const empty = tl.querySelector('.timeline-empty');
  if (empty) empty.remove();
  // Demote previous active item to done
  tl.querySelectorAll('.tl-item.active').forEach(i => i.classList.replace('active', 'done'));

  const item = document.createElement('div');
  item.className = `tl-item ${status === 'done' ? 'done' : 'active'} slide-up`;
  item.innerHTML = `
    <div class="tl-round mono">ROUND ${round}</div>
    <div class="tl-stat">${fmtStatus(status)}</div>
    <div class="tl-credits">${creditsEarned > 0 ? '+' + creditsEarned + ' pts' : ''}</div>`;
  tl.appendChild(item);
}


// ════════════════════════════════════════════════════════════════
//  AGENT SETUP GUIDE
// ════════════════════════════════════════════════════════════════

function selectOS(os) {
  ['windows','mac','linux'].forEach(o => {
    el(`os-${o}`).classList.toggle('active', o === os);
  });

  const url = S.coordURL || 'http://<coordinator-ip>:5000';
  const cid = S.clientId || 'client_A';

  /*
   * INTEGRATION (TRD v2.1 §6):
   * install_windows.bat  → registers Task Scheduler job (runs agent.py on login)
   * install_mac.sh       → registers launchd plist
   * install_linux.sh     → registers systemd user service
   *
   * Idle detection thresholds (idle_detector.py):
   *   CPU  < 20%   for 60 consecutive seconds
   *   GPU  < 15%   for 60 consecutive seconds  (Windows + Linux)
   *
   * Agent auth: agent stores Supabase JWT locally after first sign-in
   * via web platform. Token is refreshed automatically by supabase-py.
   */
  const guides = {
    windows: `
      <div class="setup-card hud-panel client-panel">
        <h3 class="card-title">Windows Installation</h3>
        <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
          Requirements: Python 3.12 · PyTorch 2.3 · psutil · GPUtil
        </p>
        <div class="agent-step"><span class="agent-step-n">1.</span> Clone the repository</div>
        <pre class="codeblock">git clone https://github.com/your-org/net-neutral-ai.git
cd net-neutral-ai/agent</pre>
        <div class="agent-step"><span class="agent-step-n">2.</span> Install dependencies</div>
        <pre class="codeblock">pip install -r requirements.txt</pre>
        <div class="agent-step"><span class="agent-step-n">3.</span> Set coordinator URL in config.py</div>
        <pre class="codeblock">COORDINATOR_URL = "${url}"
CLIENT_ID       = "${cid}"</pre>
        <div class="agent-step"><span class="agent-step-n">4.</span> Run manually</div>
        <pre class="codeblock">python agent.py --client_id ${cid} --coordinator_url ${url}</pre>
        <div class="agent-step"><span class="agent-step-n">5.</span> (Optional) Install as background service via Task Scheduler</div>
        <pre class="codeblock">install\\install_windows.bat</pre>
      </div>`,

    mac: `
      <div class="setup-card hud-panel client-panel">
        <h3 class="card-title">macOS Installation</h3>
        <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
          Requirements: Python 3.12 · PyTorch 2.3 · psutil
        </p>
        <div class="agent-step"><span class="agent-step-n">1.</span> Clone & install</div>
        <pre class="codeblock">git clone https://github.com/your-org/net-neutral-ai.git
cd net-neutral-ai/agent
pip3 install -r requirements.txt</pre>
        <div class="agent-step"><span class="agent-step-n">2.</span> Run manually</div>
        <pre class="codeblock">python3 agent.py --client_id ${cid} --coordinator_url ${url}</pre>
        <div class="agent-step"><span class="agent-step-n">3.</span> (Optional) Install as launchd service</div>
        <pre class="codeblock">bash install/install_mac.sh
launchctl load ~/Library/LaunchAgents/ai.netneutral.agent.plist</pre>
      </div>`,

    linux: `
      <div class="setup-card hud-panel client-panel">
        <h3 class="card-title">Linux Installation</h3>
        <p style="color:var(--muted);font-size:13px;margin-bottom:16px">
          Requirements: Python 3.12 · PyTorch 2.3 · psutil · nvidia-smi (for GPU)
        </p>
        <div class="agent-step"><span class="agent-step-n">1.</span> Clone & install</div>
        <pre class="codeblock">git clone https://github.com/your-org/net-neutral-ai.git
cd net-neutral-ai/agent
pip3 install -r requirements.txt</pre>
        <div class="agent-step"><span class="agent-step-n">2.</span> Run manually</div>
        <pre class="codeblock">python3 agent.py --client_id ${cid} --coordinator_url ${url}</pre>
        <div class="agent-step"><span class="agent-step-n">3.</span> (Optional) Install as systemd user service</div>
        <pre class="codeblock">bash install/install_linux.sh
systemctl --user enable netneutral-agent
systemctl --user start netneutral-agent</pre>
      </div>`,
  };

  el('agent-guide').innerHTML = guides[os] || '';
}


// ════════════════════════════════════════════════════════════════
//  PUBLIC LEADERBOARD
// ════════════════════════════════════════════════════════════════

async function loadLeaderboard() {
  const tbody = el('lb-body');
  if (!tbody) return;
  tbody.innerHTML = `<tr class="empty-row"><td colspan="6">Loading…</td></tr>`;

  try {
    /*
     * INTEGRATION: GET /leaderboard  (public, no auth — TRD v2.1 §4.2)
     *
     * Current: queries local SQLite via credits.py get_leaderboard()
     *   Returns: [{ client_id, total_points, total_samples, rounds_participated }]
     *
     * TRD v2.1: queries Supabase global_credits after end-of-session sync.
     *   Returns: [{ client_id, total_points, total_samples, total_rounds, last_session_at }]
     *   Ordered by total_points DESC, top N entries.
     *
     * credits.py sync_to_supabase() / increment_credits() RPC is called
     * automatically when round_status transitions to 'done'.
     */
    const d       = await api('/leaderboard');
    const entries = d.leaderboard || (Array.isArray(d) ? d : []);

    if (!entries.length) {
      tbody.innerHTML = `<tr class="empty-row"><td colspan="6">No data yet — complete a training session to appear here.</td></tr>`;
      return;
    }

    tbody.innerHTML = '';
    entries.forEach((e, i) => {
      const rank = i + 1;
      const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : `#${rank}`;
      const rankClass = rank === 1 ? 'rank-gold' : rank === 2 ? 'rank-silver' : rank === 3 ? 'rank-bronze' : '';
      const lastActive = (e.last_session_at || e.last_active)
        ? new Date(e.last_session_at || e.last_active).toLocaleDateString()
        : '—';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono ${rankClass}">${medal}</td>
        <td class="mono">${e.client_id}</td>
        <td class="mono" style="color:var(--client)">${(e.total_points || e.points || 0).toLocaleString()}</td>
        <td class="mono">${(e.total_samples || e.samples || 0).toLocaleString()}</td>
        <td class="mono">${e.total_rounds || e.rounds_participated || 0}</td>
        <td class="mono text-muted">${lastActive}</td>`;
      tbody.appendChild(tr);
    });

  } catch {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="6">Could not load leaderboard — is the coordinator running?</td></tr>`;
  }
}


// ════════════════════════════════════════════════════════════════
//  UTILITIES
// ════════════════════════════════════════════════════════════════

// Shorthand element getter
function el(id) { return document.getElementById(id); }

// Set text content safely
function setText(id, val) {
  const e = el(id);
  if (e) e.textContent = val;
}

// Append log line to a log-box container
function log(boxId, text, type = 'info') {
  const box = el(boxId);
  if (!box) return;
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
  line.textContent = `[${ts}] ${text}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
  // Cap at 300 lines
  const lines = box.querySelectorAll('.log-line');
  if (lines.length > 300) lines[0].remove();
}

// Set form message
function msg(id, text, type) {
  const e = el(id);
  if (!e) return;
  e.textContent = text;
  e.className   = `form-msg ${type}`;
}

// Set upload message
function uploadMsg(id, text, type) {
  const e = el(id);
  if (!e) return;
  e.textContent = text;
  e.className   = `upload-msg ${type}`;
}

// Format status strings for display
function fmtStatus(s) {
  const map = {
    waiting_for_clients: 'Waiting for clients',
    data_distributing:   'Distributing data',
    active:              'Training active',
    aggregating:         'Aggregating',
    done:                'Complete',
  };
  return map[s] ?? (s ? s.replace(/_/g, ' ') : 'Idle');
}

// Normalise accuracy to 0–100 range
function normPct(v) {
  return v <= 1 ? v * 100 : v;
}

// Format large numbers (1234567 → "1.2M")
function fmtLarge(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

// Format number with commas
function fmt(n) { return Number(n).toLocaleString(); }

// Format byte size
function fmtBytes(b) {
  if (b < 1024)        return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

// Promise-based delay
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }


// ════════════════════════════════════════════════════════════════
//  INITIALISATION
// ════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  showScreen('screen-landing');

  /*
   * TODO (TRD v2.1): Check for existing Supabase session on page load.
   *
   *   (async () => {
   *     const sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
   *     const { data: { session } } = await sb.auth.getSession();
   *     if (session) {
   *       S.sbSession = session;
   *       const { data: profile } = await sb.from('profiles')
   *         .select('role').eq('id', session.user.id).single();
   *       S.role = profile?.role;
   *       if (S.role === 'coordinator') showScreen('screen-coord-setup');
   *       else if (S.role === 'client')  showScreen('screen-client-setup');
   *     }
   *   })();
   */
});
 
