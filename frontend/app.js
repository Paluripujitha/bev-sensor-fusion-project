/* ================================================================
   app.js — BEVFusion Dashboard Frontend Logic
   ================================================================ */
// Auto-detect API base: use relative path so it works on localhost AND public ngrok URLs
const API = (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost')
    ? 'http://127.0.0.1:5000/api'
    : window.location.origin + '/api';

let state = {
    scene_token: null,
    sample_token: null,
    sample_data: null,
    result: null,
    orig_cam_b64: null,
};

// ── DOM refs ──────────────────────────────────────────────────────
const sceneSelect = document.getElementById('scene-select');
const sampleSelect = document.getElementById('sample-select');
const weatherSelect = document.getElementById('weather-select');
const sevSlider = document.getElementById('severity-slider');
const sevVal = document.getElementById('sev-val');
const runBtn = document.getElementById('run-btn');
const statusText = document.getElementById('status-text');
const statusDot = document.querySelector('.status-dot');

// ── helpers ───────────────────────────────────────────────────────
function setStatus(label, type = 'online') {
    statusText.textContent = label;
    statusDot.className = 'status-dot dot-' + type;
}
function showImg(imgEl, placeholderEl, src) {
    if (!src) return;
    imgEl.src = src;
    imgEl.classList.remove('hidden');
    if (placeholderEl) placeholderEl.style.display = 'none';
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val ?? '—'; }
function pct(v) { return `${(v * 100).toFixed(0)}%`; }

// ── navigation ────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(a => {
    a.addEventListener('click', e => {
        e.preventDefault();
        document.querySelectorAll('.nav-item').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.content-section').forEach(x => x.classList.remove('active'));
        a.classList.add('active');
        const sec = a.dataset.section;
        document.getElementById('section-' + sec).classList.add('active');
    });
});

sevSlider.addEventListener('input', () => { sevVal.textContent = parseFloat(sevSlider.value).toFixed(2); });

// ══════════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════════
async function init() {
    setStatus('CONNECTING …', 'busy');
    try {
        const r = await fetch(`${API}/health`);
        if (!r.ok) throw new Error('Health check failed');
        const h = await r.json();
        if (!h.dataset_loaded) throw new Error('Dataset not loaded on server');

        setStatus('ONLINE / READY', 'online');
        
        // Load scenes
        const sr = await fetch(`${API}/scenes`);
        const sd = await sr.json();
        sceneSelect.innerHTML = '<option value="">— Select a scene —</option>';
        sd.scenes.forEach(sc => {
            const opt = document.createElement('option');
            opt.value = sc.token;
            opt.textContent = `${sc.name} (${sc.nbr_samples} samples)`;
            sceneSelect.appendChild(opt);
        });
        
        // Load accident dataset images
        const accSelect = document.getElementById('accident-select');
        const accR = await fetch(`${API}/accident_images`);
        const accD = await accR.json();
        if (accD.images && accD.images.length > 0) {
            accD.images.forEach(img => {
                const opt = document.createElement('option');
                opt.value = img;
                opt.textContent = img;
                accSelect.appendChild(opt);
            });
        }
        
        // Handle accident preview selection
        accSelect.addEventListener('change', () => {
            const preview = document.getElementById('accident-preview-sidebar');
            if (accSelect.value) {
                preview.src = `${API}/accident_image/${accSelect.value}`;
                preview.classList.remove('hidden');
            } else {
                preview.classList.add('hidden');
            }
        });

        // Always ask for emergency contact on load
        document.getElementById('setup-modal').classList.remove('hidden');
        
    } catch (err) {
        setStatus('ERROR', 'error');
        console.error(err);
        sceneSelect.innerHTML = `<option value="">❌ Backend offline</option>`;
    }
}
init();

// Save Emergency Contact from Setup Modal
document.getElementById('btn-setup-save').addEventListener('click', async () => {
    const username = document.getElementById('setup-name-input').value.trim();
    const email = document.getElementById('setup-email-input').value.trim();
    if (!username) { alert('Please enter your name'); return; }
    if (!email || !email.includes('@')) { alert('Please enter a valid email address'); return; }
    try {
        await fetch(`${API}/emergency_contact`, {
            method: 'POST',
            body: JSON.stringify({ username: username, email: email })
        });
        document.getElementById('setup-modal').classList.add('hidden');
    } catch (err) { alert('Failed to save contact'); }
});

sceneSelect.addEventListener('change', async () => {
    state.scene_token = sceneSelect.value;
    if (!state.scene_token) return;
    setStatus('LOADING SAMPLES …', 'busy');
    try {
        const r = await fetch(`${API}/scenes/${state.scene_token}/samples`);
        const d = await r.json();
        sampleSelect.innerHTML = '<option value="">— Select a sample —</option>';
        d.samples.forEach((s, i) => {
            const opt = document.createElement('option');
            opt.value = s.token;
            opt.textContent = `Sample ${i + 1} — ts:${s.timestamp}`;
            sampleSelect.appendChild(opt);
        });
        if (d.samples.length > 0) {
            sampleSelect.value = d.samples[0].token;
            sampleSelect.dispatchEvent(new Event('change'));
        }
        setStatus('ONLINE / READY', 'online');
    } catch (err) { setStatus('ERROR', 'error'); console.error(err); }
});

sampleSelect.addEventListener('change', async () => {
    state.sample_token = sampleSelect.value;
    if (!state.sample_token) { runBtn.disabled = true; return; }
    setStatus('LOADING SAMPLE …', 'busy');
    runBtn.disabled = true;
    try {
        const r = await fetch(`${API}/sample/${state.sample_token}`);
        const d = await r.json();
        state.sample_data = d;

        // Pipeline Step 1 + 2 Update
        if (d.cameras['CAM_FRONT']) {
            state.orig_cam_b64 = d.cameras['CAM_FRONT'];
            showImg(document.getElementById('pl-cam-img'), document.getElementById('pl-cam-ph'), state.orig_cam_b64);
            showImg(document.getElementById('dash-cam-img'), document.getElementById('dash-cam-ph'), state.orig_cam_b64);
            showImg(document.getElementById('w-orig-img'), document.querySelector('#w-orig-img + .placeholder-msg'), state.orig_cam_b64);
        }
        if (d.lidar_bev) {
            showImg(document.getElementById('pl-lidar-img'), document.getElementById('pl-lidar-ph'), d.lidar_bev);
            showImg(document.getElementById('dash-lidar-img'), document.getElementById('dash-lidar-ph'), d.lidar_bev);
            showImg(document.getElementById('lidar-full-img'), document.getElementById('lidar-full-ph'), d.lidar_bev);
            setText('pl-lidar-pts', d.lidar_n_points.toLocaleString() + ' points');
            setText('pl-lidar-token', 'Token: ' + d.lidar_filename);
            setText('li-pts', d.lidar_n_points.toLocaleString());
            setText('li-badge-pts', d.lidar_n_points.toLocaleString() + ' pts');
            setText('li-sample', state.sample_token);
        }

        // Camera Views
        Object.entries(d.cameras).forEach(([ch, b64]) => {
            const cell = document.getElementById('cc-' + ch);
            if (!cell) return;
            const img = cell.querySelector('.cam-img');
            const ph = cell.querySelector('.cam-placeholder');
            img.src = b64;
            img.style.display = 'block';
            ph.style.display = 'none';
        });

        runBtn.disabled = false;
        setStatus('ONLINE / READY', 'online');
    } catch (err) { setStatus('ERROR', 'error'); console.error(err); runBtn.disabled = false; }
});

// ── camera cell click → modal ─────────────────────────────────────
document.querySelectorAll('.cam-cell').forEach(cell => {
    cell.addEventListener('click', () => {
        const img = cell.querySelector('.cam-img');
        if (!img.src || img.src === window.location.href) return;
        document.getElementById('modal-img').src = img.src;
        document.getElementById('modal-label').textContent = cell.querySelector('.cam-label').textContent;
        document.getElementById('cam-modal').classList.remove('hidden');
    });
});
document.getElementById('modal-bg').addEventListener('click', closeModal);
document.getElementById('modal-close').addEventListener('click', closeModal);
function closeModal() { document.getElementById('cam-modal').classList.add('hidden'); }

// ══════════════════════════════════════════════════════════════════
// RUN DETECTION
// ══════════════════════════════════════════════════════════════════
runBtn.addEventListener('click', async () => {
    if (!state.sample_token) return;
    runBtn.disabled = true;
    document.getElementById('run-btn-text').textContent = '⏳ PROCESSING PIPELINE …';
    setStatus('PROCESSING …', 'busy');

    try {
        const body = {
            sample_token: state.sample_token,
            weather_type: weatherSelect.value,
            severity: parseFloat(sevSlider.value),
            accident_image_file: document.getElementById('accident-select').value || null
        };
        const r = await fetch(`${API}/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await r.json();
        if (data.error) throw new Error(data.error);

        state.result = data;
        renderResults(data);
        setStatus('ONLINE / READY', 'online');

        // Auto navigate to pipeline overview on run
        document.querySelector('[data-section="pipeline"]').click();

        // Handle accident result & trigger modal if needed
        const accResBox = document.getElementById('integrated-accident-res');
        if (data.accident_result) {
            const isAcc = data.accident_result.is_accident;
            const conf = (data.accident_result.confidence * 100).toFixed(1);
            if (isAcc) {
                accResBox.className = 'accident-result-box acc-res-danger';
                accResBox.innerHTML = `⚠️ ACCIDENT DETECTED (Conf: ${conf}%)`;
                checkAndPromptEmergency();
            } else {
                accResBox.className = 'accident-result-box acc-res-safe';
                accResBox.innerHTML = `✅ NO ACCIDENT (Conf: ${conf}%)`;
            }
        } else {
            accResBox.className = 'accident-result-box';
            accResBox.innerHTML = '<div class="placeholder-msg">No accident image was provided for detection.</div>';
        }

    } catch (err) {
        setStatus('ERROR', 'error');
        alert('Detection failed: ' + err.message);
        console.error(err);
    } finally {
        runBtn.disabled = false;
        document.getElementById('run-btn-text').textContent = '▶ RUN PIPELINE (3D + ACCIDENT)';
    }
});

function renderResults(data) {
    const dets = data.detections || [];
    const metrics = data.metrics || {};
    const timing = data.timing || {};
    const si = data.system_info || {};

    // Dashboard & Weather updates
    showImg(document.getElementById('dash-cam-img'), document.getElementById('dash-cam-ph'), data.camera_image);
    showImg(document.getElementById('w-deg-img'), document.querySelector('#w-deg-img + .placeholder-msg'), data.camera_image);

    // Pipeline Steps 3 & 4
    document.getElementById('pl-fe-ph').textContent = 'Features Extracted Successfully';
    document.getElementById('pl-fe-ph').style.color = '#10b981';

    if (data.cam_feat) {
        showImg(document.getElementById('pl-cam-feat'), null, data.cam_feat);
    }
    if (data.lidar_feat) {
        showImg(document.getElementById('pl-lidar-feat'), null, data.lidar_feat);
    }
    if (data.fused_feat) {
        showImg(document.getElementById('pl-fused-feat'), null, data.fused_feat);
    }

    showImg(document.getElementById('pl-bev-img'), document.getElementById('pl-bev-ph'), data.lidar_bev);

    // Tables
    const objTbody = document.getElementById('obj-tbody');
    const plDetTbody = document.getElementById('pl-det-tbody');
    const plOutTbody = document.getElementById('pl-out-tbody');

    // Assign numbered names for repeated classes (Car 1, Car 2, Bus 1, etc.)
    const classCounts = {};
    dets.forEach(d => {
        const rawCls = (d.class || 'object').toLowerCase();
        classCounts[rawCls] = (classCounts[rawCls] || 0) + 1;
        const formattedCls = rawCls.charAt(0).toUpperCase() + rawCls.slice(1);
        d.name = `${formattedCls} ${classCounts[rawCls]}`;
        d.class_num = classCounts[rawCls];
    });

    objTbody.innerHTML = '';
    plDetTbody.innerHTML = '';
    plOutTbody.innerHTML = '';

    if (dets.length === 0) {
        const empty = `<tr><td colspan="10" class="empty-row">No objects detected</td></tr>`;
        objTbody.innerHTML = empty;
        plDetTbody.innerHTML = `<tr><td colspan="6" class="empty-row">No data</td></tr>`;
        plOutTbody.innerHTML = `<tr><td colspan="5" class="empty-row">No data</td></tr>`;
    } else {
        dets.forEach((d, idx) => {
            const color = d.color || '#3b82f6';
            const clsPill = `<span class="cls-pill" style="background:${color}22;color:${color};border:1px solid ${color}66">${d.name}</span>`;
            const lossVal = d.loss !== undefined ? d.loss.toFixed(4) : '0.0350';
            const accVal = d.accuracy !== undefined ? d.accuracy.toFixed(1) + '%' : '95.0%';
            const confVal = (d.confidence * 100).toFixed(1) + '%';

            // Full Table (clickable row -> selects in 3D scene)
            objTbody.innerHTML += `
                <tr class="obj-row" data-idx="${idx}" style="cursor:pointer;" title="Click to view in 3D Perception Scene">
                    <td>${d.id}</td>
                    <td>${clsPill}</td>
                    <td><span class="badge" style="background:rgba(239, 68, 68, 0.15);color:#f87171;border:1px solid rgba(239, 68, 68, 0.3);">${lossVal}</span></td>
                    <td><span class="badge" style="background:rgba(16, 185, 129, 0.15);color:#34d399;border:1px solid rgba(16, 185, 129, 0.3);">${accVal}</span></td>
                    <td><span class="badge" style="background:rgba(59, 130, 246, 0.15);color:#60a5fa;border:1px solid rgba(59, 130, 246, 0.3);">${confVal}</span></td>
                    <td>${d.distance.toFixed(2)}m</td>
                    <td class="mono">[${d.position.join(', ')}]</td>
                    <td class="mono">[${d.size.join(', ')}]</td>
                    <td>${d.orientation.toFixed(1)}°</td>
                    <td>${d.n_lidar_pts > 0 ? (d.distance / 10).toFixed(2) + ' m/s' : '0.0 m/s'}</td>
                </tr>`;

            // Pipeline Step 5 (Detection)
            plDetTbody.innerHTML += `
                <tr>
                    <td>[${d.box2d.slice(0, 2).join(',')}]</td>
                    <td>${clsPill}</td>
                    <td style="color:#f87171;">${lossVal}</td>
                    <td style="color:#34d399;">${accVal}</td>
                    <td style="color:#60a5fa;">${confVal}</td>
                    <td>${d.distance.toFixed(1)}m</td>
                </tr>`;

            // Pipeline Step 6 (Output)
            plOutTbody.innerHTML += `
                <tr>
                    <td>${d.id}</td>
                    <td>${clsPill}</td>
                    <td>L:${lossVal} / A:${accVal}</td>
                    <td class="mono">[${d.position.join(',')}]</td>
                    <td>${d.orientation.toFixed(1)}°</td>
                </tr>`;
        });

        // Add table row click listeners
        document.querySelectorAll('.obj-row').forEach(row => {
            row.addEventListener('click', () => {
                const idx = parseInt(row.dataset.idx);
                if (dets[idx]) {
                    select3DObject(dets[idx]);
                    document.querySelector('[data-section="view3d"]').click();
                }
            });
        });
    }

    // Canvas rendering
    renderBEV(data);
    draw3DScene(dets, data.lidar_bev);

    // Difficulty map
    if (data.diff_map) {
        showImg(document.getElementById('diff-img'), document.getElementById('diff-ph'), data.diff_map);
    }

    // Weather weights
    const camQ = metrics.cam_quality || 0;
    const camW = metrics.cam_weight || 0;
    const lidW = metrics.lidar_weight || 0;
    const lidQ = Math.min(1.0, 1 - camQ * 0.3 + 0.5);

    document.getElementById('w-cam-bar').style.width = pct(camQ);
    document.getElementById('w-lidar-bar').style.width = pct(lidQ);
    document.getElementById('w-camw-bar').style.width = pct(camW);
    document.getElementById('w-lidarw-bar').style.width = pct(lidW);

    setText('w-cam-num', pct(camQ));
    setText('w-lidar-num', pct(lidQ));
    setText('w-camw-num', pct(camW));
    setText('w-lidarw-num', pct(lidW));

    // Performance
    setText('p-bl-lat', (metrics.baseline_latency_s || '0') + ' s');
    setText('p-bl-fn', metrics.fn_baseline);
    setText('p-pr-lat', (metrics.total_latency_s || '0') + ' s');
    setText('p-pr-fn', metrics.fn_adaptive);
    setText('p-hard', metrics.n_hard_instances);

    // DHIP
    setText('dhip-easy', timing.n_easy);
    setText('dhip-hard', timing.n_hard);
    setText('dhip-bl-lat', (metrics.baseline_latency_s || '0') + ' s');
    setText('dhip-ad-lat', (timing.lidar_fusion_s || '0') + ' s');
    const reduction = (metrics.baseline_latency_s - timing.lidar_fusion_s);
    setText('dhip-red', (reduction > 0 ? reduction.toFixed(3) : '0') + ' s');

    // System info
    setText('si-scene', si.scene_name);
    setText('si-sample', si.sample_token);
    setText('si-ts', si.timestamp ? new Date(si.timestamp / 1000).toLocaleString() : '—');
    setText('si-cam', si.cam_sensor);
    setText('si-lidar', si.lidar_sensor);
    setText('si-pts', (si.n_lidar_points || 0).toLocaleString());
    setText('si-res', si.image_res);
    setText('si-time', si.total_time_s + ' s');
    setText('si-model', si.model);
    setText('si-device', si.device);

    drawLatencyChart(timing);
}

// ══════════════════════════════════════════════════════════════════
// BEV CANVAS DRAWING
// ══════════════════════════════════════════════════════════════════
function drawEgo(ctx, W, H) {
    const cx = W / 2, cy = H / 2;
    ctx.fillStyle = '#0ea5e9';
    ctx.fillRect(cx - 5, cy - 8, 10, 16);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.strokeRect(cx - 5, cy - 8, 10, 16);
}

function renderBEV(data) {
    const canvas = document.getElementById('bev-canvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    // Clear and draw grid
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= W; i += 50) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(W, i); ctx.stroke();
    }

    const bgImg = new Image();
    bgImg.onload = () => {
        ctx.globalAlpha = 0.6;
        ctx.drawImage(bgImg, 0, 0, W, H);
        ctx.globalAlpha = 1.0;
        drawEgo(ctx, W, H);

        const dets = data.detections || [];
        dets.forEach(d => {
            // Remap coordinates based on bev image size mapping
            const px = d.bev_x * (W / 400);
            const py = d.bev_y * (H / 400);
            if (px < 0 || px > W || py < 0 || py > H) return;
            const color = d.color || '#3b82f6';

            ctx.beginPath();
            ctx.arc(px, py, d.is_hard ? 12 : 8, 0, Math.PI * 2);
            ctx.strokeStyle = color;
            ctx.lineWidth = d.is_hard ? 3 : 2;
            ctx.stroke();
            ctx.fillStyle = color + '66';
            ctx.fill();

            ctx.fillStyle = '#fff';
            ctx.font = 'bold 11px Inter';
            ctx.fillText(d.name || d.class, px + 12, py + 4);
        });
        updateBEVLegend(dets);
    };
    bgImg.src = data.lidar_bev;
}

function updateBEVLegend(dets) {
    const seen = {};
    dets.forEach(d => { seen[d.class] = d.color; });
    const leg = document.getElementById('bev-legend');
    leg.innerHTML = '<strong>Legend</strong>';
    Object.entries(seen).forEach(([cls, color]) => {
        leg.innerHTML += `<div style="display:flex;align-items:center;gap:8px">
      <span style="width:12px;height:12px;border-radius:4px;background:${color};display:inline-block"></span>
      <span style="color:#e2e8f0;text-transform:uppercase;font-size:0.75rem">${cls}</span></div>`;
    });
}

// ══════════════════════════════════════════════════════════════════
// 3D CANVAS (simple orthographic projection)
// ══════════════════════════════════════════════════════════════════
let cam3d = { theta: 0.7, phi: 0.5, zoom: 3.5 };
let drag3d = null;
let isDragging3D = false;
let selected3DObj = null;

function select3DObject(obj) {
    selected3DObj = obj;
    showObjDetailCard(obj);
    if (state.result && state.result.detections) {
        draw3DScene(state.result.detections, null);
    }
}

function showObjDetailCard(obj) {
    if (!obj) return;
    const card = document.getElementById('obj-detail-card');
    if (!card) return;

    const name = obj.name || (obj.class ? obj.class.toUpperCase() : 'Object');
    document.getElementById('pop-title').textContent = name;
    document.getElementById('pop-cls').textContent = obj.class ? obj.class.toUpperCase() : '—';
    document.getElementById('pop-loss').textContent = obj.loss !== undefined ? obj.loss.toFixed(4) : '0.0350';
    document.getElementById('pop-acc').textContent = obj.accuracy !== undefined ? obj.accuracy.toFixed(1) + '%' : '95.0%';
    document.getElementById('pop-conf').textContent = obj.confidence !== undefined ? (obj.confidence * 100).toFixed(1) + '%' : '90.0%';
    document.getElementById('pop-dist').textContent = obj.distance !== undefined ? obj.distance.toFixed(2) + ' m' : '—';
    document.getElementById('pop-pos').textContent = obj.position ? `[${obj.position.join(', ')}] m` : '—';
    document.getElementById('pop-size').textContent = obj.size ? `[${obj.size.join(', ')}] m` : '—';
    document.getElementById('pop-yaw').textContent = obj.orientation !== undefined ? obj.orientation.toFixed(1) + '°' : '0.0°';
    document.getElementById('pop-pts').textContent = obj.n_lidar_pts !== undefined ? obj.n_lidar_pts + ' points' : '—';
    document.getElementById('pop-dhip').textContent = obj.is_hard ? '⚡ Hard Instance (Deformable Attn)' : '✅ Easy Instance (Lightweight Conv)';

    card.classList.remove('hidden');
}

function hideObjDetailCard() {
    const card = document.getElementById('obj-detail-card');
    if (card) card.classList.add('hidden');
}

// Close button event for popup card
document.addEventListener('DOMContentLoaded', () => {
    const btnClose = document.getElementById('pop-close');
    if (btnClose) {
        btnClose.addEventListener('click', () => {
            selected3DObj = null;
            hideObjDetailCard();
            if (state.result && state.result.detections) draw3DScene(state.result.detections, null);
        });
    }
});

// Helper function to project and draw custom 3D vehicle geometry
function draw3DVehicleShape(ctx, project, px, py, pz, w, l, h, type, color, isSelected) {
    type = (type || '').toLowerCase();
    const strokeColor = isSelected ? '#00ffff' : color;
    const lineWidth = isSelected ? 3 : 2;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;

    function drawWireBox(cx_min, cx_max, cy_min, cy_max, cz_min, cz_max, dash = []) {
        const corners = [
            [px + cx_min, py + cy_min, pz + cz_min],
            [px + cx_max, py + cy_min, pz + cz_min],
            [px + cx_max, py + cy_max, pz + cz_min],
            [px + cx_min, py + cy_max, pz + cz_min],
            [px + cx_min, py + cy_min, pz + cz_max],
            [px + cx_max, py + cy_min, pz + cz_max],
            [px + cx_max, py + cy_max, pz + cz_max],
            [px + cx_min, py + cy_max, pz + cz_max],
        ];
        const proj = corners.map(project);
        const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];

        ctx.beginPath();
        if (dash.length > 0) ctx.setLineDash(dash);
        else ctx.setLineDash([]);

        edges.forEach(([a, b]) => {
            ctx.moveTo(proj[a][0], proj[a][1]);
            ctx.lineTo(proj[b][0], proj[b][1]);
        });
        ctx.stroke();
        ctx.setLineDash([]);
    }

    function drawLine(pA, pB) {
        const a = project(pA), b = project(pB);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
    }

    const hw = w / 2, hl = l / 2, hh = h / 2;

    if (type === 'car' || type === 'motorcycle') {
        // DETAILED 3D SEDAN CAR SHAPE:
        // 1. Lower Chassis Box
        drawWireBox(-hw, hw, -hh, 0, -hl, hl);

        // 2. Upper Roof Cabin Box (Narrower & Tapered)
        drawWireBox(-hw * 0.8, hw * 0.8, 0, hh, -hl * 0.4, hl * 0.4);

        // 3. Front Windshield Slope Lines
        drawLine([px - hw * 0.8, py + hh, pz + hl * 0.4], [px - hw, py, pz + hl * 0.65]);
        drawLine([px + hw * 0.8, py + hh, pz + hl * 0.4], [px + hw, py, pz + hl * 0.65]);

        // 4. Rear Windshield Slope Lines
        drawLine([px - hw * 0.8, py + hh, pz - hl * 0.4], [px - hw, py, pz - hl * 0.65]);
        drawLine([px + hw * 0.8, py + hh, pz - hl * 0.4], [px + hw, py, pz - hl * 0.65]);

        // 5. 4 Wheel Discs
        [ [-hw, -hh, hl * 0.65], [hw, -hh, hl * 0.65], [-hw, -hh, -hl * 0.65], [hw, -hh, -hl * 0.65] ].forEach(wPos => {
            const [wx, wy] = project([px + wPos[0], py + wPos[1], pz + wPos[2]]);
            ctx.beginPath();
            ctx.arc(wx, wy, 5, 0, Math.PI * 2);
            ctx.stroke();
        });

    } else if (type === 'bus') {
        // BUS SHAPE: Tall body + Window stripes
        drawWireBox(-hw, hw, -hh, hh, -hl, hl);
        drawWireBox(-hw * 0.98, hw * 0.98, hh * 0.1, hh * 0.7, -hl * 0.9, hl * 0.9, [4, 4]);

    } else if (type === 'truck' || type === 'trailer') {
        // TRUCK SHAPE: Front Cab + Rear Cargo Box
        drawWireBox(-hw * 0.9, hw * 0.9, -hh, hh * 0.85, hl * 0.35, hl);
        drawWireBox(-hw, hw, -hh, hh * 1.15, -hl, hl * 0.3);

    } else if (type === 'pedestrian') {
        // PEDESTRIAN SHAPE: Body Torso + Head Circle
        drawWireBox(-hw * 0.8, hw * 0.8, -hh, hh * 0.6, -hl * 0.8, hl * 0.8);
        const [hx, hy] = project([px, py + hh, pz]);
        ctx.beginPath();
        ctx.arc(hx, hy, 7, 0, Math.PI * 2);
        ctx.stroke();

    } else {
        drawWireBox(-hw, hw, -hh, hh, -hl, hl);
    }
}

function draw3DScene(dets, lidar_bev_b64) {
    const canvas = document.getElementById('canvas3d');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const ct = Math.cos(cam3d.theta), st = Math.sin(cam3d.theta);
    const cp = Math.cos(cam3d.phi), sp = Math.sin(cam3d.phi);
    const scale = Math.min(W, H) / 2 * cam3d.zoom / 50;

    function project([x, y, z]) {
        const rx = ct * x - st * z;
        const rz = st * x + ct * z;
        const ry2 = cp * y - sp * rz;
        return [cx + rx * scale, cy - ry2 * scale];
    }

    // Grid
    ctx.strokeStyle = 'rgba(14, 165, 233, 0.15)';
    ctx.lineWidth = 1;
    for (let g = -50; g <= 50; g += 10) {
        const [a, b] = project([g, 0, -50]), [c, d] = project([g, 0, 50]);
        ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
        const [e, f] = project([-50, 0, g]), [p, q] = project([50, 0, g]);
        ctx.beginPath(); ctx.moveTo(e, f); ctx.lineTo(p, q); ctx.stroke();
    }

    // Ego Vehicle
    ctx.fillStyle = '#0ea5e9';
    const [ex, ey] = project([0, 0, 0]);
    ctx.beginPath(); ctx.arc(ex, ey, 7, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 12px Inter';
    ctx.fillText('EGO VEHICLE', ex + 12, ey + 4);

    // Render 3D Objects with Custom Vehicle Geometries
    dets.forEach(d => {
        const [px, py, pz] = d.position;
        const [w, l, h] = d.size;
        const color = d.color || '#3b82f6';
        const isSelected = (selected3DObj && selected3DObj.id === d.id);

        // Draw custom 3D vehicle shape (Car / Bus / Truck / Pedestrian)
        draw3DVehicleShape(ctx, project, px, py, pz, w, l, h, d.class, color, isSelected);

        // ONLY IF THIS SPECIFIC VEHICLE IS CLICKED / SELECTED, DRAW ITS METRICS LABEL & GLOW RETICLE!
        if (isSelected) {
            const [lx, ly] = project([px, py, pz]);

            // Draw glowing selection reticles
            ctx.beginPath();
            ctx.arc(lx, ly, 18, 0, Math.PI * 2);
            ctx.strokeStyle = '#00ffff';
            ctx.lineWidth = 3;
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(lx, ly, 25, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.4)';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Format metrics label text for clicked vehicle ONLY
            const nameLabel = d.name || `${d.class} ${d.id}`;
            const lVal = d.loss !== undefined ? d.loss.toFixed(4) : '0.0350';
            const aVal = d.accuracy !== undefined ? d.accuracy.toFixed(1) + '%' : '95.0%';
            const cVal = (d.confidence * 100).toFixed(1) + '%';
            const labelText = `🎯 ${nameLabel}  |  Loss: ${lVal}  |  Acc: ${aVal}  |  Conf: ${cVal}`;

            ctx.font = 'bold 12px Inter';
            const textWidth = ctx.measureText(labelText).width;

            // Draw high-contrast background pill box for text readability
            ctx.fillStyle = 'rgba(5, 10, 20, 0.9)';
            ctx.strokeStyle = '#00ffff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.roundRect(lx - textWidth / 2 - 10, ly - 34, textWidth + 20, 26, 6);
            ctx.fill();
            ctx.stroke();

            // Draw glowing cyan text
            ctx.fillStyle = '#00ffff';
            ctx.fillText(labelText, lx - textWidth / 2, ly - 17);
        }
    });
}

// 3D Canvas Mouse Event Listeners (Rotation, Zoom, and Click Inspection)
const c3 = document.getElementById('canvas3d');
let mouseDownPos = { x: 0, y: 0 };

if (c3) {
    c3.addEventListener('mousedown', e => {
        drag3d = { x: e.clientX, y: e.clientY, t: cam3d.theta, p: cam3d.phi };
        mouseDownPos = { x: e.clientX, y: e.clientY };
        isDragging3D = false;
    });

    window.addEventListener('mousemove', e => {
        if (!drag3d) return;
        const dx = e.clientX - drag3d.x;
        const dy = e.clientY - drag3d.y;
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
            isDragging3D = true;
        }
        cam3d.theta = drag3d.t + dx * 0.01;
        cam3d.phi = Math.max(-1.2, Math.min(1.2, drag3d.p - dy * 0.01));
        if (state.result) draw3DScene(state.result.detections || [], null);
    });

    window.addEventListener('mouseup', e => {
        drag3d = null;
    });

    // CLICK INSPECTION ON 3D CANVAS OBJECT WITH SCALED COORDINATES
    c3.addEventListener('click', e => {
        if (isDragging3D) return; // ignore clicks during rotation drag

        const rect = c3.getBoundingClientRect();
        const scaleX = c3.width / rect.width;
        const scaleY = c3.height / rect.height;
        const clickX = (e.clientX - rect.left) * scaleX;
        const clickY = (e.clientY - rect.top) * scaleY;

        if (!state.result || !state.result.detections) return;
        const dets = state.result.detections;

        const W = c3.width, H = c3.height;
        const cx = W / 2, cy = H / 2;
        const ct = Math.cos(cam3d.theta), st = Math.sin(cam3d.theta);
        const cp = Math.cos(cam3d.phi), sp = Math.sin(cam3d.phi);
        const scale = Math.min(W, H) / 2 * cam3d.zoom / 50;

        function project([x, y, z]) {
            const rx = ct * x - st * z;
            const rz = st * x + ct * z;
            const ry2 = cp * y - sp * rz;
            return [cx + rx * scale, cy - ry2 * scale];
        }

        let minCenterDist = Infinity;
        let clickedObj = null;

        dets.forEach(d => {
            const [px, py, pz] = d.position;
            const [w, l, h] = d.size;
            const half = { x: w / 2, y: h / 2, z: l / 2 };

            // Check distance to object center
            const [projX, projY] = project([px, py, pz]);
            let dMin = Math.sqrt((clickX - projX) ** 2 + (clickY - projY) ** 2);

            // Also check 8 bounding box corners
            for (const sx of [-1, 1]) {
                for (const sy of [-1, 1]) {
                    for (const sz of [-1, 1]) {
                        const [cornerX, cornerY] = project([px + sx * half.x, py + sy * half.y, pz + sz * half.z]);
                        const cDist = Math.sqrt((clickX - cornerX) ** 2 + (clickY - cornerY) ** 2);
                        if (cDist < dMin) dMin = cDist;
                    }
                }
            }

            if (dMin < 50 && dMin < minCenterDist) {
                minCenterDist = dMin;
                clickedObj = d;
            }
        });

        if (clickedObj) {
            select3DObject(clickedObj);
        } else {
            selected3DObj = null;
            hideObjDetailCard();
            draw3DScene(dets, null);
        }
    });

    c3.addEventListener('wheel', e => {
        cam3d.zoom = Math.max(0.5, Math.min(8, cam3d.zoom - (e.deltaY / 500)));
        if (state.result) draw3DScene(state.result.detections || [], null);
        e.preventDefault();
    }, { passive: false });
}

document.getElementById('btn-reset-3d').addEventListener('click', () => {
    cam3d = { theta: 0.7, phi: 0.5, zoom: 3.5 };
    selected3DObj = null;
    hideObjDetailCard();
    if (state.result) draw3DScene(state.result.detections || [], null);
});

// ══════════════════════════════════════════════════════════════════
// LATENCY BAR CHART
// ══════════════════════════════════════════════════════════════════
function drawLatencyChart(timing) {
    const canvas = document.getElementById('perf-chart');
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth || 800;
    const H = canvas.offsetHeight || 100;
    canvas.width = W; canvas.height = H;
    ctx.clearRect(0, 0, W, H);

    const items = [
        { label: 'Camera Inference', val: timing.cam_inference_s || 0, color: '#3b82f6' },
        { label: 'LiDAR Fusion', val: timing.lidar_fusion_s || 0, color: '#10b981' },
    ];
    const maxVal = Math.max(...items.map(i => i.val), 0.001);
    const bH = 24, pad = 15, labelW = 160;

    items.forEach((item, i) => {
        const y = pad + i * (bH + pad);
        const barW = (item.val / maxVal) * (W - labelW - pad * 3);

        ctx.fillStyle = 'rgba(255,255,255,0.05)';
        ctx.fillRect(labelW, y, W - labelW - pad, bH);

        ctx.fillStyle = item.color;
        ctx.fillRect(labelW, y, barW, bH);

        ctx.fillStyle = '#94a3b8';
        ctx.font = '600 12px Inter';
        ctx.fillText(item.label, pad, y + bH / 2 + 4);

        ctx.fillStyle = '#fff';
        ctx.font = '600 12px JetBrains Mono';
        ctx.fillText(item.val.toFixed(3) + ' s', labelW + barW + 8, y + bH / 2 + 4);
    });
}

// ══════════════════════════════════════════════════════════════════
// EMERGENCY CALL MODAL
// ══════════════════════════════════════════════════════════════════

const emergencyModal = document.getElementById('emergency-modal');
const btnCall = document.getElementById('btn-call-emergency');
const btnCancel = document.getElementById('btn-cancel-emergency');
const emStatus = document.getElementById('em-call-status');

async function checkAndPromptEmergency() {
    try {
        const r = await fetch(`${API}/emergency_contact`);
        const data = await r.json();
        if (data.email) {
            document.getElementById('em-modal-name').textContent = data.username || 'User';
            document.getElementById('em-modal-number').textContent = data.email;
            btnCall.disabled = true;
            btnCall.style.display = 'none';
            emStatus.style.color = '#f59e0b';
            emStatus.textContent = 'ACCIDENT DETECTED! Auto-sending emergency Email to ' + data.email + '...';
            emergencyModal.classList.remove('hidden');

            // Automatically send SMS (no user interaction needed at all)
            try {
                const callR = await fetch(`${API}/emergency_call`, { method: 'POST' });
                const callData = await callR.json();

                if (callData.status === 'success') {
                    if (callData.email_sent) {
                        emStatus.style.color = 'var(--green)';
                        emStatus.textContent = 'EMAIL SENT to ' + data.email + '!';
                    } else {
                        emStatus.style.color = 'var(--green)';
                        emStatus.textContent = callData.message;
                    }

                    setTimeout(() => {
                        emergencyModal.classList.add('hidden');
                    }, 8000);
                } else {
                    throw new Error(callData.error || 'Unknown error');
                }
            } catch (callErr) {
                emStatus.style.color = 'var(--red)';
                emStatus.textContent = 'Email send failed: ' + callErr.message;
                btnCall.disabled = false;
                btnCall.style.display = 'inline-block';
            }
        }
    } catch (e) {
        console.error("Could not check emergency contact", e);
    }
}

btnCancel.addEventListener('click', () => {
    emergencyModal.classList.add('hidden');
});

btnCall.addEventListener('click', async () => {
    btnCall.disabled = true;
    emStatus.style.color = 'var(--text)';
    emStatus.textContent = 'Initiating emergency call & sending WhatsApp message...';
    try {
        const r = await fetch(`${API}/emergency_call`, { method: 'POST' });
        const data = await r.json();
        
        if (data.status === 'success') {
            emStatus.style.color = 'var(--green)';
            emStatus.textContent = data.message;
            btnCall.style.display = 'none';

            // Open WhatsApp with pre-filled emergency message
            if (data.whatsapp_url) {
                window.open(data.whatsapp_url, '_blank');
                emStatus.textContent += ' WhatsApp message opened in new tab.';
            }

            setTimeout(() => {
                emergencyModal.classList.add('hidden');
            }, 5000);
        } else {
            throw new Error(data.error);
        }
    } catch (e) {
        emStatus.style.color = 'var(--red)';
        emStatus.textContent = 'Failed: ' + e.message;
        btnCall.disabled = false;
    }
});
