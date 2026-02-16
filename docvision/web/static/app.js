/* ════════════════════════════════════════════════════════════════════
   DocVision — Front-end application logic
   ════════════════════════════════════════════════════════════════════ */

// ─── Global state ─────────────────────────────────────────────────
let currentJobId   = null;
let currentResult  = null;
let selectedFile   = null;
let selectedFiles  = [];        // batch
let batchJobIds    = [];        // all job IDs from batch processing
let batchJobInfos  = {};        // map job_id -> {filename, status}
let processStart   = 0;
let outputView     = 'fields';  // 'fields' | 'report' | 'markdown' | 'json'
let markdownContent = null;     // cached markdown content
let costRefreshTimer = null;

// ─── DOM helpers ──────────────────────────────────────────────────
const $  = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

const dropZone      = $('#dropZone');
const fileInput     = $('#fileInput');
const btnProcess    = $('#btnProcess');
const btnBatch      = $('#btnBatch');
const statusBar     = $('#statusBar');
const statusText    = $('#statusText');
const statusTime    = $('#statusTime');
const lightbox      = $('#lightbox');
const lightboxImg   = $('#lightboxImg');
const modeSelect    = $('#modeSelect');
const modeBadge     = $('#modeBadge');
const docTypeSelect = $('#docTypeSelect');
const docTypeLabel  = $('#docTypeLabel');
const localDetails  = $('#localOptionsDetails');

// ─── Mode toggle ──────────────────────────────────────────────────
modeSelect.addEventListener('change', () => {
  const m = modeSelect.value;
  modeBadge.textContent = m.toUpperCase();
  modeBadge.className = 'mode-badge ' + m;
  const isAzure = m === 'azure';
  localDetails.style.display = isAzure ? 'none' : '';
  docTypeSelect.style.display = isAzure ? '' : 'none';
  docTypeLabel.style.display  = isAzure ? '' : 'none';
});

// ─── Tab switching ────────────────────────────────────────────────
$$('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    $$('.tab').forEach(t => t.classList.remove('active'));
    $$('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    $(`#panel-${tab.dataset.tab}`).classList.add('active');
    if (tab.dataset.tab === 'history') loadHistory();
    if (tab.dataset.tab === 'costs') {
      loadCosts();
      if (!costRefreshTimer) costRefreshTimer = setInterval(loadCosts, 10000);
    } else {
      if (costRefreshTimer) { clearInterval(costRefreshTimer); costRefreshTimer = null; }
    }
  });
});

// ──────────────────────────────────────────────────────────────────
//  FILE SELECTION (single + multi / drag-and-drop)
// ──────────────────────────────────────────────────────────────────
fileInput.addEventListener('change', e => {
  const files = Array.from(e.target.files);
  if (!files.length) return;
  files.length === 1 ? pickFile(files[0]) : pickFiles(files);
});
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  const files = Array.from(e.dataTransfer.files);
  files.length === 1 ? pickFile(files[0]) : pickFiles(files);
});

function pickFile(f) {
  selectedFile  = f;
  selectedFiles = [];
  btnProcess.disabled = false;
  btnProcess.style.display = '';
  btnBatch.style.display   = 'none';
  $('#fileListArea').style.display = 'none';
  generatePreview(f);
}

function pickFiles(files) {
  selectedFiles = files;
  selectedFile  = null;
  btnProcess.style.display = 'none';
  btnBatch.style.display   = '';
  btnBatch.disabled = false;
  $('#previewArea').style.display   = 'none';
  $('#fileListArea').style.display  = 'block';
  renderFileList();
}

function renderFileList() {
  const el = $('#fileList');
  el.innerHTML = selectedFiles.map((f, i) => {
    const kb  = (f.size / 1024).toFixed(0);
    const ext = f.name.split('.').pop().toUpperCase();
    return `<div class="file-item">
      <span>${ext === 'PDF' ? '📄' : '🖼'}</span>
      <span class="name">${esc(f.name)}</span>
      <span class="size">${kb} KB</span>
      <span class="remove" onclick="removeFile(${i})">✕</span>
    </div>`;
  }).join('');
}

function removeFile(idx) {
  selectedFiles = selectedFiles.filter((_, i) => i !== idx);
  if (selectedFiles.length === 0) {
    $('#fileListArea').style.display = 'none';
    btnBatch.style.display = 'none';
  } else if (selectedFiles.length === 1) {
    pickFile(selectedFiles[0]);
  } else {
    renderFileList();
  }
}

// ──────────────────────────────────────────────────────────────────
//  PDF / IMAGE PREVIEW
// ──────────────────────────────────────────────────────────────────
async function generatePreview(f) {
  const area   = $('#previewArea');
  const sizeMB = (f.size / 1024 / 1024).toFixed(2);
  const ext    = f.name.split('.').pop().toUpperCase();
  area.style.display = 'block';
  area.innerHTML = `<div class="preview-area">
    <div class="preview-info">
      <div class="filename">📎 ${esc(f.name)}</div>
      <div class="meta">${sizeMB} MB · ${ext}<br><span id="previewPageInfo"></span></div>
    </div>
  </div>`;

  // For images, show inline preview immediately
  if (['JPG','JPEG','PNG','BMP','WEBP'].includes(ext)) {
    try {
      const url = URL.createObjectURL(f);
      const thumb = document.createElement('img');
      thumb.src = url;
      thumb.className = 'preview-thumb';
      thumb.onclick = () => showLightbox(url);
      area.querySelector('.preview-area').prepend(thumb);
    } catch {}
    return;
  }

  // For PDFs, request server-side thumbnail
  try {
    const form = new FormData();
    form.append('file', f);
    const res = await fetch('/api/preview', { method: 'POST', body: form });
    if (res.ok) {
      const data = await res.json();
      if (data.preview) {
        const thumb = document.createElement('img');
        thumb.src = data.preview;
        thumb.className = 'preview-thumb';
        thumb.onclick = () => showLightbox(data.preview);
        area.querySelector('.preview-area').prepend(thumb);
      }
      const info = $('#previewPageInfo');
      if (info && data.pages) info.textContent = `${data.pages} page${data.pages > 1 ? 's' : ''}`;
    }
  } catch (e) { console.warn('Preview error:', e); }
}

// ──────────────────────────────────────────────────────────────────
//  SINGLE-FILE PROCESSING
// ──────────────────────────────────────────────────────────────────
btnProcess.addEventListener('click', async () => {
  if (!selectedFile) return;
  const form = new FormData();
  form.append('file', selectedFile);
  form.append('processing_mode', modeSelect.value);
  form.append('document_type', docTypeSelect.value);
  $$('#optionsGrid input[type=checkbox]').forEach(cb => {
    form.append(cb.name, cb.checked ? 'true' : 'false');
  });

  btnProcess.disabled = true;
  showStatus('Uploading & starting processing…');

  try {
    const res = await fetch('/api/process', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    currentJobId = data.job_id;
    statusText.textContent = 'Processing document… This may take a few minutes.';
    pollStatus();
  } catch (err) {
    statusText.textContent = `❌ Error: ${err.message}`;
    btnProcess.disabled = false;
  }
});

// ──────────────────────────────────────────────────────────────────
//  BATCH PROCESSING
// ──────────────────────────────────────────────────────────────────
btnBatch.addEventListener('click', async () => {
  if (!selectedFiles.length) return;
  const form = new FormData();
  
  // Store filenames for batch job tracking
  const filenames = selectedFiles.map(f => f.name);
  
  selectedFiles.forEach(f => form.append('files', f));
  form.append('processing_mode', modeSelect.value);
  form.append('document_type', docTypeSelect.value);

  btnBatch.disabled = true;
  showStatus(`Uploading ${selectedFiles.length} files…`);

  try {
    const res = await fetch('/api/process/batch', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    
    // Store batch job IDs and initialize job info
    batchJobIds = data.job_ids;
    batchJobInfos = {};
    data.job_ids.forEach((jid, i) => {
      batchJobInfos[jid] = { filename: filenames[i] || `Document ${i+1}`, status: 'queued' };
    });
    
    statusText.textContent = `Processing ${data.count} documents…`;
    pollBatchStatus(data.job_ids);
  } catch (err) {
    statusText.textContent = `❌ Error: ${err.message}`;
    btnBatch.disabled = false;
  }
});

async function pollBatchStatus(jobIds) {
  let allDone = true, completed = 0, failed = 0, processing = 0;
  for (const jid of jobIds) {
    try {
      const r = await fetch(`/api/jobs/${jid}`);
      const j = await r.json();
      // Update job info
      if (batchJobInfos[jid]) {
        batchJobInfos[jid].status = j.status;
        batchJobInfos[jid].filename = j.filename || batchJobInfos[jid].filename;
      }
      if (j.status === 'completed') completed++;
      else if (j.status === 'failed') failed++;
      else { processing++; allDone = false; }
    } catch { allDone = false; processing++; }
  }
  statusText.textContent = `Batch: ${completed} done, ${processing} processing, ${failed} failed`;
  if (allDone) {
    statusText.textContent = `✅ Batch complete! ${completed} succeeded, ${failed} failed`;
    setTimeout(() => statusBar.classList.remove('show'), 4000);
    btnBatch.disabled = false;
    
    // Populate job selectors for batch results
    populateBatchJobSelectors();
    
    // Load the first completed job
    for (let i = 0; i < jobIds.length; i++) {
      try {
        const r = await fetch(`/api/jobs/${jobIds[i]}`);
        const j = await r.json();
        if (j.status === 'completed') { 
          currentJobId = jobIds[i]; 
          await loadResult(); 
          await loadArtifacts(); 
          break; 
        }
      } catch {}
    }
    loadCosts(); loadHistory();
  } else {
    setTimeout(() => pollBatchStatus(jobIds), 2000);
  }
}

// ──────────────────────────────────────────────────────────────────
//  STATUS BAR & POLLING
// ──────────────────────────────────────────────────────────────────
function showStatus(msg) {
  statusBar.classList.add('show');
  statusText.textContent = msg;
  const pf = $('#progressFill');
  pf.style.width = '0%';
  pf.style.background = 'var(--primary)';
  processStart = Date.now();
  tickTimer();
}

function tickTimer() {
  if (!statusBar.classList.contains('show')) return;
  statusTime.textContent = ((Date.now() - processStart) / 1000).toFixed(0) + 's';
  requestAnimationFrame(() => setTimeout(tickTimer, 500));
}

async function pollStatus() {
  try {
    const res = await fetch(`/api/jobs/${currentJobId}`);
    if (!res.ok) {
      // Job not found (e.g. stale ID after server restart) — stop polling
      statusText.textContent = '⚠️ Job not found — please resubmit';
      setTimeout(() => statusBar.classList.remove('show'), 4000);
      btnProcess.disabled = false;
      return;
    }
    const job = await res.json();
    if (job.status === 'completed') {
      $('#progressFill').style.width = '100%';
      $('#btnCancel').style.display = 'none';
      statusText.textContent = '✅ Processing complete!';
      setTimeout(() => statusBar.classList.remove('show'), 3000);
      btnProcess.disabled = false;
      // Clear batch state for single file processing
      batchJobIds = [];
      batchJobInfos = {};
      hideJobSelectors();
      await loadResult();
      await loadArtifacts();
      loadCosts();
      $$('.tab')[1].click(); // switch to Artifacts
    } else if (job.status === 'failed') {
      $('#progressFill').style.width = '100%';
      $('#progressFill').style.background = 'var(--error)';
      $('#btnCancel').style.display = 'none';
      statusText.textContent = `❌ Failed: ${job.error}`;
      btnProcess.disabled = false;
    } else if (job.status === 'cancelled') {
      $('#progressFill').style.width = '0%';
      $('#btnCancel').style.display = 'none';
      statusText.textContent = '⚠️ Cancelled';
      setTimeout(() => statusBar.classList.remove('show'), 3000);
      btnProcess.disabled = false;
    } else {
      // Update progress from server
      if (job.progress) {
        const pct = job.progress.percent || 0;
        const stage = job.progress.stage || 'Processing';
        statusText.textContent = `${stage}…`;
        $('#progressFill').style.width = pct + '%';
      }
      setTimeout(pollStatus, 1500);
    }
  } catch { setTimeout(pollStatus, 2000); }
}

async function cancelCurrentJob() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      statusText.textContent = 'Cancelling…';
      $('#btnCancel').style.display = 'none';
    }
  } catch (e) {
    console.error('Cancel failed:', e);
  }
}

// ──────────────────────────────────────────────────────────────────
//  BATCH JOB SELECTORS
// ──────────────────────────────────────────────────────────────────
function populateBatchJobSelectors() {
  if (batchJobIds.length <= 1) {
    hideJobSelectors();
    return;
  }
  
  const artifactsSelector = $('#artifactsJobSelector');
  const outputSelector = $('#outputJobSelector');
  const artifactsSelect = $('#artifactsJobSelect');
  const outputSelect = $('#outputJobSelect');
  
  // Build options HTML
  let optionsHtml = '';
  batchJobIds.forEach((jid, index) => {
    const info = batchJobInfos[jid] || {};
    const filename = info.filename || `Document ${index + 1}`;
    const status = info.status || 'unknown';
    const statusIcon = status === 'completed' ? '✅' : status === 'failed' ? '❌' : '⏳';
    optionsHtml += `<option value="${jid}" ${jid === currentJobId ? 'selected' : ''}>${statusIcon} ${filename}</option>`;
  });
  
  artifactsSelect.innerHTML = optionsHtml;
  outputSelect.innerHTML = optionsHtml;
  
  // Show selectors
  artifactsSelector.style.display = 'flex';
  outputSelector.style.display = 'flex';
}

function hideJobSelectors() {
  $('#artifactsJobSelector').style.display = 'none';
  $('#outputJobSelector').style.display = 'none';
}

async function onArtifactsJobChange(jobId) {
  if (!jobId || jobId === currentJobId) return;
  currentJobId = jobId;
  // Update both selects to stay in sync
  $('#outputJobSelect').value = jobId;
  await loadResult();
  await loadArtifacts();
}

async function onOutputJobChange(jobId) {
  if (!jobId || jobId === currentJobId) return;
  currentJobId = jobId;
  // Update both selects to stay in sync
  $('#artifactsJobSelect').value = jobId;
  await loadResult();
  await loadArtifacts();
}

// ──────────────────────────────────────────────────────────────────
//  LOAD RESULT + ARTIFACTS
// ──────────────────────────────────────────────────────────────────
async function loadResult() {
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/result`);
    if (!res.ok) { console.error('Failed to load result:', res.status); return; }
    currentResult = await res.json();
    renderOutput();
  } catch (e) { console.error('Error loading result:', e); }
}

async function loadArtifacts() {
  const res = await fetch(`/api/jobs/${currentJobId}/artifacts`);
  const data = await res.json();
  renderArtifacts(data.artifacts || []);
}

// ──────────────────────────────────────────────────────────────────
//  CONFIDENCE HELPERS
// ──────────────────────────────────────────────────────────────────
function confClass(c) { return c >= 0.8 ? 'high' : c >= 0.5 ? 'mid' : 'low'; }
function confBadge(c) { return `<span class="conf-badge ${confClass(c)}">${(c * 100).toFixed(0)}%</span>`; }

// ──────────────────────────────────────────────────────────────────
//  RENDER ARTIFACTS  (with confidence highlighting)
// ──────────────────────────────────────────────────────────────────
function renderArtifacts(artifacts) {
  const el = $('#artifactsContent');
  if (!artifacts.length && !currentResult) {
    el.innerHTML = '<div class="empty"><div class="icon">🔍</div><p>Upload and process a document first</p></div>';
    return;
  }
  let html = '';

  // Stats row
  if (currentResult) {
    html += '<div class="stats">';
    html += stat('Pages', currentResult.page_count || 0);
    html += stat('Text Lines', countLines());
    html += stat('Tables', countTables());
    html += stat('Fields', (currentResult.fields || []).length);
    html += stat('Time', (currentResult.metadata?.processing_time_seconds || 0).toFixed(1) + 's');
    html += '</div>';

    // Confidence legend
    html += `<div class="conf-legend">
      Confidence: <span class="dot" style="background:var(--conf-high)"></span> High (≥80%)
      <span class="dot" style="background:var(--conf-mid)"></span> Medium (50-79%)
      <span class="dot" style="background:var(--conf-low)"></span> Low (&lt;50%)
    </div>`;

    // Per-page confidence summary with low-confidence line details
    if (currentResult.pages) {
      currentResult.pages.forEach((page, pi) => {
        const pageNum = page.number || (pi + 1);
        const lines = page.text_lines || [];
        if (!lines.length) return;
        const lowLines = lines.filter(l => (l.confidence||0) < 0.5);
        const midLines = lines.filter(l => (l.confidence||0) >= 0.5 && (l.confidence||0) < 0.8);
        const highCount = lines.length - midLines.length - lowLines.length;
        if (lowLines.length || midLines.length) {
          html += `<div style="margin:10px 0;padding:10px 14px;background:var(--surface);border-radius:var(--radius);border:1px solid var(--border);font-size:.84rem">`;
          html += `<strong>Page ${pageNum}</strong> — ${lines.length} lines: `;
          html += `<span class="conf-high">${highCount} high</span>, `;
          html += `<span class="conf-mid">${midLines.length} medium</span>, `;
          html += `<span class="conf-low">${lowLines.length} low</span>`;
          if (lowLines.length) {
            html += `<details style="margin-top:6px"><summary style="cursor:pointer;color:var(--error);font-size:.8rem">⚠ ${lowLines.length} low-confidence line(s)</summary>`;
            html += `<div style="margin-top:6px;font-family:var(--mono);font-size:.78rem">`;
            lowLines.forEach(l => {
              html += `<div style="padding:3px 0;color:var(--conf-low)">• ${confBadge(l.confidence||0)} "${esc(l.text)}"</div>`;
            });
            html += '</div></details>';
          }
          html += '</div>';
        }
      });
    }
  }

  // Group artifacts by page
  if (artifacts.length) {
    const pages = {};
    artifacts.forEach(a => {
      const m = a.name.match(/page_(\d+)/);
      const pn = m ? parseInt(m[1]) : 0;
      if (!pages[pn]) pages[pn] = [];
      pages[pn].push(a);
    });

    for (const [pageNum, arts] of Object.entries(pages)) {
      html += `<h3 style="margin:20px 0 10px;color:var(--muted)">Page ${pageNum}</h3>`;
      html += '<div class="artifacts-grid">';
      for (const a of arts) {
        const friendly = a.name.replace(/page_\d+_/, '').replace(/_/g, ' ');
        html += `<div class="artifact-card">
          <img src="${escAttr(a.url)}" alt="${esc(a.name)}" onclick="showLightbox('${escAttr(a.url)}')" loading="lazy">
          <div class="label"><span>${friendly}</span><span>${a.size_kb} KB</span></div>
        </div>`;
      }
      html += '</div>';
    }
  } else if (!currentResult) {
    html += '<div class="empty"><div class="icon">🔍</div><p>No artifacts generated</p></div>';
  }

  el.innerHTML = html;
}

function stat(lbl, val) { return `<div class="stat"><div class="value">${val}</div><div class="label">${lbl}</div></div>`; }
function countLines() {
  if (!currentResult?.pages) return 0;
  return currentResult.pages.reduce((s, p) => s + (p.text_lines?.length || 0), 0);
}
function countTables() {
  if (currentResult?.tables) return currentResult.tables.length;
  if (!currentResult?.pages) return 0;
  return currentResult.pages.reduce((s, p) => s + (p.tables?.length || 0), 0);
}

// ──────────────────────────────────────────────────────────────────
//  RENDER OUTPUT  (Fields ↔ JSON toggle)
// ──────────────────────────────────────────────────────────────────
function renderOutput() {
  const el = $('#outputContent');
  if (!currentResult) {
    el.innerHTML = '<div class="empty"><div class="icon">📝</div><p>No result yet</p></div>';
    return;
  }

  let html = `
    <div class="view-toggle">
      <button class="vt-btn ${outputView === 'fields' ? 'active' : ''}" onclick="switchView('fields')"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>Fields &amp; Tables</button>
      <button class="vt-btn ${outputView === 'report' ? 'active' : ''}" onclick="switchView('report')"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>Report</button>
      <button class="vt-btn ${outputView === 'spatial' ? 'active' : ''}" onclick="switchView('spatial')"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>Spatial</button>
      <button class="vt-btn ${outputView === 'json' ? 'active' : ''}" onclick="switchView('json')"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>JSON</button>
    </div>
    <div class="editor-toolbar">
      <button class="btn btn-secondary btn-sm" onclick="saveAllEdits()" style="color:#14b8a6"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>Save Edits</button>
      <button class="btn btn-secondary btn-sm" onclick="saveToDisk()" style="color:#14b8a6"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>Save to Disk</button>
      <button class="btn btn-secondary btn-sm" onclick="downloadJson()" style="color:#f59e0b"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>JSON</button>
      <button class="btn btn-secondary btn-sm" onclick="downloadMarkdown()" style="color:#60a5fa"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>Markdown</button>
    </div>`;

  if (outputView === 'fields') html += renderFieldEditor();
  else if (outputView === 'report') html += renderReportView();
  else if (outputView === 'spatial') html += renderSpatialView();
  else html += renderJsonEditor();
  html += '<div class="json-status" id="jsonStatus"></div>';
  el.innerHTML = html;

  if (outputView === 'spatial') {
    // Initialize spatial view after DOM is ready
    setTimeout(initSpatialView, 0);
  }
  if (outputView === 'json') {
    const ed = $('#jsonEditor');
    if (ed) { ed.addEventListener('input', validateJson); validateJson(); }
  }
}

function switchView(view) {
  if (outputView === 'json') {
    const ed = $('#jsonEditor');
    if (ed) { try { currentResult = JSON.parse(ed.value); } catch {} }
  }
  if (outputView === 'fields') syncFieldEdits();
  outputView = view;
  renderOutput();
}

// ── Field Editor view ─────────────────────────────────────────────
function renderFieldEditor() {
  let html = '';
  const fields = currentResult.fields || [];
  const tables = currentResult.tables || [];

  // ─ Fields ─
  if (fields.length) {
    html += `<div class="cost-section"><h3>📋 Extracted Fields (${fields.length})</h3></div>`;
    html += `<div class="conf-legend">
      Confidence: <span class="dot" style="background:var(--conf-high)"></span> High
      <span class="dot" style="background:var(--conf-mid)"></span> Medium
      <span class="dot" style="background:var(--conf-low)"></span> Low — edit inline to correct
    </div>`;
    html += '<div class="field-grid">';
    fields.forEach((f, i) => {
      const c = f.confidence || 0;
      const cls = c < 0.5 ? 'low-conf' : c < 0.8 ? 'mid-conf' : '';
      const src = f.source ? `<span class="field-source">${f.source}</span>` : '';
      html += `<div class="field-card ${cls}">
        <div class="field-header">
          <span class="field-name">${esc(f.name || 'unnamed')}</span>
          ${confBadge(c)}
        </div>
        <input class="field-input" data-field-idx="${i}" value="${escAttr(f.value ?? '')}" placeholder="(empty)">
        <div class="field-meta">
          <span>Type: ${f.data_type || 'string'}</span>
          ${f.status ? `<span>Status: ${f.status}</span>` : ''}
          ${src}
        </div>
      </div>`;
    });
    html += '</div>';
  } else {
    html += '<div style="margin:20px 0;padding:20px;background:var(--surface);border-radius:var(--radius);border:1px solid var(--border);color:var(--muted);text-align:center">No fields extracted — try Azure mode or enable Donut / LayoutLMv3 KIE</div>';
  }

  // ─ Tables ─
  if (tables.length) {
    html += `<div class="cost-section"><h3>📊 Tables (${tables.length})</h3></div>`;
    tables.forEach((table, ti) => {
      const cells = table.cells || [];
      html += `<div style="margin-bottom:20px">`;
      html += `<div style="font-size:.85rem;color:var(--muted);margin-bottom:8px">Table ${ti + 1} — ${table.row_count || table.rows || '?'} rows × ${table.col_count || table.columns || '?'} cols ${confBadge(table.confidence || 0)}</div>`;
      if (cells.length) {
        const maxR = Math.max(...cells.map(c => c.row));
        const maxC = Math.max(...cells.map(c => c.col || c.column || 0));
        html += '<div style="overflow-x:auto"><table class="result-table"><tbody>';
        for (let r = 0; r <= maxR; r++) {
          html += '<tr>';
          for (let c = 0; c <= maxC; c++) {
            const cell = cells.find(x => x.row === r && (x.col === c || x.column === c));
            const txt = cell?.text || '';
            const cc = cell?.confidence || 0;
            const tag = (cell?.is_header || r === 0) ? 'th' : 'td';
            html += `<${tag}><input data-table="${ti}" data-row="${r}" data-col="${c}" value="${escAttr(txt)}" style="color:var(--conf-${confClass(cc)})"></${tag}>`;
          }
          html += '</tr>';
        }
        html += '</tbody></table></div>';
      }
      html += '</div>';
    });
  }

  // ─ Text lines per page (collapsible) ─
  if (currentResult.pages) {
    html += '<div class="cost-section"><h3>📃 Text Lines by Page</h3></div>';
    currentResult.pages.forEach((page, pi) => {
      const lines = page.text_lines || [];
      if (!lines.length) return;
      html += `<details style="margin-bottom:10px"><summary style="cursor:pointer;color:var(--muted);font-size:.88rem;padding:6px 0">Page ${page.number || pi + 1} — ${lines.length} lines</summary><div style="margin-top:6px">`;
      lines.forEach(l => {
        const c = l.confidence || 0;
        html += `<div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-size:.84rem">
          ${confBadge(c)}
          <span style="color:var(--conf-${confClass(c)});font-family:var(--mono);font-size:.8rem">${esc(l.text)}</span>
          <span style="color:var(--muted);font-size:.72rem;margin-left:auto">${l.content_type || ''}</span>
        </div>`;
      });
      html += '</div></details>';
    });
  }

  return html;
}

// ── Report Viewer ─────────────────────────────────────────────────
function renderReportView() {
  const r = currentResult;
  if (!r) return '<div class="empty"><div class="icon">📑</div><p>No result yet</p></div>';
  let h = '<div class="report">';

  // ── Title & Meta ──
  const fn = r.metadata?.filename || 'Document';
  const meta = r.metadata || {};
  h += `<h2>🔎 OCR Report — ${esc(fn)}</h2>`;
  h += '<div class="report-meta">';
  h += rptMeta('Filename', fn);
  h += rptMeta('File Type', meta.file_type || '—');
  h += rptMeta('File Size', fmtBytes(meta.file_size_bytes || 0));
  h += rptMeta('Pages', r.page_count || 0);
  h += rptMeta('Processing Time', (meta.processing_time_seconds || 0).toFixed(2) + 's');
  h += rptMeta('Processed At', meta.processed_at ? new Date(meta.processed_at).toLocaleString() : '—');
  h += rptMeta('Total Fields', (r.fields || []).length);
  h += rptMeta('Total Tables', (r.tables || []).length);
  h += '</div>';

  // ── Per-page sections ──
  const pages = r.pages || [];
  pages.forEach((page, pi) => {
    const pn = page.page_number || page.number || (pi + 1);
    const pm = page.metadata || {};
    h += '<div class="page-section">';
    h += '<div class="page-header">';
    h += `<span class="page-num">📑 Page ${pn}</span>`;
    h += '<span class="page-info">';
    if (pm.width && pm.height) h += `<span>${pm.width}×${pm.height} px</span>`;
    if (pm.dpi) h += `<span>${pm.dpi} DPI</span>`;
    if (pm.content_type) h += `<span>${pm.content_type}</span>`;
    if (pm.readability) {
      const rIcon = {good:'✅',fair:'⚠️',poor:'❌'}[pm.readability] || '❓';
      h += `<span>${rIcon} ${pm.readability}</span>`;
    }
    h += '</span></div>';

    // Layout regions
    const regions = page.layout_regions || [];
    if (regions.length) {
      h += `<h4>🗺️ Layout Regions (${regions.length})</h4>`;
      h += '<table class="report-table"><thead><tr><th>#</th><th>Type</th><th>Position</th><th>Confidence</th><th>Content</th></tr></thead><tbody>';
      regions.forEach((rg, ri) => {
        const bb = rg.bbox || {};
        h += `<tr><td>${ri+1}</td><td><strong>${rg.type || '?'}</strong></td>`;
        h += `<td><span class="spatial-badge">x:${bb.x1||0} y:${bb.y1||0} → ${bb.x2||0},${bb.y2||0}</span></td>`;
        h += `<td>${confBadge(rg.confidence||0)} ${confBarHtml(rg.confidence||0)}</td>`;
        h += `<td>${rg.content_type || '—'}</td></tr>`;
      });
      h += '</tbody></table>';
    }

    // Text lines
    const lines = page.text_lines || [];
    if (lines.length) {
      h += `<h4>📝 Text Lines (${lines.length})</h4>`;
      h += '<table class="report-table"><thead><tr><th>#</th><th>Text</th><th>Position</th><th>Confidence</th><th>Source</th></tr></thead><tbody>';
      lines.forEach((tl, ti) => {
        const bb = tl.bbox || {};
        const cls = confClass(tl.confidence||0);
        h += `<tr><td>${ti+1}</td>`;
        h += `<td style="font-family:var(--mono);font-size:.82rem;color:var(--conf-${cls})">${esc(tl.text||'')}</td>`;
        h += `<td><span class="spatial-badge">x:${bb.x1||0} y:${bb.y1||0}</span></td>`;
        h += `<td>${confBadge(tl.confidence||0)} ${confBarHtml(tl.confidence||0)}</td>`;
        h += `<td>${tl.source || '—'}</td></tr>`;
      });
      h += '</tbody></table>';
    }

    // Tables on this page
    const pageTables = (page.tables || []);
    const docTables = (r.tables || []).filter(t => t.page === pn);
    const tables = pageTables.length ? pageTables : docTables;
    if (tables.length) {
      tables.forEach((tbl, ti) => {
        const cells = tbl.cells || [];
        const rows = tbl.rows || tbl.row_count || 0;
        const cols = tbl.cols || tbl.col_count || tbl.columns || 0;
        const tbb = tbl.bbox || {};
        h += `<h4>📊 Table ${ti+1}</h4>`;
        h += `<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;font-size:.82rem;color:var(--muted)">`;
        h += `<span><strong>Size:</strong> ${rows} rows × ${cols} cols</span>`;
        h += `<span><strong>Confidence:</strong> ${confBadge(tbl.confidence||0)}</span>`;
        if (tbb.x1 !== undefined) h += `<span class="spatial-badge page-loc">📍 x:${tbb.x1} y:${tbb.y1} → ${tbb.x2},${tbb.y2}</span>`;
        h += `<span><strong>Borders:</strong> ${tbl.has_borders ? 'Yes' : 'No'}</span>`;
        h += '</div>';

        if (cells.length && rows > 0 && cols > 0) {
          // Build grid
          const grid = Array.from({length: rows}, () => Array(cols).fill({text:'', confidence:1, isHeader:false}));
          cells.forEach(c => {
            const cr = c.row, cc = c.col ?? c.column ?? 0;
            if (cr >= 0 && cr < rows && cc >= 0 && cc < cols)
              grid[cr][cc] = {text: c.text||'', confidence: c.confidence||0, isHeader: c.is_header||false};
          });

          h += '<div class="rpt-table-wrap"><table class="rpt-table">';
          // Determine header row
          const hasHeaders = cells.some(c => c.is_header);
          if (hasHeaders) {
            const hr = Math.min(...cells.filter(c => c.is_header).map(c => c.row));
            h += '<thead><tr>';
            grid[hr].forEach(cell => { h += `<th>${esc(cell.text) || '&nbsp;'}</th>`; });
            h += '</tr></thead><tbody>';
            for (let rr = hr+1; rr < rows; rr++) {
              h += '<tr>';
              grid[rr].forEach(cell => {
                h += `<td class="cell-${confClass(cell.confidence)}">${esc(cell.text) || '&nbsp;'}</td>`;
              });
              h += '</tr>';
            }
          } else {
            h += '<thead><tr>';
            for (let cc = 0; cc < cols; cc++) h += `<th>Col ${cc+1}</th>`;
            h += '</tr></thead><tbody>';
            for (let rr = 0; rr < rows; rr++) {
              h += '<tr>';
              grid[rr].forEach(cell => {
                h += `<td class="cell-${confClass(cell.confidence)}">${esc(cell.text) || '&nbsp;'}</td>`;
              });
              h += '</tr>';
            }
          }
          h += '</tbody></table></div>';

          // Low-confidence cells
          const lowCells = cells.filter(c => (c.confidence||1) < 0.7);
          if (lowCells.length) {
            h += `<details style="margin:6px 0"><summary class="candidates-toggle">⚠️ ${lowCells.length} low-confidence cell(s)</summary><div style="margin:6px 0">`;
            h += '<table class="report-table" style="font-size:.8rem"><thead><tr><th>Row</th><th>Col</th><th>Text</th><th>Confidence</th></tr></thead><tbody>';
            lowCells.forEach(c => {
              h += `<tr><td>${c.row}</td><td>${c.col??c.column??0}</td><td>${esc(c.text||'')}</td><td>${confBadge(c.confidence||0)}</td></tr>`;
            });
            h += '</tbody></table></div></details>';
          }
        } else {
          h += '<p style="color:var(--muted);font-style:italic">No cell data available</p>';
        }
      });
    }

    h += '</div>'; // page-section
  });

  // ── Extracted Fields ──
  const fields = r.fields || [];
  if (fields.length) {
    h += '<hr><h2>🏷️ Extracted Fields</h2>';
    h += `<div class="conf-legend">
      Confidence: <span class="dot" style="background:var(--conf-high)"></span> High (≥80%)
      <span class="dot" style="background:var(--conf-mid)"></span> Medium (50-79%)
      <span class="dot" style="background:var(--conf-low)"></span> Low (&lt;50%)
    </div>`;

    // Group fields by page
    const byPage = {};
    fields.forEach(f => {
      const pg = f.page || 'doc';
      if (!byPage[pg]) byPage[pg] = [];
      byPage[pg].push(f);
    });

    Object.entries(byPage).forEach(([pg, flds]) => {
      h += `<h3>📄 ${pg === 'doc' ? 'Document-level' : 'Page ' + pg} Fields (${flds.length})</h3>`;
      h += '<div class="field-group">';
      flds.forEach((f, fi) => {
        const c = f.confidence || 0;
        const uid = `cand_${pg}_${fi}`;
        h += '<div class="field-row">';
        h += `<span class="fr-name">${esc(f.name || 'unnamed')}</span>`;
        h += `<span class="fr-value">${esc(String(f.value ?? ''))}</span>`;
        h += '<span class="fr-conf">';
        if (f.status) h += `<span class="status-pill ${f.status}">${f.status}</span>`;
        h += `${confBadge(c)} ${confBarHtml(c)}`;
        if (f.chosen_source || f.source) h += `<span class="fr-source">${f.chosen_source || f.source}</span>`;
        h += '</span>';
        h += '</div>';

        // Candidates
        const cands = f.candidates || [];
        if (cands.length > 1) {
          h += `<div style="padding-left:172px"><span class="candidates-toggle" onclick="toggleCandidates('${uid}')">▸ ${cands.length} candidates</span>`;
          h += `<div class="candidates-panel" id="${uid}">`;
          cands.forEach(cd => {
            h += `<div class="candidate-row"><span class="cr-src">${cd.source||'?'}</span><span>${esc(String(cd.value||''))}</span><span>${confBadge(cd.confidence||0)}</span></div>`;
          });
          h += '</div></div>';
        }
      });
      h += '</div>';
    });
  }

  // ── Validation ──
  const v = r.validation;
  if (v) {
    h += '<hr>';
    const passed = v.passed !== false;
    h += `<div class="validation-box ${passed ? 'passed' : 'failed'}">`;
    h += `<div class="vb-header">${passed ? '✅' : '❌'} Validation ${passed ? 'Passed' : 'Failed'}</div>`;
    h += `<div class="vb-stats">${v.passed_checks||0} / ${v.total_checks||0} checks passed, ${v.failed_checks||0} failed</div>`;
    const issues = v.issues || [];
    if (issues.length) {
      issues.forEach(iss => { h += `<div class="issue-item">⚠️ ${esc(iss)}</div>`; });
    }
    const details = v.details || [];
    if (details.length) {
      h += `<details style="margin-top:8px"><summary class="candidates-toggle">Detailed check results (${details.length})</summary>`;
      h += '<table class="report-table" style="margin-top:6px"><thead><tr><th>Check</th><th>Result</th><th>Message</th></tr></thead><tbody>';
      details.forEach(d => {
        h += `<tr><td>${esc(d.name||'?')}</td><td>${d.passed ? '✅' : '❌'}</td><td>${esc(d.message||'')}</td></tr>`;
      });
      h += '</tbody></table></details>';
    }
    h += '</div>';
  }

  // ── Normalized (Azure) ──
  if (r.normalized) {
    h += '<hr><h2>📦 Normalized Output (Business-Ready)</h2>';
    const n = r.normalized;
    if (n.doc_type) h += `<p><strong>Document Type:</strong> ${esc(n.doc_type)}</p>`;
    if (n.header_fields && Object.keys(n.header_fields).length) {
      h += '<h3>Header Fields</h3><div class="field-group">';
      Object.entries(n.header_fields).forEach(([k, v]) => {
        h += `<div class="field-row"><span class="fr-name">${esc(k)}</span><span class="fr-value">${esc(String(v))}</span></div>`;
      });
      h += '</div>';
    }
    if (n.line_items && n.line_items.length) {
      h += `<h3>Line Items (${n.line_items.length})</h3>`;
      const keys = [...new Set(n.line_items.flatMap(li => Object.keys(li)))];
      h += '<div class="rpt-table-wrap"><table class="rpt-table"><thead><tr>';
      keys.forEach(k => { h += `<th>${esc(k)}</th>`; });
      h += '</tr></thead><tbody>';
      n.line_items.forEach(li => {
        h += '<tr>';
        keys.forEach(k => { h += `<td>${esc(String(li[k] ?? ''))}</td>`; });
        h += '</tr>';
      });
      h += '</tbody></table></div>';
    }
    if (n.totals && Object.keys(n.totals).length) {
      h += '<h3>Totals</h3><div class="field-group">';
      Object.entries(n.totals).forEach(([k, v]) => {
        h += `<div class="field-row"><span class="fr-name">${esc(k)}</span><span class="fr-value">${esc(String(v))}</span></div>`;
      });
      h += '</div>';
    }
  }

  h += '<hr><p style="text-align:center;font-size:.78rem;color:var(--muted);margin-top:24px">Report generated by DocVision — Horizon OCR Pipeline</p>';
  h += '</div>'; // .report
  return h;
}

// ── Spatial OCR View ───────────────────────────────────────────────
let spatialCurrentPage = 1;
let spatialArtifacts = [];

function renderSpatialView() {
  if (!currentJobId || !currentResult) {
    return '<div class="empty"><div class="icon">🗺</div><p>No result yet</p></div>';
  }
  
  const pageCount = currentResult.page_count || currentResult.pages?.length || 1;
  
  return `
    <div class="spatial-viewer" id="spatialViewer">
      <div class="spatial-page-nav">
        <button onclick="spatialPrevPage()" id="spatialPrev">◀ Previous</button>
        <span class="page-info">Page <span id="spatialPageNum">1</span> of ${pageCount}</span>
        <button onclick="spatialNextPage()" id="spatialNext">Next ▶</button>
      </div>
      <div class="spatial-container" id="spatialContainer">
        <div class="spatial-side image-side" id="spatialImageSide">
          <div class="spatial-loading">
            <div class="spinner"></div>
            <p>Loading page image...</p>
          </div>
        </div>
        <div class="spatial-side text-side" id="spatialTextSide">
          <div class="spatial-loading">
            <div class="spinner"></div>
            <p>Loading OCR data...</p>
          </div>
        </div>
      </div>
      <div class="spatial-footer">Left: Original document image | Right: Reconstructed OCR text layout</div>
    </div>`;
}

async function initSpatialView() {
  spatialCurrentPage = 1;
  spatialArtifacts = [];
  
  // Fetch artifacts list
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/artifacts`);
    if (res.ok) {
      const data = await res.json();
      spatialArtifacts = data.artifacts || [];
    }
  } catch (e) {
    console.error('Failed to load artifacts:', e);
  }
  
  renderSpatialPage();
  updateSpatialNavButtons();
}

function spatialPrevPage() {
  if (spatialCurrentPage > 1) {
    spatialCurrentPage--;
    renderSpatialPage();
    updateSpatialNavButtons();
  }
}

function spatialNextPage() {
  const pageCount = currentResult.page_count || currentResult.pages?.length || 1;
  if (spatialCurrentPage < pageCount) {
    spatialCurrentPage++;
    renderSpatialPage();
    updateSpatialNavButtons();
  }
}

function updateSpatialNavButtons() {
  const pageCount = currentResult.page_count || currentResult.pages?.length || 1;
  const prevBtn = $('#spatialPrev');
  const nextBtn = $('#spatialNext');
  const pageNum = $('#spatialPageNum');
  
  if (prevBtn) prevBtn.disabled = spatialCurrentPage <= 1;
  if (nextBtn) nextBtn.disabled = spatialCurrentPage >= pageCount;
  if (pageNum) pageNum.textContent = spatialCurrentPage;
}

function renderSpatialPage() {
  const imageSide = $('#spatialImageSide');
  const textSide = $('#spatialTextSide');
  if (!imageSide || !textSide) return;
  
  // Get page data
  const pages = currentResult.pages || [];
  const pageIdx = spatialCurrentPage - 1;
  const pageData = pages[pageIdx];
  
  if (!pageData) {
    imageSide.innerHTML = '<div class="spatial-no-data"><div class="icon">📄</div><p>No page data found</p></div>';
    textSide.innerHTML = '<div class="spatial-no-data"><div class="icon">📝</div><p>No OCR data for this page</p></div>';
    return;
  }
  
  // Render image side
  const paddedPage = String(spatialCurrentPage).padStart(3, '0');
  const preprocessedArtifact = spatialArtifacts.find(a => a.name === `page_${paddedPage}_preprocessed`);
  const ocrArtifact = spatialArtifacts.find(a => a.name === `page_${paddedPage}_ocr`);
  const anyArtifact = preprocessedArtifact || ocrArtifact || spatialArtifacts.find(a => a.name.includes(`page_${paddedPage}`));
  
  if (anyArtifact) {
    imageSide.innerHTML = `<img src="${anyArtifact.url}" alt="Page ${spatialCurrentPage}" id="spatialPageImg" onload="onSpatialImageLoad()">`;
  } else {
    imageSide.innerHTML = '<div class="spatial-no-data"><div class="icon">🖼</div><p>No page image available</p><small>Artifacts may not have been generated</small></div>';
  }
  
  // Render text side with positioned elements
  renderSpatialTextOverlay(pageData, textSide);
}

function onSpatialImageLoad() {
  // Re-render text side to match image dimensions if needed
  const img = $('#spatialPageImg');
  const textSide = $('#spatialTextSide');
  if (img && textSide) {
    // Set min-height to match image
    textSide.style.minHeight = img.naturalHeight ? `${img.offsetHeight}px` : '500px';
  }
}

function renderSpatialTextOverlay(pageData, container) {
  // ═══════════════════════════════════════════════════════════════════
  // EXACT SPATIAL LAYOUT - Text positioned at exact document coordinates
  // with hover-to-expand for overlapping text
  // ═══════════════════════════════════════════════════════════════════
  
  const meta = pageData.metadata || {};
  
  // Calculate actual page dimensions from content if not in metadata
  // This handles documents where text coordinates exceed metadata dimensions
  const artifactLines = pageData.text_lines || [];
  let maxX = 0, maxY = 0;
  artifactLines.forEach(line => {
    if (line.bbox) {
      const x2 = line.bbox.x2 ?? line.bbox[2] ?? 0;
      const y2 = line.bbox.y2 ?? line.bbox[3] ?? 0;
      if (x2 > maxX) maxX = x2;
      if (y2 > maxY) maxY = y2;
    }
  });
  
  // Use whichever is larger: metadata dimensions or calculated from content
  const pageWidth = Math.max(meta.width || 2550, maxX + 50);
  const pageHeight = Math.max(meta.height || 3300, maxY + 50);
  
  // Use a fixed display width and scale everything proportionally
  const displayWidth = 800;
  const scale = displayWidth / pageWidth;
  const displayHeight = Math.round(pageHeight * scale);
  
  // Use text_lines for BOTH display AND counting (same source as artifacts tab)
  const textLines = [];
  
  // Debug: log low confidence items
  const lowConfDebug = artifactLines.filter(l => (l.confidence || 0) < 0.5);
  console.log('Low confidence items from text_lines:', lowConfDebug.map(l => ({
    text: l.text,
    confidence: l.confidence,
    hasBbox: !!l.bbox,
    bbox: l.bbox
  })));
  
  // Debug page dimensions
  console.log('Page dimensions:', { pageWidth, pageHeight, displayWidth, scale });
  
  artifactLines.forEach((line) => {
    if (!line || !line.text) return;
    const text = line.text.trim();
    if (!text) return;
    const bbox = line.bbox || {};
    const x1 = bbox.x1 ?? bbox[0] ?? 0;
    const y1 = bbox.y1 ?? bbox[1] ?? 0;
    const x2 = bbox.x2 ?? bbox[2] ?? x1 + 100;
    const y2 = bbox.y2 ?? bbox[3] ?? y1 + 20;
    const hasBbox = !!(line.bbox && (line.bbox.x1 !== undefined || line.bbox[0] !== undefined));
    textLines.push({ 
      text, 
      confidence: line.confidence ?? 0,
      x1, y1, x2, y2,
      hasBbox
    });
  });
  
  // Count confidence (same as artifacts tab)
  const lowCount = textLines.filter(l => (l.confidence || 0) < 0.5).length;
  const medCount = textLines.filter(l => (l.confidence || 0) >= 0.5 && (l.confidence || 0) < 0.8).length;
  const highCount = textLines.length - lowCount - medCount;
  const totalLines = textLines.length;
  
  // Debug: log what will be rendered
  const lowWithBbox = textLines.filter(l => (l.confidence || 0) < 0.5 && l.hasBbox);
  console.log('Low confidence items WITH bbox (will be rendered):', lowWithBbox);
  
  // Collect tables
  const pageTables = [];
  const seenTableIds = new Set();
  
  const addTable = (table) => {
    if (!table || !table.bbox) return;
    const tableId = table.id || `t_${Math.round(table.bbox?.x1 ?? 0)}_${Math.round(table.bbox?.y1 ?? 0)}`;
    if (!seenTableIds.has(tableId)) {
      seenTableIds.add(tableId);
      pageTables.push(table);
    }
  };
  
  (currentResult.tables || [])
    .filter(t => (t.page || 1) === spatialCurrentPage)
    .forEach(addTable);
  (pageData.tables || []).forEach(addTable);
  
  // Get all text lines with bbox for display (don't filter by table overlap anymore)
  const filteredTextLines = textLines.filter(line => line.hasBbox);
  
  // Debug: Show what will be rendered
  console.log('Filtered text lines count:', filteredTextLines.length);
  console.log('Low conf in filtered:', filteredTextLines.filter(l => l.confidence < 0.5));
  console.log('Scale and dimensions:', { pageWidth, pageHeight, displayWidth, displayHeight, scale });
  
  if (textLines.length === 0 && pageTables.length === 0) {
    container.innerHTML = '<div class="spatial-no-data"><div class="icon">📝</div><p>No OCR content available</p></div>';
    return;
  }
  
  // Build the spatial reconstructed view
  let html = `<div class="spatial-reconstructed" style="width:${displayWidth}px;height:${displayHeight}px;position:relative;overflow:visible;">`;
  
  // Render text lines at exact positions
  filteredTextLines.forEach((line, idx) => {
    const left = Math.round(line.x1 * scale);
    const top = Math.round(line.y1 * scale);
    const width = Math.max(20, Math.round((line.x2 - line.x1) * scale));
    const height = Math.max(12, Math.round((line.y2 - line.y1) * scale));
    
    const conf = line.confidence || 0;
    let confClass = '';
    let inlineStyle = '';
    if (conf < 0.5) { 
      confClass = ' low-conf'; 
      inlineStyle = 'background:rgba(254,226,226,0.95);border-left-color:#dc2626;color:#dc2626;';
    }
    else if (conf < 0.8) { 
      confClass = ' med-conf'; 
      inlineStyle = 'background:rgba(254,243,199,0.95);border-left-color:#f59e0b;color:#92400e;';
    }
    else { 
      confClass = ' high-conf'; 
      inlineStyle = 'background:rgba(220,252,231,0.95);border-left-color:#22c55e;color:#166534;';
    }
    
    const confPct = Math.round(conf * 100);
    // Truncate display text if too long, full text shown on hover
    const displayText = line.text.length > 50 ? line.text.substring(0, 47) + '...' : line.text;
    
    html += `<div class="spatial-text${confClass}" 
      style="left:${left}px;top:${top}px;max-width:${width}px;min-height:${Math.max(10, height - 4)}px;${inlineStyle}"
      title="${confPct}% confidence: ${esc(line.text)}"
      data-full-text="${esc(line.text)}">${esc(displayText)}</div>`;
  });
  
  // Render tables at exact positions
  pageTables.forEach((table, tidx) => {
    const bbox = normalizeBbox(table.bbox);
    const left = Math.round(bbox.x1 * scale);
    const top = Math.round(bbox.y1 * scale);
    const width = Math.max(50, Math.round((bbox.x2 - bbox.x1) * scale));
    const height = Math.max(30, Math.round((bbox.y2 - bbox.y1) * scale));
    
    html += `<div class="spatial-table" style="left:${left}px;top:${top}px;width:${width}px;height:${height}px;">`;
    
    // Render table cells
    if (table.cells && table.cells.length > 0) {
      table.cells.forEach(cell => {
        if (!cell.bbox) return;
        const cellBbox = normalizeBbox(cell.bbox);
        // Position relative to table
        const cellLeft = Math.round((cellBbox.x1 - bbox.x1) * scale);
        const cellTop = Math.round((cellBbox.y1 - bbox.y1) * scale);
        const cellWidth = Math.max(20, Math.round((cellBbox.x2 - cellBbox.x1) * scale));
        const cellHeight = Math.max(12, Math.round((cellBbox.y2 - cellBbox.y1) * scale));
        
        const isHeader = cell.is_header ? ' header' : '';
        const cellConf = cell.confidence || 1;
        let cellConfClass = '';
        // Only apply styling, don't count (artifacts only counts text_lines, not table cells)
        if (cellConf < 0.5) { cellConfClass = ' low-conf'; }
        else if (cellConf < 0.8) { cellConfClass = ' med-conf'; }
        else { cellConfClass = ' high-conf'; }
        const cellConfPct = Math.round(cellConf * 100);
        const cellText = cell.text || '';
        const displayCellText = cellText.length > 30 ? cellText.substring(0, 27) + '...' : cellText;
        
        html += `<div class="spatial-table-cell${isHeader}${cellConfClass}" 
          style="left:${cellLeft}px;top:${cellTop}px;width:${cellWidth}px;height:${cellHeight}px;"
          title="${cellConfPct}% confidence: ${esc(cellText)}"
          data-full-text="${esc(cellText)}">${esc(displayCellText)}</div>`;
      });
    }
    
    html += '</div>';
  });
  
  html += '</div>';
  
  // Find low/medium confidence items that weren't rendered (no bbox)
  const renderedSet = new Set(filteredTextLines.map(l => l.text));
  const lowConfItems = textLines.filter(l => (l.confidence || 0) < 0.5);
  const medConfItems = textLines.filter(l => (l.confidence || 0) >= 0.5 && (l.confidence || 0) < 0.8);
  const lowNotRendered = lowConfItems.filter(l => !l.hasBbox);
  const medNotRendered = medConfItems.filter(l => !l.hasBbox);
  
  // Add info bar with confidence counts from artifacts tab (text_lines)
  html += `<div style="font-size:12px;color:#444;padding:10px 12px;border-top:1px solid #ddd;background:#f5f5f5;display:flex;flex-wrap:wrap;gap:12px;align-items:center;">
    <span><strong>${totalLines}</strong> lines (${filteredTextLines.length} shown)</span>
    <span style="color:#888">|</span>
    <span style="background:#dcfce7;color:#166534;padding:2px 8px;border-radius:3px;border-left:3px solid #22c55e;"><strong>${highCount}</strong> high</span>
    <span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:3px;border-left:3px solid #f59e0b;"><strong>${medCount}</strong> medium</span>
    <span style="background:#fee2e2;color:#dc2626;padding:2px 8px;border-radius:3px;border-left:3px solid #dc2626;"><strong>${lowCount}</strong> low</span>
    <span style="color:#888;margin-left:auto;font-size:11px;"><em>Hover for full text + confidence %</em></span>
  </div>`;
  
  // Show low-confidence items that couldn't be positioned (no bbox)
  if (lowNotRendered.length > 0 || medNotRendered.length > 0) {
    html += `<details style="font-size:11px;padding:8px 12px;border-top:1px solid #eee;background:#fafafa;">
      <summary style="cursor:pointer;color:#666;">⚠ ${lowNotRendered.length + medNotRendered.length} low/medium items without position data</summary>
      <div style="margin-top:6px;max-height:150px;overflow-y:auto;">`;
    lowNotRendered.forEach(l => {
      const pct = Math.round((l.confidence || 0) * 100);
      html += `<div style="padding:2px 0;color:#dc2626;"><span style="background:#fee2e2;padding:1px 4px;border-radius:2px;font-size:10px;">${pct}%</span> "${esc(l.text)}"</div>`;
    });
    medNotRendered.forEach(l => {
      const pct = Math.round((l.confidence || 0) * 100);
      html += `<div style="padding:2px 0;color:#92400e;"><span style="background:#fef3c7;padding:1px 4px;border-radius:2px;font-size:10px;">${pct}%</span> "${esc(l.text)}"</div>`;
    });
    html += `</div></details>`;
  }
  
  container.innerHTML = html;
}

// Render a table as a clean HTML table
function renderCleanTable(table) {
  if (!table.cells || table.cells.length === 0) {
    return '<p class="text-line">Empty table</p>';
  }
  
  // Find table dimensions
  let maxRow = 0, maxCol = 0;
  table.cells.forEach(cell => {
    const row = cell.row ?? 0;
    const col = cell.col ?? 0;
    const rowSpan = cell.row_span ?? 1;
    const colSpan = cell.col_span ?? 1;
    maxRow = Math.max(maxRow, row + rowSpan);
    maxCol = Math.max(maxCol, col + colSpan);
  });
  
  // Create 2D grid
  const grid = Array(maxRow).fill(null).map(() => Array(maxCol).fill(null));
  const occupied = Array(maxRow).fill(null).map(() => Array(maxCol).fill(false));
  
  // Place cells in grid
  table.cells.forEach(cell => {
    const row = cell.row ?? 0;
    const col = cell.col ?? 0;
    const rowSpan = cell.row_span ?? 1;
    const colSpan = cell.col_span ?? 1;
    
    // Find first unoccupied position
    let startRow = row, startCol = col;
    while (startRow < maxRow && startCol < maxCol && occupied[startRow][startCol]) {
      startCol++;
      if (startCol >= maxCol) {
        startCol = 0;
        startRow++;
      }
    }
    
    if (startRow < maxRow && startCol < maxCol) {
      grid[startRow][startCol] = {
        text: cell.text || '',
        isHeader: cell.is_header || startRow === 0,
        rowSpan,
        colSpan,
        confidence: cell.confidence || 1
      };
      
      // Mark occupied cells
      for (let r = startRow; r < Math.min(startRow + rowSpan, maxRow); r++) {
        for (let c = startCol; c < Math.min(startCol + colSpan, maxCol); c++) {
          occupied[r][c] = true;
        }
      }
    }
  });
  
  // Generate HTML table
  let html = '<table>';
  for (let r = 0; r < maxRow; r++) {
    html += '<tr>';
    for (let c = 0; c < maxCol; c++) {
      const cell = grid[r][c];
      if (cell === null) {
        // Check if this cell is part of a span (skip it)
        if (!occupied[r][c]) {
          html += '<td></td>';
        }
        continue;
      }
      
      const tag = cell.isHeader ? 'th' : 'td';
      const spanAttrs = [];
      if (cell.rowSpan > 1) spanAttrs.push(`rowspan="${cell.rowSpan}"`);
      if (cell.colSpan > 1) spanAttrs.push(`colspan="${cell.colSpan}"`);
      const confClass = cell.confidence < 0.7 ? ' class="low-conf"' : '';
      
      html += `<${tag} ${spanAttrs.join(' ')}${confClass}>${esc(cell.text)}</${tag}>`;
    }
    html += '</tr>';
  }
  html += '</table>';
  
  return html;
}

// Convert bbox to pixel position with scaling
function bboxToPixels(bbox, scale) {
  let x1, y1;
  
  if (Array.isArray(bbox)) {
    [x1, y1] = bbox;
  } else if (bbox.x1 !== undefined) {
    x1 = bbox.x1;
    y1 = bbox.y1;
  } else if (bbox.x !== undefined) {
    x1 = bbox.x;
    y1 = bbox.y;
  } else {
    return { left: 0, top: 0 };
  }
  
  return {
    left: Math.round(x1 * scale),
    top: Math.round(y1 * scale)
  };
}

// Convert bbox to pixel rect (position + size) with scaling
function bboxToPixelsRect(bbox, scale) {
  let x1, y1, x2, y2;
  
  if (Array.isArray(bbox)) {
    [x1, y1, x2, y2] = bbox;
  } else if (bbox.x1 !== undefined) {
    x1 = bbox.x1;
    y1 = bbox.y1;
    x2 = bbox.x2;
    y2 = bbox.y2;
  } else if (bbox.x !== undefined && bbox.width !== undefined) {
    x1 = bbox.x;
    y1 = bbox.y;
    x2 = bbox.x + bbox.width;
    y2 = bbox.y + bbox.height;
  } else {
    return { left: 0, top: 0, width: 50, height: 20 };
  }
  
  return {
    left: Math.round(x1 * scale),
    top: Math.round(y1 * scale),
    width: Math.max(10, Math.round((x2 - x1) * scale)),
    height: Math.max(10, Math.round((y2 - y1) * scale))
  };
}

function bboxToPercent(bbox, pageWidth, pageHeight) {
  // Handle different bbox formats: {x1,y1,x2,y2} or [x1,y1,x2,y2]
  let x1, y1, x2, y2;
  
  if (Array.isArray(bbox)) {
    [x1, y1, x2, y2] = bbox;
  } else if (bbox.x1 !== undefined) {
    x1 = bbox.x1;
    y1 = bbox.y1;
    x2 = bbox.x2;
    y2 = bbox.y2;
  } else if (bbox.x !== undefined && bbox.width !== undefined) {
    x1 = bbox.x;
    y1 = bbox.y;
    x2 = bbox.x + bbox.width;
    y2 = bbox.y + bbox.height;
  } else {
    return { left: 0, top: 0, width: 5, height: 2 };
  }
  
  // Convert to percentages
  const left = (x1 / pageWidth) * 100;
  const top = (y1 / pageHeight) * 100;
  const width = ((x2 - x1) / pageWidth) * 100;
  const height = ((y2 - y1) / pageHeight) * 100;
  
  return {
    left: Math.max(0, left),
    top: Math.max(0, top),
    width: Math.max(1, Math.min(100 - left, width)),
    height: Math.max(0.5, Math.min(100 - top, height))
  };
}

// Normalize bbox to {x1,y1,x2,y2} format
function normalizeBbox(bbox) {
  if (Array.isArray(bbox)) {
    return { x1: bbox[0], y1: bbox[1], x2: bbox[2], y2: bbox[3] };
  }
  if (bbox.x1 !== undefined) {
    return { x1: bbox.x1, y1: bbox.y1, x2: bbox.x2, y2: bbox.y2 };
  }
  if (bbox.x !== undefined && bbox.width !== undefined) {
    return { x1: bbox.x, y1: bbox.y, x2: bbox.x + bbox.width, y2: bbox.y + bbox.height };
  }
  return { x1: 0, y1: 0, x2: 0, y2: 0 };
}

// Calculate overlap ratio of bbox A inside bbox B
function bboxOverlap(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  
  if (x1 >= x2 || y1 >= y2) return 0; // No overlap
  
  const overlapArea = (x2 - x1) * (y2 - y1);
  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  
  return aArea > 0 ? overlapArea / aArea : 0;
}

function rptMeta(label, value) {
  return `<div class="report-meta-item"><div class="rml">${label}</div><div class="rmv">${esc(String(value))}</div></div>`;
}

function confBarHtml(c) {
  const pct = Math.round(c * 100);
  return `<span class="conf-bar"><span class="conf-bar-fill ${confClass(c)}" style="width:${pct}%"></span></span>`;
}

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1024*1024) return (b/1024).toFixed(1) + ' KB';
  return (b/(1024*1024)).toFixed(2) + ' MB';
}

function toggleCandidates(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('show');
}

// ── JSON Editor view ──────────────────────────────────────────────
function renderJsonEditor() {
  const jsonStr = JSON.stringify(currentResult, null, 2);
  return `
    <div style="display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap">
      <input style="flex:1;min-width:200px;padding:8px 14px;border-radius:var(--radius);border:1px solid var(--border);background:var(--surface);color:var(--text);font-size:.88rem;font-family:var(--mono)"
             id="jsonSearch" placeholder="🔍 Search in JSON…" oninput="highlightJson()">
      <button class="btn btn-secondary btn-sm" onclick="formatJson()">Format</button>
      <button class="btn btn-secondary btn-sm" onclick="collapseJson()">Collapse</button>
    </div>
    <textarea class="json-editor" id="jsonEditor" spellcheck="false">${esc(jsonStr)}</textarea>`;
}

function syncFieldEdits() {
  if (!currentResult) return;
  $$('.field-input[data-field-idx]').forEach(inp => {
    const idx = parseInt(inp.dataset.fieldIdx);
    if (currentResult.fields?.[idx]) currentResult.fields[idx].value = inp.value;
  });
  $$('input[data-table]').forEach(inp => {
    const ti = parseInt(inp.dataset.table);
    const row = parseInt(inp.dataset.row);
    const col = parseInt(inp.dataset.col);
    const cell = currentResult.tables?.[ti]?.cells?.find(c => c.row === row && (c.col === col || c.column === col));
    if (cell) cell.text = inp.value;
  });
}

function validateJson() {
  const ed = $('#jsonEditor'), st = $('#jsonStatus');
  if (!ed || !st) return;
  try { JSON.parse(ed.value); st.textContent = '✓ Valid JSON'; st.className = 'json-status valid'; }
  catch (e) { st.textContent = '✗ ' + e.message; st.className = 'json-status invalid'; }
}
function formatJson() {
  const ed = $('#jsonEditor');
  try { ed.value = JSON.stringify(JSON.parse(ed.value), null, 2); validateJson(); } catch {}
}
function collapseJson() {
  const ed = $('#jsonEditor');
  try { ed.value = JSON.stringify(JSON.parse(ed.value)); validateJson(); } catch {}
}
function highlightJson() {
  const term = $('#jsonSearch')?.value;
  if (!term) return;
  const ed = $('#jsonEditor');
  const idx = ed.value.toLowerCase().indexOf(term.toLowerCase());
  if (idx !== -1) {
    ed.focus(); ed.setSelectionRange(idx, idx + term.length);
    ed.scrollTop = ed.value.substring(0, idx).split('\n').length * 18 - 100;
  }
}

// ──────────────────────────────────────────────────────────────────
//  SAVE / DOWNLOAD
// ──────────────────────────────────────────────────────────────────
async function saveAllEdits() {
  if (!currentJobId) return;
  if (outputView === 'fields') syncFieldEdits();
  else {
    const ed = $('#jsonEditor');
    if (ed) { try { currentResult = JSON.parse(ed.value); } catch { return alert('Fix JSON errors before saving'); } }
  }
  const res = await fetch(`/api/jobs/${currentJobId}/result`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(currentResult)
  });
  if (res.ok) { const st = $('#jsonStatus'); if (st) { st.textContent = '✓ Saved successfully'; st.className = 'json-status valid'; } }
}

function downloadJson() {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url;
  a.download = (currentResult?.metadata?.filename || 'document').replace(/\.[^.]+$/, '') + '_result.json';
  a.click(); URL.revokeObjectURL(url);
}

async function downloadMarkdown() {
  if (!currentJobId) return alert('No processed document');
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/download/markdown`);
    if (!res.ok) throw new Error('Download failed');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (currentResult?.metadata?.filename || 'document').replace(/\.[^.]+$/, '') + '_report.md';
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) { alert('Markdown download failed: ' + e.message); }
}

async function saveToDisk() {
  if (!currentJobId) return alert('No processed document to save');
  if (outputView === 'fields') syncFieldEdits();
  else {
    const ed = $('#jsonEditor');
    if (ed) { try { currentResult = JSON.parse(ed.value); } catch { return alert('Fix JSON errors first'); } }
  }
  await fetch(`/api/jobs/${currentJobId}/result`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(currentResult)
  });
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/save`, { method: 'POST' });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Save failed'); }
    const data = await res.json();
    const st = $('#jsonStatus');
    if (st) { st.textContent = `✓ Saved to ${data.path}`; st.className = 'json-status valid'; }
  } catch (e) { alert('Failed to save: ' + e.message); }
}

// ──────────────────────────────────────────────────────────────────
//  HISTORY PANEL
// ──────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    if (!res.ok) return;
    renderHistory((await res.json()).jobs);
  } catch (e) { console.warn('History load error:', e); }
}

function renderHistory(jobs) {
  const tbody = $('#historyTableBody');
  if (!jobs?.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:40px">No documents processed yet</td></tr>';
    return;
  }
  tbody.innerHTML = jobs.map(j => {
    const mode = j.processing_mode === 'azure' ? '☁ Azure' : '🖥 Local';
    const dot  = `<span class="status-dot ${j.status}"></span>${j.status}`;
    const time = j.processing_time ? j.processing_time.toFixed(1) + 's' : '—';
    const created = j.created ? new Date(j.created).toLocaleString() : '—';
    return `<tr onclick="loadJob('${j.job_id}')">
      <td><strong>${esc(j.filename)}</strong></td>
      <td>${mode}</td><td>${dot}</td>
      <td>${j.page_count ?? '—'}</td><td>${j.text_lines ?? '—'}</td>
      <td>${j.tables ?? '—'}</td><td>${j.fields ?? '—'}</td>
      <td style="font-family:var(--mono);font-size:.8rem">${time}</td>
      <td style="font-size:.8rem;color:var(--muted)">${created}</td>
    </tr>`;
  }).join('');
}

async function loadJob(jobId) {
  currentJobId = jobId;
  try { await loadResult(); await loadArtifacts(); $$('.tab')[1].click(); }
  catch (e) { alert('Could not load job: ' + e.message); }
}

// ──────────────────────────────────────────────────────────────────
//  COST & USAGE DASHBOARD
// ──────────────────────────────────────────────────────────────────
async function loadCosts() {
  try {
    const res = await fetch('/api/costs');
    if (!res.ok) return;
    renderCosts(await res.json());
  } catch (e) { console.warn('Costs load error:', e); }
}

function renderCosts(data) {
  const c = data.costs || {};
  const cache = data.cache || {};

  $('#ccTotalCost').textContent  = '$' + (c.estimated_cost_usd || 0).toFixed(4);
  $('#ccTotalCalls').textContent = c.total_calls || 0;
  $('#ccDICalls').textContent    = c.total_di_calls || 0;
  $('#ccGPTCalls').textContent   = c.total_gpt_calls || 0;
  $('#ccPages').textContent      = c.total_pages_analysed || 0;
  $('#ccTokens').textContent     = (c.total_tokens || 0).toLocaleString();
  $('#ccCacheHits').textContent  = c.cache_hits || 0;
  $('#ccSaved').textContent      = '$' + (c.cost_saved_by_cache_usd || 0).toFixed(4);

  const hits = cache.hits || 0, misses = cache.misses || 0, total = hits + misses;
  const hitPct = total > 0 ? (hits / total * 100) : 0;
  $('#cacheHitBar').style.width  = hitPct + '%';
  $('#cacheMissBar').style.width = (100 - hitPct) + '%';
  $('#cacheHitPct').textContent  = hitPct.toFixed(0) + '% hits (' + hits + ')';
  $('#cacheMissPct').textContent = (100 - hitPct).toFixed(0) + '% misses (' + misses + ')';
  $('#cacheEntries').textContent = cache.entries || 0;
  const cs = $('#cacheStatus');
  cs.textContent = cache.enabled ? '✓ Active' : '○ Inactive';
  cs.style.color = cache.enabled ? 'var(--success)' : 'var(--muted)';

  const records = c.records || [];
  let diCost = 0, gptCost = 0, diPages = 0, gptTokens = 0;
  records.forEach(r => {
    if (r.service === 'doc_intelligence') { diCost += r.estimated_cost_usd; diPages += r.pages; }
    else if (r.service === 'gpt_vision') { gptCost += r.estimated_cost_usd; gptTokens += (r.prompt_tokens || 0) + (r.completion_tokens || 0); }
  });
  $('#bdDICalls').textContent  = c.total_di_calls || 0;
  $('#bdDIPages').textContent  = diPages;
  $('#bdDICost').textContent   = '$' + diCost.toFixed(4);
  $('#bdGPTCalls').textContent = c.total_gpt_calls || 0;
  $('#bdGPTTokens').textContent = gptTokens.toLocaleString();
  $('#bdGPTCost').textContent  = '$' + gptCost.toFixed(4);

  const tbody = $('#costTableBody');
  if (!records.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:30px">No API calls recorded yet</td></tr>';
    return;
  }
  tbody.innerHTML = [...records].reverse().map(r => {
    const svcCls = r.service === 'doc_intelligence' ? 'svc-di' : 'svc-gpt';
    const svcLbl = r.service === 'doc_intelligence' 
      ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:3px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>DI' 
      : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:3px"><path d="M12 8V4H8"/><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11v6"/><path d="m9 14 3-3 3 3"/></svg>GPT';
    const detail = r.service === 'doc_intelligence'
      ? (r.pages || 0) + ' page' + ((r.pages || 0) !== 1 ? 's' : '')
      : ((r.prompt_tokens || 0) + (r.completion_tokens || 0)).toLocaleString() + ' tok';
    const ts = r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : '—';
    const cached = r.cached ? '<span class="cached-tag">CACHED</span>' : '';
    return `<tr><td class="time-col">${ts}</td><td><span class="svc ${svcCls}">${svcLbl}</span></td><td>${r.model || '—'}</td><td>${detail}</td><td class="time-col">${(r.latency_seconds || 0).toFixed(1)}s</td><td>$${(r.estimated_cost_usd || 0).toFixed(4)}</td><td>${cached}</td></tr>`;
  }).join('');
}

async function resetCosts() {
  if (!confirm('Reset all cost tracking counters?')) return;
  try { await fetch('/api/costs/reset', { method: 'POST' }); await loadCosts(); } catch (e) { alert('Failed: ' + e.message); }
}
async function clearCache() {
  if (!confirm('Clear the entire Azure response cache?')) return;
  try {
    const res = await fetch('/api/cache/clear', { method: 'POST' });
    const data = await res.json();
    alert(`Cache cleared — ${data.entries_cleared} entries removed`);
    await loadCosts();
  } catch (e) { alert('Failed: ' + e.message); }
}

// ──────────────────────────────────────────────────────────────────
//  LIGHTBOX
// ──────────────────────────────────────────────────────────────────
function showLightbox(url) {
  lightboxImg.src = url;
  lightbox.classList.add('show');
}
lightbox.addEventListener('click', () => lightbox.classList.remove('show'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') lightbox.classList.remove('show'); });

// ──────────────────────────────────────────────────────────────────
//  UTILITY
// ──────────────────────────────────────────────────────────────────
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/'/g,'&#39;').replace(/"/g,'&quot;'); }
function escAttr(s) { return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/'/g,'&#39;'); }
