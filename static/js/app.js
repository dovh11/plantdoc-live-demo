/**
 * PlantDoc -- Frontend Application Logic
 *
 * Responsibilities:
 *  - Theme management (system / light / dark, persisted to localStorage)
 *  - Health-check polling (model status badge)
 *  - File upload handling (gallery + camera)
 *  - Drag-and-drop support
 *  - Image preview on canvas
 *  - POST /predict API call with threshold
 *  - Canvas bounding-box & label rendering
 *  - Detection list building
 *  - UI state helpers (error banner, empty state)
 */

'use strict';

/* ─────────────────────────────────────────────────────────────────────────
   Application State
───────────────────────────────────────────────────────────────────────── */
const state = {
  imageFile:   null,
  imageBitmap: null,
  origW:       0,
  origH:       0,
  lastResults: [],
  inferring:   false,
};

/* ─────────────────────────────────────────────────────────────────────────
   DOM References
───────────────────────────────────────────────────────────────────────── */
const dropZone       = document.getElementById('drop-zone');
const inputGallery   = document.getElementById('upload-gallery');
const inputCamera    = document.getElementById('upload-camera');
const threshSlider   = document.getElementById('threshold-slider');
const threshDisplay  = document.getElementById('threshold-display');
const btnAnalyze     = document.getElementById('btn-analyze');
const resultsSection = document.getElementById('results-section');
const canvas         = document.getElementById('result-canvas');
const ctx            = canvas.getContext('2d');
const canvasOverlay  = document.getElementById('canvas-overlay');
const overlayText    = document.getElementById('overlay-text');
const detectionsList = document.getElementById('detections-list');
const detCount       = document.getElementById('det-count');
const resultsTime    = document.getElementById('results-time');
const errorBanner    = document.getElementById('error-banner');
const errorText      = document.getElementById('error-text');
const badgeDot       = document.getElementById('badge-dot');
const badgeLabel     = document.getElementById('badge-label');

// Theme controls
const settingsBtn    = document.getElementById('settings-btn');
const settingsPanel  = document.getElementById('settings-panel');
const themeOptions   = document.querySelectorAll('.theme-option');

/**
 * Base URL for all API calls.
 * Populated from config.js (window.PLANTDOC_CONFIG.API_BASE_URL).
 * Falls back to '' (same origin) when running in unified Docker/uvicorn mode.
 */
const API_BASE = (window.PLANTDOC_CONFIG && window.PLANTDOC_CONFIG.API_BASE_URL) || '';

/* ─────────────────────────────────────────────────────────────────────────
   Color Palette — 28 vivid, distinct colors mapped by class index
───────────────────────────────────────────────────────────────────────── */
const CLASS_COLORS = [
  '#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#ff922b',
  '#cc5de8','#22b8cf','#f06595','#a9e34b','#74c0fc',
  '#ff8787','#fcc419','#51cf66','#339af0','#f76707',
  '#ae3ec9','#0ca678','#e64980','#94d82d','#228be6',
  '#fa5252','#fab005','#40c057','#1c7ed6','#fd7014',
  '#9c36b5','#099268','#c2255c',
];

/**
 * Returns a deterministic color for a given class index.
 * @param {number} classId
 * @returns {string} hex color
 */
function classColor(classId) {
  return CLASS_COLORS[classId % CLASS_COLORS.length];
}

/**
 * Converts a hex color string to {r, g, b} components.
 * @param {string} hex  e.g. "#ff6b6b"
 * @returns {{ r: number, g: number, b: number }}
 */
function hexToRgb(hex) {
  return {
    r: parseInt(hex.slice(1, 3), 16),
    g: parseInt(hex.slice(3, 5), 16),
    b: parseInt(hex.slice(5, 7), 16),
  };
}

/* ─────────────────────────────────────────────────────────────────────────
   Health Check — polls /health until model is ready
───────────────────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const res  = await fetch(API_BASE + '/health');
    const data = await res.json();
    if (data.model_ready) {
      badgeDot.style.background = '#3fb950';
      badgeLabel.textContent    = `Model ready · ${data.labels_count} classes`;
    } else {
      badgeDot.style.background = '#e3b341';
      badgeLabel.textContent    = 'Model loading…';
      setTimeout(checkHealth, 3000);
    }
  } catch {
    badgeDot.style.background = '#f85149';
    badgeLabel.textContent    = 'Server offline';
    setTimeout(checkHealth, 5000);
  }
}
checkHealth();

/* ─────────────────────────────────────────────────────────────────────────
   Theme Management
   Preferences stored in localStorage key 'plantdoc-theme'.
   Possible values: 'system' | 'light' | 'dark'
───────────────────────────────────────────────────────────────────────── */

/**
 * Resolves the effective theme string ('light' or 'dark') from a preference.
 * When preference is 'system', reads the OS media query.
 * @param {'system'|'light'|'dark'} preference
 * @returns {'light'|'dark'}
 */
function resolveTheme(preference) {
  if (preference === 'system') {
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }
  return preference;
}

/**
 * Applies a theme preference: updates data-theme on <html>,
 * persists to localStorage, and syncs the radio-button UI.
 * @param {'system'|'light'|'dark'} preference
 */
function applyTheme(preference) {
  const resolved = resolveTheme(preference);
  document.documentElement.setAttribute('data-theme', resolved);
  localStorage.setItem('plantdoc-theme', preference);

  // Sync aria-checked states on all theme-option buttons
  themeOptions.forEach(btn => {
    const isActive = btn.dataset.themeValue === preference;
    btn.setAttribute('aria-checked', String(isActive));
  });
}

/**
 * Opens or closes the settings panel.
 * @param {boolean} [force]  If provided, forces open (true) or closed (false).
 */
function toggleSettingsPanel(force) {
  const isHidden = settingsPanel.hasAttribute('hidden');
  const shouldOpen = force !== undefined ? force : isHidden;

  if (shouldOpen) {
    settingsPanel.removeAttribute('hidden');
    settingsBtn.setAttribute('aria-expanded', 'true');
    // Focus the first option for keyboard users
    const first = settingsPanel.querySelector('.theme-option');
    if (first) first.focus();
  } else {
    settingsPanel.setAttribute('hidden', '');
    settingsBtn.setAttribute('aria-expanded', 'false');
  }
}

// Open / close on gear button click
settingsBtn.addEventListener('click', e => {
  e.stopPropagation();
  toggleSettingsPanel();
});

// Select theme when a radio option is clicked
themeOptions.forEach(btn => {
  btn.addEventListener('click', () => {
    applyTheme(btn.dataset.themeValue);
    // Small delay so user sees the checkmark animate before panel closes
    setTimeout(() => toggleSettingsPanel(false), 120);
  });
});

// Keyboard: Escape closes the panel
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !settingsPanel.hasAttribute('hidden')) {
    toggleSettingsPanel(false);
    settingsBtn.focus();
  }
});

// Click outside closes the panel
document.addEventListener('click', e => {
  if (!settingsPanel.hasAttribute('hidden') &&
      !settingsBtn.contains(e.target) &&
      !settingsPanel.contains(e.target)) {
    toggleSettingsPanel(false);
  }
});

// Live-update when OS theme changes (only relevant when preference = 'system')
window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
  const saved = localStorage.getItem('plantdoc-theme') || 'system';
  if (saved === 'system') applyTheme('system');
});

// Initialise: sync UI to whatever theme was applied by the inline script
(function initThemeUI() {
  const saved = localStorage.getItem('plantdoc-theme') || 'system';
  applyTheme(saved);
}());

/* ─────────────────────────────────────────────────────────────────────────
   Upload Triggers
───────────────────────────────────────────────────────────────────────── */

/**
 * Programmatically open the gallery or camera file picker.
 * @param {'gallery'|'camera'} mode
 */
function triggerUpload(mode) {
  if (mode === 'camera') {
    inputCamera.click();
  } else {
    inputGallery.click();
  }
}
// Expose to inline onclick attributes in HTML
window.triggerUpload = triggerUpload;

/**
 * Validates and initiates preview for a selected file.
 * @param {File|null|undefined} file
 */
function handleFileSelect(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please select a valid image file (JPEG, PNG, WebP).');
    return;
  }
  hideError();
  state.imageFile = file;
  loadImagePreview(file);
}

inputGallery.addEventListener('change', e => handleFileSelect(e.target.files[0]));
inputCamera.addEventListener('change',  e => handleFileSelect(e.target.files[0]));

/* ─────────────────────────────────────────────────────────────────────────
   Drag & Drop
───────────────────────────────────────────────────────────────────────── */
['dragenter', 'dragover'].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  })
);

['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
  })
);

dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  handleFileSelect(file);
});

// Keyboard accessibility for the drop zone
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    triggerUpload('gallery');
  }
});

/* ─────────────────────────────────────────────────────────────────────────
   Image Preview
   Decodes the file with createImageBitmap and draws it on the canvas.
───────────────────────────────────────────────────────────────────────── */
function loadImagePreview(file) {
  const url = URL.createObjectURL(file);

  createImageBitmap(file)
    .then(bitmap => {
      state.imageBitmap = bitmap;
      state.origW = bitmap.width;
      state.origH = bitmap.height;
      URL.revokeObjectURL(url);

      // Reveal results section
      resultsSection.style.display = 'block';

      // Set canvas intrinsic size to match the original image
      canvas.width  = bitmap.width;
      canvas.height = bitmap.height;
      ctx.drawImage(bitmap, 0, 0);

      // Reset previous results
      state.lastResults        = [];
      detectionsList.innerHTML = '';
      detCount.textContent     = '0';
      resultsTime.textContent  = '';
      renderEmptyState('Image loaded — click Analyze to detect diseases.');

      btnAnalyze.disabled = false;
      btnAnalyze.focus();

      // Update drop zone to show file info
      dropZone.querySelector('.drop-icon').textContent = '✅';
      dropZone.querySelector('h2').textContent =
        file.name.length > 30 ? file.name.slice(0, 28) + '…' : file.name;
      dropZone.querySelector('p').textContent =
        `${(file.size / 1024).toFixed(1)} KB · ${bitmap.width}×${bitmap.height}px`;
    })
    .catch(() => showError('Failed to load image. Please try another file.'));
}

/* ─────────────────────────────────────────────────────────────────────────
   Confidence Threshold Slider
───────────────────────────────────────────────────────────────────────── */
threshSlider.addEventListener('input', () => {
  const v = parseFloat(threshSlider.value).toFixed(2);
  threshDisplay.textContent = v;
  threshSlider.setAttribute('aria-valuenow', v);
});

/* ─────────────────────────────────────────────────────────────────────────
   Inference — POST /predict
───────────────────────────────────────────────────────────────────────── */
btnAnalyze.addEventListener('click', runInference);

async function runInference() {
  if (!state.imageFile || state.inferring) return;

  state.inferring = true;
  hideError();

  const threshold = parseFloat(threshSlider.value);
  const t0 = performance.now();

  // Show loading overlay
  canvasOverlay.classList.add('active');
  overlayText.textContent  = 'Running inference…';
  btnAnalyze.disabled      = true;
  detectionsList.innerHTML = '';

  try {
    const fd = new FormData();
    fd.append('file',      state.imageFile);
    fd.append('threshold', threshold);

    const res = await fetch(API_BASE + '/predict', { method: 'POST', body: fd });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(detail.detail || `Server error ${res.status}`);
    }

    const detections = await res.json();
    const elapsed    = (performance.now() - t0).toFixed(0);

    state.lastResults = detections;
    renderResults(detections, elapsed);

  } catch (err) {
    showError(`Inference failed: ${err.message}`);
    renderEmptyState('Detection failed. Check your connection and try again.');
    console.error(err);
  } finally {
    canvasOverlay.classList.remove('active');
    state.inferring     = false;
    btnAnalyze.disabled = false;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
   Results Rendering
───────────────────────────────────────────────────────────────────────── */

/**
 * Redraws the canvas with all detection boxes and populates the list.
 * @param {Array<{class:string, class_id:number, confidence:number, box:number[]}>} detections
 * @param {string} elapsedMs  Round-trip time string (ms)
 */
function renderResults(detections, elapsedMs) {
  // 1. Redraw original image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.imageBitmap, 0, 0);

  // 2. Update meta row
  detCount.textContent    = detections.length;
  resultsTime.textContent = `${elapsedMs} ms`;

  if (detections.length === 0) {
    renderEmptyState(
      'No diseases detected above the confidence threshold.\nTry lowering the slider.'
    );
    return;
  }

  // 3. Draw bounding boxes on canvas
  detections.forEach((det, idx) => drawBox(det, idx));

  // 4. Build detection list cards
  detectionsList.innerHTML = '';
  detections.forEach((det, idx) => {
    const color   = classColor(det.class_id ?? idx);
    const confPct = (det.confidence * 100).toFixed(1);
    const [x1, y1, x2, y2] = det.box;

    const item = document.createElement('div');
    item.className = 'detection-item';
    item.setAttribute('role', 'listitem');
    item.style.animationDelay = `${idx * 55}ms`;
    item.innerHTML = `
      <div class="det-color-swatch" style="background:${color}"></div>
      <div class="det-name">${escHtml(det.class)}</div>
      <div class="det-box-coords">[${x1},${y1}→${x2},${y2}]</div>
      <div class="det-conf-bar-wrap">
        <div class="det-conf-bar">
          <div class="det-conf-fill"
               style="width:${det.confidence * 100}%;background:${color}"></div>
        </div>
        <div class="det-conf-text">${confPct}%</div>
      </div>
    `;
    detectionsList.appendChild(item);
  });
}

/**
 * Draws a single bounding box with corner accents and a label pill on the canvas.
 * @param {{ class:string, class_id:number, confidence:number, box:number[] }} det
 * @param {number} idx  Detection index (fallback for color)
 */
function drawBox(det, idx) {
  const [x1, y1, x2, y2] = det.box;
  const w = x2 - x1;
  const h = y2 - y1;
  const color     = classColor(det.class_id ?? idx);
  const { r, g, b } = hexToRgb(color);

  const lineW = Math.max(2, Math.min(4, canvas.width / 300));
  ctx.lineWidth   = lineW;
  ctx.strokeStyle = color;

  // Semi-transparent fill
  ctx.fillStyle = `rgba(${r},${g},${b},0.10)`;
  ctx.fillRect(x1, y1, w, h);

  // Full border
  ctx.strokeRect(x1, y1, w, h);

  // L-shaped corner accents
  const cLen = Math.min(20, w * 0.25, h * 0.25);
  ctx.lineWidth   = lineW + 1;
  ctx.strokeStyle = color;
  ctx.beginPath();
  // Top-left
  ctx.moveTo(x1, y1 + cLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cLen, y1);
  // Top-right
  ctx.moveTo(x2 - cLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cLen);
  // Bottom-right
  ctx.moveTo(x2, y2 - cLen); ctx.lineTo(x2, y2); ctx.lineTo(x2 - cLen, y2);
  // Bottom-left
  ctx.moveTo(x1 + cLen, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - cLen);
  ctx.stroke();

  // Label pill
  const label    = `${det.class}  ${(det.confidence * 100).toFixed(1)}%`;
  const fontSize = Math.max(11, Math.min(16, canvas.width / 50));
  ctx.font       = `600 ${fontSize}px 'Inter', sans-serif`;
  const textW    = ctx.measureText(label).width;
  const padX = 8, padY = 5;
  const pillH = fontSize + padY * 2;
  const pillW = textW + padX * 2;

  // Clamp pill position inside canvas bounds
  const pillX = Math.min(Math.max(x1, 0), canvas.width  - pillW);
  const pillY = Math.max(y1 - pillH - 3, 0);

  // Pill background
  ctx.fillStyle = color;
  roundRect(ctx, pillX, pillY, pillW, pillH, 5);
  ctx.fill();

  // Pill text
  ctx.fillStyle    = isLightColor(r, g, b) ? '#111' : '#fff';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, pillX + padX, pillY + pillH / 2);
}

/**
 * Draws a filled rounded rectangle path on a 2D canvas context.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x  @param {number} y  @param {number} w  @param {number} h
 * @param {number} r  Corner radius
 */
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y,     x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h,     x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y,         x + r, y);
  ctx.closePath();
}

/**
 * Returns true if the given RGB values represent a light color
 * (perceived luminance > 160), used to pick black or white label text.
 * @param {number} r @param {number} g @param {number} b
 * @returns {boolean}
 */
function isLightColor(r, g, b) {
  return (r * 299 + g * 587 + b * 114) / 1000 > 160;
}

/* ─────────────────────────────────────────────────────────────────────────
   UI State Helpers
───────────────────────────────────────────────────────────────────────── */

/** Renders a centred empty-state message in the detections list. */
function renderEmptyState(msg) {
  detectionsList.innerHTML = `
    <div class="state-empty" role="status">
      <div class="state-icon">🔍</div>
      <p>${escHtml(msg)}</p>
    </div>`;
}

/** Shows the error banner with a given message. */
function showError(msg) {
  errorText.textContent = msg;
  errorBanner.classList.add('show');
}

/** Hides the error banner. */
function hideError() {
  errorBanner.classList.remove('show');
}

/**
 * Escapes HTML special characters to prevent XSS in innerHTML assignments.
 * @param {string} str
 * @returns {string}
 */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
