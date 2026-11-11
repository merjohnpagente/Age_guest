/* ──────────────────────────────────────────────────────────────────────────
   FACELENS
   Real-time, on-device face detection + experimental age / gender /
   expression estimation. No camera frame ever leaves the browser.
   ────────────────────────────────────────────────────────────────────────── */

const APP = {
  cameraOn: false,
  cameraStarting: false,
  cameraError: '',
  cameraGeneration: 0,
  mirrored: true,
  tracks: [],
  trackById: new Map(),
  faceMemory: [],
  nextFaceId: 1,
  detector: null,
  detectorReady: false,
  detectorFailed: false,
  detectorDelegate: 'GPU',
  analysisReady: false,
  analysisSettled: false,
  ageReady: false,
  expressionReady: false,
  analysisBusy: false,
  analysisLoopId: 0,
  rafId: 0,
  lastDetectAt: 0,
  detectInterval: 58,
  lastFrameAt: 0,
  lastVideoTimestamp: 0,
  detectionErrors: 0,
  faceCount: 0,
  lastDetectionDuration: 0,
  lastAnalysisDuration: 0,
  reidWindow: 10000,
  trackHideAfter: 300,
  trackRemoveAfter: 1500,
  hintTimer: 0,
  toastTimer: 0,
  infoOpen: false,
  lastFocused: null,
};

const MODEL_ROOT = new URL('./models/faceapi', import.meta.url).href;
const FACE_MODEL = new URL('./models/blaze_face_short_range.tflite', import.meta.url).href;
const VISION_ROOT = new URL('./vendor/wasm', import.meta.url).href;
const VISION_MODULE = new URL('./vendor/vision_bundle.mjs', import.meta.url).href;

const ANALYSIS_INTERVAL = 1700;
const MIN_TRACK_HIDE_AFTER = 300;
const MIN_TRACK_REMOVE_AFTER = 1500;
const LABEL_GAP = 7;
const DEBUG = new URLSearchParams(location.search).has('debug');

const el = {
  video: document.querySelector('#camera'),
  overlay: document.querySelector('#face-overlay'),
  gate: document.querySelector('#gate'),
  gateTitle: document.querySelector('#gate-title'),
  gateCopy: document.querySelector('#gate-copy'),
  gateAction: document.querySelector('#gate-action'),
  gateActionLabel: document.querySelector('#gate-action-label'),
  gateProgress: document.querySelector('#gate-progress'),
  gateProgressLabel: document.querySelector('#gate-progress-label'),
  liveDot: document.querySelector('#live-dot'),
  liveLabel: document.querySelector('#live-label'),
  brandStatus: document.querySelector('#brand-status'),
  faceCount: document.querySelector('#face-count'),
  scanHint: document.querySelector('#scan-hint'),
  stopButton: document.querySelector('#stop-button'),
  fullscreenButton: document.querySelector('#fullscreen-button'),
  infoButton: document.querySelector('#info-button'),
  toast: document.querySelector('#toast'),
  infoBackdrop: document.querySelector('#info-backdrop'),
  infoSheet: document.querySelector('#info-sheet'),
  closeInfo: document.querySelector('#close-info'),
  infoDismiss: document.querySelector('#info-dismiss'),
  topbar: document.querySelector('.topbar'),
  bottombar: document.querySelector('.bottombar'),
};

const EXPRESSION_LABELS = {
  neutral: 'Neutral',
  happy: 'Happy',
  sad: 'Sad',
  angry: 'Angry',
  surprised: 'Surprised',
  disgusted: 'Disgusted',
  fearful: 'Fearful',
};

/* ── Small helpers ───────────────────────────────────────────────────────── */

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const lerp = (from, to, alpha) => from + (to - from) * alpha;

function lerpBox(from, to, alpha) {
  return {
    x: lerp(from.x, to.x, alpha),
    y: lerp(from.y, to.y, alpha),
    w: lerp(from.w, to.w, alpha),
    h: lerp(from.h, to.h, alpha),
  };
}

function intersectionArea(a, b) {
  const width = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const height = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return width > 0 && height > 0 ? width * height : 0;
}

function unionArea(a, b) {
  return a.w * a.h + b.w * b.h - intersectionArea(a, b);
}

function intersectionOverUnion(a, b) {
  const union = unionArea(a, b);
  return union > 0 ? intersectionArea(a, b) / union : 0;
}

function centerDistance(a, b) {
  return Math.hypot(
    (a.x + a.w / 2) - (b.x + b.w / 2),
    (a.y + a.h / 2) - (b.y + b.h / 2),
  );
}

function normaliseBox(box, frameWidth, frameHeight) {
  const x = clamp(box.originX / frameWidth, 0, 1);
  const y = clamp(box.originY / frameHeight, 0, 1);
  const w = clamp(box.width / frameWidth, 0, 1 - x);
  const h = clamp(box.height / frameHeight, 0, 1 - y);
  return { x, y, w, h };
}

function padId(number) {
  return String(number).padStart(2, '0');
}

function safeText(node, text) {
  if (node && node.textContent !== text) node.textContent = text;
}

/* ── UI state ────────────────────────────────────────────────────────────── */

function setCameraState(state) {
  document.body.dataset.camera = state;
  const live = state === 'live';
  el.liveLabel.textContent = live ? 'LIVE' : state === 'starting' || state === 'booting' ? 'STARTING' : 'OFFLINE';
  el.liveDot.classList.toggle('is-offline', !live);
  el.stopButton.setAttribute('aria-label', live ? 'Stop camera' : 'Start camera');
  el.stopButton.title = live ? 'Stop camera' : 'Start camera';
  if (!live) {
    el.faceCount.hidden = true;
    el.scanHint.hidden = true;
  }
  refreshStatus();
}

function refreshStatus() {
  if (APP.cameraStarting) {
    safeText(el.brandStatus, 'Requesting camera access');
    return;
  }
  if (!APP.cameraOn) {
    safeText(el.brandStatus, APP.cameraError ? 'Camera access needed' : 'Camera paused');
    return;
  }
  if (APP.detectorFailed) {
    safeText(el.brandStatus, 'Face detector unavailable');
    return;
  }
  if (!APP.detectorReady) {
    safeText(el.brandStatus, 'Loading face detector');
    return;
  }
  if (!APP.analysisReady) {
    safeText(
      el.brandStatus,
      APP.analysisSettled ? 'Detection live · AI estimates unavailable' : 'Detection live · loading AI estimates',
    );
    return;
  }
  safeText(el.brandStatus, 'On-device AI · analysis live');
}

function setGate({ title, copy, action, progress, busy = false }) {
  el.gateTitle.textContent = title;
  el.gateCopy.textContent = copy;
  if (action) {
    el.gateAction.hidden = false;
    safeText(el.gateActionLabel, action);
  } else {
    el.gateAction.hidden = true;
  }
  el.gateProgress.hidden = !progress;
  if (progress) safeText(el.gateProgressLabel, progress);
  el.gate.classList.toggle('is-busy', busy);
}

function showCameraError(error) {
  const name = error?.name || '';
  let title = 'Camera unavailable.';
  let copy = 'The camera could not start. Close other apps using the camera, then try again.';
  let action = 'Try again';
  APP.cameraError = name || 'unavailable';

  if (name === 'NotAllowedError' || name === 'SecurityError') {
    title = 'Camera access is off.';
    copy = 'Your browser blocked the camera. Allow camera access for this page, then try again. Video never leaves this device.';
    action = 'Try camera again';
  } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    title = 'No camera found.';
    copy = 'Connect a camera and allow access to continue. A secure HTTPS connection is required on phones.';
    action = 'Look again';
  } else if (name === 'NotReadableError' || name === 'AbortError') {
    title = 'Camera is busy.';
    copy = 'Another app or browser tab may be using the camera. Close it, then try again.';
    action = 'Try again';
  } else if (!window.isSecureContext) {
    title = 'A secure connection is needed.';
    copy = 'Camera access only works over HTTPS or localhost. Reopen this page over HTTPS, then try again.';
    action = 'Reload';
  }

  APP.cameraOn = false;
  APP.cameraStarting = false;
  stopRenderLoop();
  clearTracks();
  closeInfo();

  if (el.video.srcObject) {
    el.video.srcObject.getTracks().forEach((track) => track.stop());
    el.video.srcObject = null;
  }

  setCameraState('error');
  setGate({ title, copy, action, progress: null });
  refreshStatus();
}

function setStartingGate() {
  setGate({
    title: 'Live face analysis.',
    copy: 'Your camera is starting. Face detection and AI estimates run locally in this browser.',
    action: null,
    progress: 'Requesting camera permission',
    busy: true,
  });
}

function showStoppedGate() {
  setGate({
    title: 'Camera is off.',
    copy: 'Detection is paused. Your camera feed never leaves this device, and no frames are stored.',
    action: 'Start camera',
    progress: null,
  });
}

let toastTimeout = null;

function showToast(message, duration = 4500) {
  el.toast.hidden = false;
  el.toast.textContent = message;
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    el.toast.hidden = true;
  }, duration);
}

/* ── Camera ──────────────────────────────────────────────────────────────── */

async function startCamera() {
  if (APP.cameraOn || APP.cameraStarting) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    showCameraError(new Error('unsupported'));
    return;
  }

  APP.cameraStarting = true;
  APP.cameraError = '';
  setStartingGate();
  setCameraState('starting');
  closeInfo();
  refreshStatus();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'user' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
    });

    const previousStream = el.video.srcObject;
    if (previousStream && previousStream !== stream) {
      previousStream.getTracks().forEach((track) => track.stop());
    }

    el.video.srcObject = stream;
    await waitForVideo(el.video);
    await el.video.play().catch(() => {});
    await nextAnimationFrame();

    const videoTrack = stream.getVideoTracks()[0];
    if (videoTrack) {
      const settings = videoTrack.getSettings ? videoTrack.getSettings() : {};
      APP.mirrored = settings.facingMode !== 'environment';
      el.video.classList.toggle('is-mirrored', APP.mirrored);
      videoTrack.addEventListener('ended', () => {
        if (APP.cameraOn) showCameraError({ name: 'NotReadableError' });
      });
    }

    APP.cameraOn = true;
    APP.cameraStarting = false;
    APP.cameraGeneration += 1;
    APP.detectionErrors = 0;
    APP.lastDetectAt = 0;

    setCameraState('live');
    refreshStatus();
    startRenderLoop();
    scheduleAnalysisLoop();
  } catch (error) {
    console.warn('[FACELENS] camera:', error?.name || error?.message || error);
    showCameraError(error);
    if (error?.name === 'NotAllowedError' && el.video.srcObject) {
      el.video.srcObject.getTracks?.().forEach((track) => track.stop());
      el.video.srcObject = null;
    }
  } finally {
    APP.cameraStarting = false;
    refreshStatus();
  }
}

function waitForVideo(video) {
  return new Promise((resolve) => {
    if (video.readyState >= 2 && video.videoWidth > 0) {
      resolve();
      return;
    }
    const done = () => {
      if (video.videoWidth > 0) resolve();
    };
    video.addEventListener('loadedmetadata', done, { once: true });
    video.addEventListener('canplay', done, { once: true });
    setTimeout(resolve, 2500);
  });
}

function nextAnimationFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function stopCamera({ fromButton = false } = {}) {
  APP.cameraOn = false;
  APP.cameraGeneration += 1;
  APP.cameraStarting = false;
  APP.cameraError = '';
  stopRenderLoop();
  clearTracks();
  stateFaceCount(0);
  el.scanHint.hidden = true;

  if (el.video.srcObject) {
    el.video.srcObject.getTracks().forEach((track) => track.stop());
    el.video.srcObject = null;
  }

  setCameraState('stopped');
  showStoppedGate();
  closeInfo();
  refreshStatus();
  if (fromButton) {
    setTimeout(() => el.gateAction.focus({ preventScroll: true }), 60);
  }
}

/* ── Models ──────────────────────────────────────────────────────────────── */

async function loadModels() {
  refreshStatus();
  const detectorTask = initialiseDetector();
  const analysisTask = initialiseAnalysis();

  const [detectorResult, analysisResult] = await Promise.allSettled([detectorTask, analysisTask]);
  APP.analysisSettled = true;

  if (detectorResult.status === 'rejected') {
    APP.detectorFailed = true;
    APP.detectorReady = false;
    console.error('[FACELENS] detector failed:', detectorResult.reason);
    showToast('Face detection could not start. Check the browser console, then reload.', 7000);
  }

  if (analysisResult.status === 'rejected') {
    console.error('[FACELENS] analysis failed:', analysisResult.reason);
  }

  APP.analysisReady = APP.ageReady || APP.expressionReady;
  refreshStatus();

  if (!APP.analysisReady) {
    APP.tracks.forEach((track) => markAnalysisError(track, true));
  }

  if (APP.detectorReady && APP.cameraOn) startRenderLoop();
  if (APP.analysisReady && APP.cameraOn) scheduleAnalysisLoop();
}

async function initialiseDetector() {
  try {
    const vision = await import(VISION_MODULE);
    const fileset = await vision.FilesetResolver.forVisionTasks(VISION_ROOT);

    const options = (delegate) => ({
      baseOptions: {
        modelAssetPath: FACE_MODEL,
        delegate,
      },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.48,
      minSuppressionThreshold: 0.3,
    });

    let detector;
    try {
      detector = await vision.FaceDetector.createFromOptions(fileset, options('GPU'));
      APP.detectorDelegate = 'GPU';
    } catch (gpuError) {
      console.warn('[FACELENS] GPU detector unavailable, falling back to CPU.', gpuError?.message);
      detector = await vision.FaceDetector.createFromOptions(fileset, options('CPU'));
      APP.detectorDelegate = 'CPU';
    }

    APP.detector = detector;
    APP.detectorReady = true;
    APP.detectorFailed = false;
    scheduleAnalysisLoop();
    refreshStatus();
    if (APP.cameraOn) startRenderLoop();
    return detector;
  } catch (error) {
    APP.detectorReady = false;
    APP.detectorFailed = true;
    refreshStatus();
    throw error;
  }
}

async function initialiseAnalysis() {
  if (!window.faceapi) {
    throw new Error('The local face-analysis runtime did not load.');
  }

  const tf = window.faceapi.tf;
  let backend = null;

  for (const candidate of ['webgl', 'cpu']) {
    try {
      await tf.setBackend(candidate);
      await tf.ready();
      backend = tf.getBackend();
      if (backend) break;
    } catch (error) {
      console.warn(`[FACELENS] TensorFlow ${candidate} backend unavailable.`, error?.message);
    }
  }

  if (!backend) throw new Error('No compatible TensorFlow.js backend was available.');
  console.info('[FACELENS] AI backend:', backend);

  const results = await Promise.allSettled([
    window.faceapi.nets.ageGenderNet.loadFromUri(MODEL_ROOT),
    window.faceapi.nets.faceExpressionNet.loadFromUri(MODEL_ROOT),
  ]);

  APP.ageReady = results[0].status === 'fulfilled';
  APP.expressionReady = results[1].status === 'fulfilled';

  if (!APP.ageReady) {
    console.error('[FACELENS] age model failed:', results[0].reason);
  }
  if (!APP.expressionReady) {
    console.error('[FACELENS] expression model failed:', results[1].reason);
  }

  APP.analysisReady = APP.ageReady || APP.expressionReady;
  refreshStatus();

  if (!APP.ageReady && !APP.expressionReady) {
    throw new Error('The local age and expression models could not be loaded.');
  }

  if (!APP.ageReady || !APP.expressionReady) {
    showToast('One AI estimate model did not load. Available detections and estimates remain local.', 6000);
  }

  scheduleAnalysisLoop();
  return { age: APP.ageReady, expression: APP.expressionReady };
}

/* ── Detection loop ──────────────────────────────────────────────────────── */

function startRenderLoop() {
  if (APP.rafId) return;
  APP.lastFrameAt = performance.now();
  APP.rafId = requestAnimationFrame(renderLoop);
}

function stopRenderLoop() {
  if (APP.rafId) cancelAnimationFrame(APP.rafId);
  APP.rafId = 0;
}

function renderLoop(now) {
  APP.rafId = requestAnimationFrame(renderLoop);

  const dt = clamp((now - APP.lastFrameAt) / 1000, 0.001, 0.05);
  APP.lastFrameAt = now;

  if (!APP.cameraOn) return;

  const videoReady = el.video.readyState >= 2 && el.video.videoWidth > 0;
  if (
    APP.detectorReady
    && APP.detector
    && videoReady
    && !document.hidden
    && now - APP.lastDetectAt >= APP.detectInterval
  ) {
    detectFaces(now);
  }

  renderTracks(now, dt);
}

function detectFaces(now) {
  APP.lastDetectAt = now;
  const started = performance.now();
  let result;

  try {
    const timestamp = Math.max(Math.round(now), APP.lastVideoTimestamp + 1);
    APP.lastVideoTimestamp = timestamp;
    result = APP.detector.detectForVideo(el.video, timestamp);
    APP.detectionErrors = 0;
  } catch (error) {
    APP.detectionErrors += 1;
    if (APP.detectionErrors === 1 || APP.detectionErrors % 12 === 0) {
      console.warn('[FACELENS] detection:', error?.message || error);
    }
    if (APP.detectionErrors >= 12) {
      showToast('The face detector hit a problem. It will keep retrying.', 3500);
      APP.detectionErrors = 0;
    }
    return;
  }

  const elapsed = performance.now() - started;
  APP.lastDetectionDuration = elapsed;
  APP.trackHideAfter = clamp(elapsed * 1.3 + 300, MIN_TRACK_HIDE_AFTER, 3000);
  APP.trackRemoveAfter = clamp(elapsed * 3.2 + 800, MIN_TRACK_REMOVE_AFTER, 7500);
  APP.reidWindow = clamp(8000 + elapsed * 2 + APP.lastAnalysisDuration * 1.5, 8000, 120000);
  APP.detectInterval = clamp(elapsed * 1.65 + 10, 48, 150);

  processDetections(result, now);
}

function processDetections(result, now) {
  const frameWidth = el.video.videoWidth;
  const frameHeight = el.video.videoHeight;
  if (!frameWidth || !frameHeight) return;

  const detections = (result?.detections || [])
    .map((detection) => {
      const box = detection.boundingBox;
      if (!box || box.width < 2 || box.height < 2) return null;
      const normalised = normaliseBox(box, frameWidth, frameHeight);
      if (normalised.w < 0.025 || normalised.h < 0.025) return null;
      return {
        box: normalised,
        score: clamp(detection.categories?.[0]?.score ?? 0.5, 0, 1),
      };
    })
    .filter(Boolean);

  const usableTracks = APP.tracks.filter((track) => now - track.lastSeen <= Math.max(APP.trackRemoveAfter, APP.reidWindow));
  const pairs = [];

  if (DEBUG) {
    console.log('[FACELENS debug] detection', {
      elapsed: Math.round(APP.lastDetectionDuration),
      interval: Math.round(APP.detectInterval),
      hideAfter: Math.round(APP.trackHideAfter),
      removeAfter: Math.round(APP.trackRemoveAfter),
      detections: detections.map((d) => ({ box: d.box, score: d.score })),
      tracks: usableTracks.map((t) => ({ id: t.id, target: t.target, seen: Math.round(now - t.lastSeen) })),
    });
  }

  detections.forEach((detection, detectionIndex) => {
    usableTracks.forEach((track, trackIndex) => {
      const distance = centerDistance(detection.box, track.target);
      const scale = Math.max(
        0.035,
        (detection.box.w + detection.box.h + track.target.w + track.target.h) / 4,
      );
      const iou = intersectionOverUnion(detection.box, track.target);
      const proximity = Math.max(0, 1 - distance / (scale * 1.75));
      if (iou < 0.015 && proximity <= 0.02) return;
      pairs.push({
        detectionIndex,
        trackIndex,
        score: iou * 4 + proximity + (detection.score * 0.1),
      });
    });
  });

  pairs.sort((a, b) => b.score - a.score);

  const usedDetections = new Set();
  const usedTracks = new Set();

  for (const pair of pairs) {
    if (usedDetections.has(pair.detectionIndex) || usedTracks.has(pair.trackIndex)) continue;
    const track = usableTracks[pair.trackIndex];
    const detection = detections[pair.detectionIndex];
    usedDetections.add(pair.detectionIndex);
    usedTracks.add(pair.trackIndex);

    track.target = detection.box;
    track.lastSeen = now;
    track.misses = 0;
    track.detectionCount += 1;
    track.confidence = track.detectionCount === 1
      ? detection.score
      : track.confidence * 0.72 + detection.score * 0.28;
  }

  detections.forEach((detection, index) => {
    if (usedDetections.has(index)) return;
    const rememberedId = takeRememberedFace(detection.box, now);
    const track = createTrack(detection, now, rememberedId);
    APP.tracks.push(track);
    APP.trackById.set(track.id, track);
  });

  APP.tracks.forEach((track) => {
    if (now - track.lastSeen > APP.trackRemoveAfter) {
      // Removed during render so all DOM state is cleaned in one place.
      track.expired = true;
    }
  });
}

function createTrack(detection, now, rememberedId = null) {
  const id = rememberedId ?? APP.nextFaceId++;
  const root = document.createElement('div');
  root.className = 'face-track';
  root.dataset.faceId = String(id);

  root.innerHTML = `
    <div class="face-box" aria-hidden="true"></div>
    <div class="face-label" data-analysis="pending" aria-hidden="true">
      <div class="label-head">
        <span class="face-id">FACE <span class="face-id-value"></span></span>
        <span class="confidence">CONF <strong class="confidence-value">—</strong></span>
      </div>
      <div class="label-body">
        <span class="metric">
          <span class="metric-name">Age estimate</span>
          <span class="metric-value age-value is-loading is-analyzing">—</span>
        </span>
        <span class="metric">
          <span class="metric-name">Gender</span>
          <span class="metric-value gender-value is-loading is-analyzing">—</span>
        </span>
        <span class="metric">
          <span class="metric-name">Expression</span>
          <span class="metric-value expression-value is-loading is-analyzing">—</span>
        </span>
      </div>
      <div class="label-foot">Analyzing face</div>
    </div>
  `;

  const elements = {
    box: root.querySelector('.face-box'),
    label: root.querySelector('.face-label'),
    idValue: root.querySelector('.face-id-value'),
    confidence: root.querySelector('.confidence-value'),
    age: root.querySelector('.age-value'),
    gender: root.querySelector('.gender-value'),
    expression: root.querySelector('.expression-value'),
    foot: root.querySelector('.label-foot'),
  };

  elements.idValue.textContent = padId(id);
  el.overlay.appendChild(root);

  const track = {
    id,
    root,
    elements,
    target: { ...detection.box },
    current: { ...detection.box },
    lastSeen: now,
    createdAt: now,
    confidence: detection.score,
    detectionCount: 1,
    misses: 0,
    lastAnalysisAt: 0,
    analysisPending: false,
    analysisError: false,
    analysis: null,
    ageSmoothed: null,
    maleProbability: null,
    gender: null,
    expressionScores: null,
    expression: null,
    lastConfidencePercent: null,
    labelPlacement: null,
    labelSize: null,
    expired: false,
  };

  if (APP.analysisSettled && !APP.analysisReady) {
    markAnalysisError(track, true);
  }

  return track;
}

function stateFaceCount(count) {
  if (APP.faceCount === count) return;
  APP.faceCount = count;
  if (count <= 0) {
    el.faceCount.hidden = true;
    return;
  }
  el.faceCount.hidden = false;
  el.faceCount.textContent = count === 1 ? '1 face' : `${count} faces`;
}

function clearTracks() {
  APP.tracks.forEach((track) => {
    track.root.remove();
  });
  APP.tracks = [];
  APP.trackById.clear();
  APP.faceMemory = [];
  stateFaceCount(0);
}

function pruneFaceMemory(now) {
  if (!APP.faceMemory.length) return;
  APP.faceMemory = APP.faceMemory.filter((entry) => now - entry.at <= APP.reidWindow);
}

function rememberFace(track, now) {
  pruneFaceMemory(now);
  APP.faceMemory = APP.faceMemory.filter((entry) => entry.id !== track.id);
  APP.faceMemory.push({ id: track.id, box: { ...track.target }, at: now });
}

function takeRememberedFace(box, now) {
  pruneFaceMemory(now);
  let bestIndex = -1;
  let bestScore = 0;

  APP.faceMemory.forEach((entry, index) => {
    const iou = intersectionOverUnion(box, entry.box);
    const distance = centerDistance(box, entry.box);
    const scale = Math.max(0.035, (box.w + box.h + entry.box.w + entry.box.h) / 4);
    const proximity = Math.max(0, 1 - distance / (scale * 1.6));
    if (iou < 0.02 && proximity <= 0.05) return;
    const score = iou * 3 + proximity;
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  });

  if (bestIndex < 0) return null;
  return APP.faceMemory.splice(bestIndex, 1)[0].id;
}

/* ── Projection + smooth rendering ───────────────────────────────────────── */

function projectBox(box) {
  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);
  const videoWidth = el.video.videoWidth || 1280;
  const videoHeight = el.video.videoHeight || 720;
  const scale = Math.max(width / videoWidth, height / videoHeight);
  const displayWidth = videoWidth * scale;
  const displayHeight = videoHeight * scale;
  const offsetX = (width - displayWidth) / 2;
  const offsetY = (height - displayHeight) / 2;

  const left = offsetX + box.x * displayWidth;
  const top = offsetY + box.y * displayHeight;
  const boxWidth = box.w * displayWidth;
  const boxHeight = box.h * displayHeight;

  if (APP.mirrored) {
    return {
      x: width - left - boxWidth,
      y: top,
      w: boxWidth,
      h: boxHeight,
    };
  }

  return { x: left, y: top, w: boxWidth, h: boxHeight };
}

function renderTracks(now, dt) {
  const smoothing = 1 - Math.exp(-dt * 15.5);
  const visible = [];
  const removals = [];

  for (const track of APP.tracks) {
    if (track.expired) {
      removals.push(track);
      continue;
    }

    const staleFor = now - track.lastSeen;
    if (staleFor > APP.trackRemoveAfter) {
      removals.push(track);
      continue;
    }

    track.current = lerpBox(track.current, track.target, smoothing);
    const projected = projectBox(track.current);
    if (projected.w < 8 || projected.h < 8) continue;

    const hidden = staleFor > APP.trackHideAfter;
    track.root.classList.toggle('is-stale', hidden);
    track.elements.box.style.transform = `translate3d(${projected.x.toFixed(2)}px, ${projected.y.toFixed(2)}px, 0)`;
    track.elements.box.style.width = `${projected.w.toFixed(2)}px`;
    track.elements.box.style.height = `${projected.h.toFixed(2)}px`;

    const confidence = Math.round(track.confidence * 100);
    if (confidence !== track.lastConfidencePercent) {
      track.lastConfidencePercent = confidence;
      safeText(track.elements.confidence, `${confidence}%`);
    }

    if (!hidden) visible.push({ track, projected });
  }

  removeTracks(removals, now);
  layoutLabels(visible);
  stateFaceCount(visible.length);

  const shouldHint = APP.cameraOn
    && APP.detectorReady
    && visible.length === 0
    && !document.hidden;
  el.scanHint.hidden = !shouldHint;
}

function removeTracks(tracks, now) {
  if (!tracks.length) return;
  const removeSet = new Set(tracks);
  tracks.forEach((track) => {
    rememberFace(track, now);
    track.root.remove();
    APP.trackById.delete(track.id);
  });
  APP.tracks = APP.tracks.filter((track) => !removeSet.has(track));
}

function layoutLabels(visible) {
  if (!visible.length) return;

  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const marginX = 10;
  const topLimit = Math.max(60, (el.topbar?.getBoundingClientRect().bottom || 44) + 8);
  const bottomInset = Math.max(48, viewportHeight - (el.bottombar?.getBoundingClientRect().top || viewportHeight - 44) + 8);
  const maxBottom = viewportHeight - bottomInset;
  const placed = [];
  const faces = visible.map((entry) => ({ ...entry.projected, track: entry.track }));

  const sorted = [...visible].sort((a, b) => a.projected.x - b.projected.x);

  for (const entry of sorted) {
    const { track, projected } = entry;
    const label = track.elements.label;
    const width = track.labelSize?.width || label.offsetWidth || 216;
    const height = track.labelSize?.height || label.offsetHeight || 72;
    track.labelSize = { width, height };

    const minY = topLimit;
    const maxY = Math.max(minY, maxBottom - height);

    const candidates = [
      { key: 'below-left', x: projected.x, y: projected.y + projected.h + LABEL_GAP, rank: 0 },
      { key: 'below-right', x: projected.x + projected.w - width, y: projected.y + projected.h + LABEL_GAP, rank: 1 },
      { key: 'above-left', x: projected.x, y: projected.y - height - LABEL_GAP, rank: 2 },
      { key: 'above-right', x: projected.x + projected.w - width, y: projected.y - height - LABEL_GAP, rank: 3 },
      { key: 'right', x: projected.x + projected.w + LABEL_GAP, y: projected.y + (projected.h - height) / 2, rank: 4 },
      { key: 'left', x: projected.x - width - LABEL_GAP, y: projected.y + (projected.h - height) / 2, rank: 5 },
      { key: 'below-center', x: projected.x + (projected.w - width) / 2, y: projected.y + projected.h + LABEL_GAP, rank: 6 },
    ];

    let best = null;

    for (const candidate of candidates) {
      const rawX = candidate.x;
      const rawY = candidate.y;
      const x = clamp(rawX, marginX, viewportWidth - width - marginX);
      const y = clamp(rawY, minY, maxY);
      const clampedDistance = Math.abs(rawX - x) + Math.abs(rawY - y);
      const rect = { x, y, w: width, h: height };

      let score = candidate.rank * 0.7 + clampedDistance * 2.2;
      if (track.labelPlacement === candidate.key) score -= 3.5;

      for (const prior of placed) {
        const overlap = intersectionArea(rect, prior);
        if (overlap > 0) score += 150000 + overlap * 4;
      }

      for (const face of faces) {
        if (face.track === track) continue;
        const overlap = intersectionArea(rect, face);
        if (overlap > 0) score += 4500 + overlap * 0.8;
      }

      score += intersectionArea(rect, projected) * 2.8;

      if (!best || score < best.score) {
        best = { ...candidate, x, y, score };
      }
    }

    if (!best) {
      best = {
        key: 'below-left',
        x: clamp(projected.x, marginX, viewportWidth - width - marginX),
        y: clamp(projected.y + projected.h + LABEL_GAP, minY, maxY),
      };
    }

    track.labelPlacement = best.key;
    label.style.transform = `translate3d(${best.x.toFixed(2)}px, ${best.y.toFixed(2)}px, 0)`;
    placed.push({ x: best.x, y: best.y, w: width, h: height });
  }
}

/* ── Age + expression analysis ───────────────────────────────────────────── */

function scheduleAnalysisLoop() {
  if (APP.analysisLoopId) return;
  APP.analysisLoopId = window.setInterval(analysisTick, 260);
}

function analysisTick() {
  if (
    !APP.cameraOn
    || APP.analysisBusy
    || !APP.analysisReady
    || APP.analysisLoopId === 0
    || document.hidden
    || !el.video.videoWidth
  ) {
    return;
  }

  const now = performance.now();
  const analysisGrace = Math.max(420, Math.min(APP.trackRemoveAfter - 100, 2200));
  const candidates = APP.tracks.filter((track) => (
    !track.expired
    && now - track.lastSeen <= analysisGrace
    && now - track.createdAt > 130
    && !track.analysisPending
    && now - track.lastAnalysisAt >= ANALYSIS_INTERVAL
  ));

  if (!candidates.length) return;
  candidates.sort((a, b) => a.lastAnalysisAt - b.lastAnalysisAt);
  if (DEBUG) console.log('[FACELENS debug] analysing track', candidates[0].id, candidates[0].target);
  analyseTrack(candidates[0]);
}

async function analyseTrack(track) {
  if (APP.analysisBusy || !APP.cameraOn) return;

  const crop = makeFaceCrop(track.target);
  if (!crop) return;

  const generation = APP.cameraGeneration;
  const analysisStarted = performance.now();
  // Detection can run while inference is awaiting. Keep recently seen faces
  // matchable through the whole analysis so temporary IDs do not churn.
  APP.reidWindow = Math.max(APP.reidWindow, 120000);
  APP.analysisBusy = true;
  track.analysisPending = true;
  track.lastAnalysisAt = analysisStarted;

  if (!track.analysis) {
    track.elements.label.dataset.analysis = 'pending';
    safeText(track.elements.age, '—');
    safeText(track.elements.gender, '—');
    safeText(track.elements.expression, '—');
    track.elements.age.classList.add('is-loading', 'is-analyzing');
    track.elements.gender.classList.add('is-loading', 'is-analyzing');
    track.elements.expression.classList.add('is-loading', 'is-analyzing');
    safeText(track.elements.foot, 'Analyzing face');
  }

  let age = null;
  let maleProbability = null;
  let expressionScores = null;
  let hadFailure = false;

  try {
    if (APP.ageReady) {
      try {
        const result = await window.faceapi.nets.ageGenderNet.predictAgeAndGender(crop);
        if (result) {
          if (Number.isFinite(result.age)) age = result.age;
          if (result.gender === 'male' || result.gender === 'female') {
            const probability = clamp(Number(result.genderProbability) || 0, 0, 1);
            maleProbability = result.gender === 'male' ? probability : 1 - probability;
          }
        }
      } catch (error) {
        hadFailure = true;
        console.warn('[FACELENS] age + gender estimate:', error?.message || error);
      }
    }

    if (generation !== APP.cameraGeneration || !APP.cameraOn) return;
    if (APP.expressionReady) {
      try {
        const result = await window.faceapi.nets.faceExpressionNet.predictExpressions(crop);
        if (result) expressionScores = result.asSortedArray();
      } catch (error) {
        hadFailure = true;
        console.warn('[FACELENS] expression estimate:', error?.message || error);
      }
    }

    if (generation !== APP.cameraGeneration || !APP.cameraOn || APP.trackById.get(track.id) !== track) {
      return;
    }

    if (age !== null) {
      if (track.ageSmoothed === null) track.ageSmoothed = age;
      else track.ageSmoothed = track.ageSmoothed * 0.62 + age * 0.38;
    }

    if (maleProbability !== null) {
      if (track.maleProbability === null) track.maleProbability = maleProbability;
      else track.maleProbability = track.maleProbability * 0.58 + maleProbability * 0.42;
      track.gender = track.maleProbability >= 0.5 ? 'Man' : 'Woman';
    }

    if (expressionScores) {
      updateExpressionScores(track, expressionScores);
    }

    if (age !== null || maleProbability !== null || expressionScores) {
      track.analysis = {
        age: track.ageSmoothed,
        gender: track.gender,
        expression: track.expression,
        at: Date.now(),
      };
      track.analysisError = false;
      renderAnalysis(track);
      if (DEBUG) {
        console.log('[FACELENS debug] analysis done', {
          id: track.id,
          age: track.ageSmoothed,
          gender: track.gender,
          expression: track.expression,
          ms: Math.round(performance.now() - analysisStarted),
        });
      }
    } else {
      markAnalysisError(track, hadFailure);
      if (DEBUG) console.log('[FACELENS debug] analysis empty', { id: track.id, age, maleProbability, expressionScores });
    }
  } catch (error) {
    console.warn('[FACELENS] analysis:', error?.message || error);
    markAnalysisError(track, true);
  } finally {
    APP.lastAnalysisDuration = performance.now() - analysisStarted;
    APP.reidWindow = clamp(
      8000 + APP.lastDetectionDuration * 2 + APP.lastAnalysisDuration * 1.5,
      8000,
      120000,
    );
    APP.analysisBusy = false;
    track.analysisPending = false;
  }
}

function updateExpressionScores(track, sorted) {
  const incoming = {};
  for (const item of sorted) incoming[item.expression] = item.probability;

  if (!track.expressionScores) {
    track.expressionScores = { ...incoming };
  } else {
    const keys = new Set([...Object.keys(track.expressionScores), ...Object.keys(incoming)]);
    keys.forEach((key) => {
      const previous = track.expressionScores[key] || 0;
      const next = incoming[key] || 0;
      track.expressionScores[key] = previous * 0.58 + next * 0.42;
    });
  }

  const best = Object.entries(track.expressionScores).sort((a, b) => b[1] - a[1])[0];
  if (best) track.expression = EXPRESSION_LABELS[best[0]] || 'Uncertain';
}

function renderAnalysis(track) {
  const label = track.elements.label;
  label.dataset.analysis = 'ready';
  safeText(track.elements.foot, 'Estimate · temporary ID');

  if (track.ageSmoothed !== null) {
    safeText(track.elements.age, formatAgeRange(track.ageSmoothed));
    track.elements.age.classList.remove('is-loading', 'is-analyzing');
  } else {
    safeText(track.elements.age, '—');
    track.elements.age.classList.add('is-loading');
    track.elements.age.classList.remove('is-analyzing');
  }

  if (track.gender) {
    safeText(track.elements.gender, track.gender);
    track.elements.gender.classList.remove('is-loading', 'is-analyzing');
  } else {
    safeText(track.elements.gender, '—');
    track.elements.gender.classList.add('is-loading');
    track.elements.gender.classList.remove('is-analyzing');
  }

  if (track.expression) {
    safeText(track.elements.expression, track.expression);
    track.elements.expression.classList.remove('is-loading', 'is-analyzing');
  } else {
    safeText(track.elements.expression, '—');
    track.elements.expression.classList.add('is-loading');
    track.elements.expression.classList.remove('is-analyzing');
  }

  if (track.ageSmoothed === null || !track.gender || !track.expression) {
    label.dataset.analysis = 'partial';
    const pieces = [];
    if (track.ageSmoothed === null) pieces.push('age');
    if (!track.gender) pieces.push('gender');
    if (!track.expression) pieces.push('expression');
    safeText(track.elements.foot, `${pieces.join(' + ')} unavailable`);
  }
}

function markAnalysisError(track, hadFailure) {
  if (track.analysis) return;
  track.analysisError = true;
  track.elements.label.dataset.analysis = 'error';
  safeText(track.elements.foot, hadFailure ? 'AI estimate unavailable' : 'Awaiting clearer face');
  safeText(track.elements.gender, '—');
  safeText(track.elements.expression, '—');
  track.elements.gender.classList.add('is-loading');
  track.elements.gender.classList.remove('is-analyzing');
  track.elements.expression.classList.add('is-loading');
  track.elements.expression.classList.remove('is-analyzing');
  track.elements.age.classList.remove('is-analyzing');
}

function formatAgeRange(age) {
  const centre = clamp(age, 2, 98);
  const lower = Math.max(1, Math.round(centre - 2.5));
  const upper = Math.min(100, Math.round(centre + 2.5));
  return `${lower}–${upper}`;
}

function makeFaceCrop(box) {
  const videoWidth = el.video.videoWidth;
  const videoHeight = el.video.videoHeight;
  if (!videoWidth || !videoHeight) return null;

  const centreX = (box.x + box.w / 2) * videoWidth;
  const centreY = (box.y + box.h / 2) * videoHeight;
  const faceWidth = box.w * videoWidth;
  const faceHeight = box.h * videoHeight;
  const side = clamp(Math.max(faceWidth, faceHeight) * 1.34, 60, Math.min(videoWidth, videoHeight));

  const sourceX = clamp(centreX - side / 2, 0, Math.max(0, videoWidth - side));
  const sourceY = clamp(centreY - side / 2, 0, Math.max(0, videoHeight - side));

  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: false });
  if (!context) return null;
  context.drawImage(el.video, sourceX, sourceY, side, side, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/* ── Info sheet ──────────────────────────────────────────────────────────── */

function openInfo() {
  if (APP.infoOpen) return;
  APP.infoOpen = true;
  APP.lastFocused = document.activeElement;
  el.infoBackdrop.hidden = false;
  el.infoSheet.hidden = false;
  document.body.classList.add('info-visible');
  setTimeout(() => el.closeInfo.focus({ preventScroll: true }), 60);
}

function closeInfo() {
  if (!APP.infoOpen) return;
  APP.infoOpen = false;
  el.infoBackdrop.hidden = true;
  el.infoSheet.hidden = true;
  document.body.classList.remove('info-visible');
  if (APP.lastFocused && typeof APP.lastFocused.focus === 'function') {
    APP.lastFocused.focus({ preventScroll: true });
  }
}

/* ── Fullscreen ──────────────────────────────────────────────────────────── */

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

async function toggleFullscreen() {
  const root = document.documentElement;
  const request = root.requestFullscreen || root.webkitRequestFullscreen;
  const exit = document.exitFullscreen || document.webkitExitFullscreen;

  try {
    if (fullscreenElement()) {
      if (exit) await exit.call(document);
    } else if (request) {
      await request.call(root);
    } else {
      showToast('Fullscreen is not supported by this browser.');
    }
  } catch (error) {
    console.warn('[FACELENS] fullscreen:', error?.message || error);
    showToast('Fullscreen could not be enabled.');
  }
  updateFullscreenUI();
}

function updateFullscreenUI() {
  const active = Boolean(fullscreenElement());
  document.body.dataset.fullscreen = active ? 'true' : 'false';
  el.fullscreenButton.setAttribute('aria-label', active ? 'Exit fullscreen' : 'Enter fullscreen');
  el.fullscreenButton.title = active ? 'Exit fullscreen' : 'Fullscreen';
}

/* ── Events + boot ───────────────────────────────────────────────────────── */

function bindEvents() {
  el.gateAction.addEventListener('click', () => {
    if (!window.isSecureContext && location.protocol === 'http:') {
      location.reload();
      return;
    }
    startCamera();
  });

  el.stopButton.addEventListener('click', () => {
    if (APP.cameraOn) stopCamera({ fromButton: true });
    else startCamera();
  });

  el.fullscreenButton.addEventListener('click', toggleFullscreen);
  el.infoButton.addEventListener('click', openInfo);
  el.closeInfo.addEventListener('click', closeInfo);
  el.infoDismiss.addEventListener('click', closeInfo);
  el.infoBackdrop.addEventListener('click', closeInfo);

  document.addEventListener('fullscreenchange', updateFullscreenUI);
  document.addEventListener('webkitfullscreenchange', updateFullscreenUI);

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      APP.lastDetectAt = 0;
    }
  });

  window.addEventListener('resize', () => {
    APP.tracks.forEach((track) => {
      track.labelSize = null;
    });
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && APP.infoOpen) {
      event.preventDefault();
      closeInfo();
    }
  });

  const stopStream = () => {
    if (el.video.srcObject) {
      el.video.srcObject.getTracks().forEach((track) => track.stop());
    }
    if (APP.detector?.close) {
      try { APP.detector.close(); } catch (error) { /* already closed */ }
    }
  };
  window.addEventListener('pagehide', stopStream, { once: true });
  window.addEventListener('beforeunload', stopStream, { once: true });
}

async function boot() {
  bindEvents();
  updateFullscreenUI();

  if (!('requestFullscreen' in document.documentElement) && !('webkitRequestFullscreen' in document.documentElement)) {
    el.fullscreenButton.hidden = true;
  }

  // The detector and age/expression networks start loading while the camera
  // permission prompt is open, so the first face can be analysed sooner.
  loadModels();
  startCamera();
}

boot();
