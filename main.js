import vision from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/+esm';
const { HandLandmarker, FilesetResolver } = vision;

const video        = document.getElementById('webcam');
const canvas       = document.getElementById('skeleton-canvas');
const ctx          = canvas.getContext('2d');
const hamsterImg   = document.getElementById('hamster-img');
const statusText   = document.getElementById('status-text');
const loadingDiv   = document.getElementById('loading');
const errorMsg     = document.getElementById('error-msg');
const loadingTitle = document.getElementById('loading-title');
const loadingSub   = document.getElementById('loading-sub');
const spinnerEl    = document.getElementById('spinner');
const handLabel    = document.getElementById('hand-label');
const countLabel   = document.getElementById('count-label');
const badgeHand    = document.getElementById('badge-hand');
const badgeCount   = document.getElementById('badge-count');
const handLeftEl   = document.getElementById('hand-left');
const handRightEl  = document.getElementById('hand-right');
const chipPalm     = document.getElementById('chip-palm');
const chipBack     = document.getElementById('chip-back');

let handLandmarker;
let lastVideoTime = -1;
let currentKey = '';
let stableKey = '';
let stableCount = 0;
const STABLE_FRAMES = 4;

// ─── 錯誤顯示 ───────────────────────────────────────────────
function showError(msg) {
  spinnerEl.style.display = 'none';
  loadingTitle.innerText = '發生錯誤';
  loadingSub.style.display = 'none';
  errorMsg.style.display = 'block';
  errorMsg.innerText = msg;
}

// ─── 圖片切換（含穩定幀機制）────────────────────────────────
function setImage(key, imgSrc, label) {
  if (key === currentKey) return;
  currentKey = key;
  hamsterImg.classList.remove('pop-anim');
  void hamsterImg.offsetWidth;
  hamsterImg.classList.add('pop-anim');
  hamsterImg.src = imgSrc;
  statusText.innerText = label;
}

function setImageStable(key, imgSrc, label) {
  if (key === stableKey) { stableCount++; }
  else { stableKey = key; stableCount = 1; }
  if (stableCount >= STABLE_FRAMES) { setImage(key, imgSrc, label); }
}

// ─── 骨架繪製 ────────────────────────────────────────────────
// 修正 1: 畫布需要水平翻轉，與鏡像影片保持一致
const HAND_CONNECTIONS = [
  [0,1],[0,5],[0,17],[5,9],[9,13],[13,17],
  [1,2],[2,3],[3,4],
  [5,6],[6,7],[7,8],
  [9,10],[10,11],[11,12],
  [13,14],[14,15],[15,16],
  [17,18],[18,19],[19,20],
];

const FINGER_COLORS = {
  thumb:'#f59e0b', index:'#34d399', middle:'#60a5fa',
  ring:'#f472b6',  pinky:'#a78bfa', palm:'rgba(255,255,255,0.55)',
};

function getLandmarkColor(idx) {
  if (idx <= 4)  return FINGER_COLORS.thumb;
  if (idx <= 8)  return FINGER_COLORS.index;
  if (idx <= 12) return FINGER_COLORS.middle;
  if (idx <= 16) return FINGER_COLORS.ring;
  if (idx <= 20) return FINGER_COLORS.pinky;
  return FINGER_COLORS.palm;
}

function drawSkeleton(results) {
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!results.landmarks || results.landmarks.length === 0) return;

  results.landmarks.forEach((landmarks) => {
    // 畫骨骼連線
    HAND_CONNECTIONS.forEach(([i, j]) => {
      const lA = landmarks[i], lB = landmarks[j];
      ctx.beginPath();
      ctx.moveTo(lA.x * W, lA.y * H);
      ctx.lineTo(lB.x * W, lB.y * H);
      ctx.strokeStyle = getLandmarkColor(Math.max(i, j));
      ctx.lineWidth = 2.8;
      ctx.stroke();
    });

    // 畫關節點
    landmarks.forEach((lm, idx) => {
      const isTip = [4, 8, 12, 16, 20].includes(idx);
      ctx.beginPath();
      ctx.arc(lm.x * W, lm.y * H, isTip ? 6 : (idx === 0 ? 7 : 4), 0, 2 * Math.PI);
      ctx.fillStyle = getLandmarkColor(idx);
      ctx.fill();
      if (isTip) {
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = 'white';
        ctx.fill();
      }
    });
  });
}

// ─── 手心 / 手背判斷 ─────────────────────────────────────────
// 修正 2：外積法，考慮前鏡頭左右翻轉
function detectPalmOrBack(landmarks, handedness) {
  const wrist    = landmarks[0];
  const indexMCP = landmarks[5];
  const pinkyMCP = landmarks[17];
  const v1 = { x: indexMCP.x - wrist.x, y: indexMCP.y - wrist.y };
  const v2 = { x: pinkyMCP.x - wrist.x, y: pinkyMCP.y - wrist.y };
  const cross = v1.x * v2.y - v1.y * v2.x;
  // MediaPipe 前鏡頭：你的右手會被辨識為 'Left'
  // 反轉判斷邏輯：解決手心手背判定相反的問題
  if (handedness === 'Left') return cross < 0 ? 'palm' : 'back';
  else                        return cross > 0 ? 'palm' : 'back';
}

// ─── 手指伸展偵測 ────────────────────────────────────────────
// 【關節角度法】：計算三關節形成的夾角 cos 值，完全不受手部旋轉方向影響
// 距離法在手傾斜時會失準；角度法只看關節彎曲程度，與手的朝向無關
function dist(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

// 手掌中心（拇指展開判斷用）
function palmCenter(lm) {
  const ids = [0, 5, 9, 13, 17];
  return {
    x: ids.reduce((s, i) => s + lm[i].x, 0) / ids.length,
    y: ids.reduce((s, i) => s + lm[i].y, 0) / ids.length,
  };
}

function getFingerStates(landmarks) {
  const L = landmarks;
  const palm = palmCenter(L);

  // a─b─c 三點在 b 的夾角餘弦值（內積法）―含 z 軸（真實 3D 角度）
  //   cos = -1 → 180°（完全伸直）  cos =  0 → 90°（直角彎曲）  cos = 1 → 0°（完全反折）
  // ❗ 加入 z 軸是關鍵：手面對鏡頭時，手指彎曲方向是深度軸，2D 算法根本無法偵測
  function jCos(ai, bi, ci) {
    const v1x = L[ai].x - L[bi].x, v1y = L[ai].y - L[bi].y, v1z = (L[ai].z ?? 0) - (L[bi].z ?? 0);
    const v2x = L[ci].x - L[bi].x, v2y = L[ci].y - L[bi].y, v2z = (L[ci].z ?? 0) - (L[bi].z ?? 0);
    const dot = v1x * v2x + v1y * v2y + v1z * v2z;
    const mag = Math.sqrt((v1x*v1x + v1y*v1y + v1z*v1z) * (v2x*v2x + v2y*v2y + v2z*v2z));
    return mag > 1e-6 ? dot / mag : 1;
  }
  // 四指伸直判斷：
  //   PIP 關節（第一道彎）夾角 > 120°  → jCos < -0.5
  //   DIP 關節（第二道彎）夾角 > 90°   → jCos < 0
  //   兩關節同時偏直，才視為「伸直」；只要任一關節彎曲就視為「彎曲」
  const iPIP = jCos(5,  6,  7),  iDIP = jCos(6,  7,  8);
  const mPIP = jCos(9,  10, 11), mDIP = jCos(10, 11, 12);
  const rPIP = jCos(13, 14, 15), rDIP = jCos(14, 15, 16);
  const pPIP = jCos(17, 18, 19), pDIP = jCos(18, 19, 20);
  const tIP  = jCos(2,  3,  4);

  const indexUp  = iPIP < -0.5 && iDIP < 0;
  const middleUp = mPIP < -0.5 && mDIP < 0;
  const ringUp   = rPIP < -0.5 && rDIP < 0;
  const pinkyUp  = pPIP < -0.5 && pDIP < 0;

  // 拇指伸直：
  //   IP 關節夾角 > 120°（拇指本身不彎曲）
  //   指尖 (4) 比 MCP (2) 更遠離手掌中心（拇指確實展開而非貼著手）
  const thumbUp = tIP < -0.5 && dist(L[4], palm) > dist(L[2], palm) * 1.05;

  // _raw 供 Debug 面板顯示原始角度數據
  return { thumbUp, indexUp, middleUp, ringUp, pinkyUp,
           _raw: { iPIP, mPIP, rPIP, pPIP, tIP } };
}

function countFingers(f) {
  return [f.thumbUp, f.indexUp, f.middleUp, f.ringUp, f.pinkyUp].filter(Boolean).length;
}

// ─── 手勢分類 ────────────────────────────────────────────────
function classifyGesture(fingers, numHands, allFingers, allHandedness, allLandmarks) {
  const { thumbUp, indexUp, middleUp, ringUp, pinkyUp } = fingers;

  // 雙手手勢
  if (numHands === 2 && allFingers && allFingers.length === 2) {
    const f1 = allFingers[0], f2 = allFingers[1];
    const c1 = countFingers(f1), c2 = countFingers(f2);

    if (f1.indexUp && f1.middleUp && !f1.ringUp && !f1.pinkyUp &&
        f2.indexUp && f2.middleUp && !f2.ringUp && !f2.pinkyUp) return 'double_victory';
    if (f1.thumbUp && c1 <= 2 && f2.thumbUp && c2 <= 2)          return 'double_thumbsup';
    if (c1 >= 4 && c2 >= 4)                                       return 'double_open';
    if (c1 === 0 && c2 === 0)                                     return 'double_fist';

    const side1 = detectPalmOrBack(allLandmarks[0], allHandedness[0]);
    const side2 = detectPalmOrBack(allLandmarks[1], allHandedness[1]);
    if ((side1 === 'palm' && side2 === 'back') ||
        (side1 === 'back' && side2 === 'palm')) return 'two_hands_sides';

    return 'unknown';
  }

  // 單手手勢
  const side = allLandmarks && allLandmarks[0]
    ? detectPalmOrBack(allLandmarks[0], allHandedness[0])
    : 'palm';

  const count = countFingers(fingers);
  if (count === 0) return 'fist';

  // 比讚：大拇指伸直，其餘四指（含食指）均彎曲
  if (thumbUp && !indexUp && !middleUp && !ringUp && !pinkyUp) return 'thumbsup';
  if (indexUp && middleUp && !ringUp && !pinkyUp)                           return 'victory';
  if (indexUp && !middleUp && !ringUp && !pinkyUp)                          return 'pointing';
  if (count >= 4) return side === 'back' ? 'open_back' : 'open_palm';
  if (count === 3) return 'three';

  return 'unknown';
}

// ─── 手勢對應圖片 ────────────────────────────────────────────
const GESTURE_MAP = {
  'double_victory':  { img: './assets/double_ya.jpg', label: '✌️✌️ 你是棒鼠鼠' },
  'double_thumbsup': { img: './assets/cheers.jpg',    label: '👍👍 鼠鼠歡呼！' },
  'double_open':     { img: './assets/scared.jpg',    label: '🖐️🖐️ 鼠鼠嚇到！' },
  'double_fist':     { img: './assets/angry.jpg',     label: '✊✊ 氣氣鼠！' },
  'two_hands_sides': { img: './assets/heart.jpg',     label: '🤲 鼠鼠的愛！' },
  'victory':         { img: './assets/ya.jpg',        label: '✌️ 鼠鼠手收！' },
  'thumbsup':        { img: './assets/haha.jpg',      label: '👍 開心鼠鼠！' },
  'pointing':        { img: './assets/cool.png',      label: '☝️ 酷鼠鼠！' },
  'open_palm':       { img: './assets/wait.jpg',      label: '🖐️ 鼠鼠嗨！' },
  'open_back':       { img: './assets/shy.jpg',       label: '🫲 鼠鼠害羞！' },
  'fist':            { img: './assets/chef.jpg',      label: '✊ 料理鼠王！' },
  'three':           { img: './assets/confuse.jpg',   label: '🤔 鼠鼠困惑！' },
  'unknown':         { img: './assets/confuse.jpg',   label: '❓ 鼠鼠困惑...' },
};

// ─── UI 更新 ─────────────────────────────────────────────────
function updateHandUI(numHands, allHandedness, allLandmarks) {
  handLeftEl.classList.toggle('active', numHands >= 1);
  handRightEl.classList.toggle('active', numHands >= 2);
  countLabel.innerText = numHands + ' 隻手';
  badgeCount.classList.toggle('active-hand', numHands > 0);

  if (numHands === 0) {
    chipPalm.classList.remove('active');
    chipBack.classList.remove('active');
    return;
  }

  let hasPalm = false, hasBack = false;
  for (let i = 0; i < numHands; i++) {
    const side = detectPalmOrBack(allLandmarks[i], allHandedness[i]);
    if (side === 'palm') hasPalm = true;
    if (side === 'back') hasBack = true;
  }
  chipPalm.classList.toggle('active', hasPalm);
  chipBack.classList.toggle('active', hasBack);
}

// ─── Debug 即時狀態面板 ─────────────────────────────────────
let _debugEl = null;
function _getDebugEl() {
  if (!_debugEl) {
    _debugEl = document.createElement('div');
    _debugEl.id = 'debug-panel';
    _debugEl.style.cssText = [
      'position:fixed', 'bottom:16px', 'left:16px',
      'background:rgba(8,8,20,0.88)',
      'color:#e2e8f0', 'padding:12px 14px',
      'border-radius:14px', 'font-family:"Courier New",monospace',
      'font-size:13px', 'line-height:1.6',
      'z-index:9999', 'backdrop-filter:blur(10px)',
      'border:1px solid rgba(148,163,184,0.2)',
      'min-width:140px', 'pointer-events:none',
      'box-shadow:0 4px 24px rgba(0,0,0,0.5)',
    ].join(';');
    document.body.appendChild(_debugEl);
  }
  return _debugEl;
}

function updateDebug(fingers, gestureKey, numHands) {
  const el = _getDebugEl();
  if (!fingers || numHands === 0) {
    el.innerHTML = '<span style="color:#64748b;font-size:12px">🔍 等待偵測手部...</span>';
    return;
  }
  const { thumbUp, indexUp, middleUp, ringUp, pinkyUp, _raw } = fingers;
  // 顯示伸直/彎曲 + 實際 PIP cos 數值（越負 = 越直）
  const row = (label, up, cos) => {
    // 1. 處理數值格式
    const val = cos !== undefined ? cos.toFixed(2) : '';
    const isNegative = val.startsWith('-');
    const numberValue = isNegative ? val.substring(1) : val; // 取出純數字部分
    const sign = isNegative ? '-' : ' '; // 符號部分

    // 2. 狀態顏色顯示
    const state = up
      ? '<span style="color:#4ade80;font-weight:bold">▶ 伸</span>'
      : '<span style="color:#f87171">▷ 彎</span>';

    // 3. 組合 HTML
    // 注意：我們把符號放在一個寬度固定的 span 裡，強制對齊
    return `<div style="display:flex;justify-content:space-between;margin-bottom:2px;font-family:monospace;">
              <span>${label}</span>
              <span style="display:flex; align-items:center;">
                  ${state}
                  <span style="font-variant-numeric: tabular-nums; margin-left: 8px;">
                      (<span style="display:inline-block; width:0.6em; text-align:center;">${sign}</span>${numberValue})
                  </span>
              </span>
            </div>`;
  };
  el.innerHTML = `
    <div style="color:#fbbf24;margin-bottom:8px;font-size:11px;font-weight:bold;letter-spacing:1px;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:4px">
      🔍 狀態 <span style="color:#475569">(cos)</span>
    </div>
    ${row('拇指', thumbUp, _raw?.tIP)}
    ${row('食指', indexUp, _raw?.iPIP)}
    ${row('中指', middleUp, _raw?.mPIP)}
    ${row('無名', ringUp, _raw?.rPIP)}
    ${row('小指', pinkyUp, _raw?.pPIP)}
    <div style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(148,163,184,0.2);color:#93c5fd;font-size:11px;display:flex;justify-content:space-between">
      <span>結果:</span> <b style="color:#e0f2fe">${gestureKey || '—'}</b>
    </div>`;
}

// ─── 主偵測迴圈 ──────────────────────────────────────────────
function predictWebcam() {
  // 修正 4：等影片真正有尺寸才開始處理，防止崩潰
  if (video.videoWidth === 0) {
    requestAnimationFrame(predictWebcam);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // 讓 canvas 與影片解析度同步
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ts = performance.now();
    let numHands = 0, allFingers = [], allHandedness = [], allLandmarks = [], gestureKey = null;

    try {
      const hr = handLandmarker.detectForVideo(video, ts);
      numHands = hr.landmarks ? hr.landmarks.length : 0;

      if (numHands > 0) {
        allLandmarks  = hr.landmarks;
        allHandedness = (hr.handednesses || hr.handedness).map(h => h[0].categoryName);

        for (let i = 0; i < numHands; i++) {
          allFingers.push(getFingerStates(hr.landmarks[i]));
        }

        gestureKey = classifyGesture(
          allFingers[0], numHands, allFingers, allHandedness, allLandmarks
        );

        if (gestureKey && GESTURE_MAP[gestureKey]) {
          handLabel.innerText = GESTURE_MAP[gestureKey].label.split('→')[0].trim();
          badgeHand.classList.add('active-hand');
        }
        // 即時更新 Debug 面板（顯示第一隻手的手指狀態）
        updateDebug(allFingers[0], gestureKey, numHands);
      } else {
        handLabel.innerText = '—';
        badgeHand.classList.remove('active-hand');
        updateDebug(null, null, 0);
      }

      drawSkeleton(hr);
    } catch (e) {
      console.error(e);
      document.getElementById('status-text').innerText = 'Error: ' + e.message;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    updateHandUI(numHands, allHandedness, allLandmarks);

    if (gestureKey && gestureKey !== 'unknown' && GESTURE_MAP[gestureKey]) {
      setImageStable('hand_' + gestureKey, GESTURE_MAP[gestureKey].img, GESTURE_MAP[gestureKey].label);
    } else if (numHands > 0) {
      setImageStable('hand_unknown', GESTURE_MAP.unknown.img, GESTURE_MAP.unknown.label);
    } else {
      setImageStable('face_neutral', './assets/wait.jpg', '🐹 等你伸出手收...');
    }
  }

  requestAnimationFrame(predictWebcam);
}

// ─── 啟動攝影機 ──────────────────────────────────────────────
async function startWebcam() {
  loadingTitle.innerText = '請允許使用攝影機...';
  loadingSub.innerText   = '瀏覽器將跳出請求，請點擊「允許」';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
    });
    video.srcObject = stream;
    await video.play();

    // 修正 5：等 video 真正有尺寸後才隱藏 loading，避免第一幀崩潰
    await new Promise(resolve => {
      if (video.videoWidth > 0) return resolve();
      video.addEventListener('loadeddata', resolve, { once: true });
    });

    loadingDiv.classList.add('hidden');
    predictWebcam();
  } catch (err) {
    showError('無法存取攝影機\n\n' + err.message);
  }
}

// ─── 初始化 ──────────────────────────────────────────────────
async function init() {
  try {
    loadingTitle.innerText = '🐹 載入模型中...';
    loadingSub.innerText   = '正在載入手勢辨識模型（約 5~10 秒）';

    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
    );

    handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    });

    await startWebcam();
  } catch (err) {
    showError('初始化失敗\n\n' + err.message);
  }
}

init();