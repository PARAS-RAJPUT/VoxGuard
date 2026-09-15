const API_BASE = "http://localhost:8000";

// ---------------------------------------------------------------------
// Enroll
// ---------------------------------------------------------------------
document.getElementById("enrollBtn").addEventListener("click", async () => {
  const name = document.getElementById("enrollName").value.trim();
  const fileInput = document.getElementById("enrollFile");
  const statusEl = document.getElementById("enrollStatus");

  if (!name || !fileInput.files[0]) {
    statusEl.textContent = "Please provide a name and a .wav file.";
    statusEl.className = "status error";
    return;
  }

  const formData = new FormData();
  formData.append("name", name);
  formData.append("file", fileInput.files[0]);

  statusEl.textContent = "Enrolling...";
  statusEl.className = "status";

  try {
    const res = await fetch(`${API_BASE}/enroll`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    statusEl.textContent = `Enrolled '${data.name}' successfully.`;
    statusEl.className = "status success";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}. Is the backend running on ${API_BASE}?`;
    statusEl.className = "status error";
  }
});

// ---------------------------------------------------------------------
// Analyze (file upload)
// ---------------------------------------------------------------------
document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("analyzeFile");
  const statusEl = document.getElementById("analyzeStatus");
  const btn = document.getElementById("analyzeBtn");

  if (!fileInput.files[0]) {
    statusEl.textContent = "Please choose a .wav file first.";
    statusEl.className = "status error";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  statusEl.textContent = "Analyzing... (this can take a few seconds)";
  statusEl.className = "status";
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();

    renderResult(data);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}. Is the backend running on ${API_BASE}?`;
    statusEl.className = "status error";
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------------
// Live Microphone Monitoring
// ---------------------------------------------------------------------
const CHUNK_MS = 4000; // 4 seconds per chunk

let liveStream = null;
let liveRunning = false;

const startLiveBtn = document.getElementById("startLiveBtn");
const stopLiveBtn = document.getElementById("stopLiveBtn");
const liveStatus = document.getElementById("liveStatus");

startLiveBtn.addEventListener("click", async () => {
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    liveStatus.textContent = "Microphone access denied or unavailable.";
    liveStatus.className = "status error";
    return;
  }

  liveRunning = true;
  startLiveBtn.disabled = true;
  stopLiveBtn.disabled = false;
  liveStatus.textContent = "Live monitoring started...";
  liveStatus.className = "status";

  recordLoop();
});

stopLiveBtn.addEventListener("click", () => {
  liveRunning = false;
  if (liveStream) {
    liveStream.getTracks().forEach(track => track.stop());
  }
  startLiveBtn.disabled = false;
  stopLiveBtn.disabled = true;
  liveStatus.textContent = "Live monitoring stopped.";
});

async function recordLoop() {
  while (liveRunning) {
    const chunkBlob = await recordChunk(liveStream, CHUNK_MS);
    if (!liveRunning) break;

    liveStatus.textContent = "Analyzing chunk...";

    try {
      const formData = new FormData();
      formData.append("file", chunkBlob, "live_chunk.webm");

      const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();

      renderResult(data);
      liveStatus.textContent = "Listening...";
    } catch (err) {
      liveStatus.textContent = `Error analyzing chunk: ${err.message}`;
      liveStatus.className = "status error";
    }
  }
}

function recordChunk(stream, durationMs) {
  return new Promise((resolve) => {
    const recorder = new MediaRecorder(stream);
    const chunks = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => resolve(new Blob(chunks, { type: "audio/webm" }));

    recorder.start();
    setTimeout(() => recorder.stop(), durationMs);
  });
}

// ---------------------------------------------------------------------
// Shared result rendering (used by both file-upload and live analysis)
// ---------------------------------------------------------------------
function renderResult(data) {
  document.getElementById("resultCard").classList.remove("hidden");

  document.getElementById("metricSynthetic").textContent =
    `${(data.synthetic_probability * 100).toFixed(1)}%`;
  document.getElementById("metricWatchlist").textContent =
    data.watchlist_match ? "Yes" : "No";
  document.getElementById("metricIntent").textContent = data.intent_category;
  document.getElementById("metricSpeaker").textContent = data.matched_name || "—";

  const riskDot = document.getElementById("riskDot");
  riskDot.className = "risk-dot";
  if (data.risk_level === "MEDIUM-RISK CALL") riskDot.classList.add("medium");
  if (data.risk_level === "HIGH-RISK CALL") riskDot.classList.add("high");

  document.getElementById("riskLevel").textContent = data.risk_level;
  document.getElementById("riskScore").textContent = `${data.risk_score}/100`;

  const reasonsList = document.getElementById("reasonsList");
  reasonsList.innerHTML = "";
  data.reasons.forEach(reason => {
    const li = document.createElement("li");
    li.textContent = reason;
    reasonsList.appendChild(li);
  });

  document.getElementById("actionText").textContent = data.recommended_action;
  document.getElementById("transcriptText").textContent = data.transcript;
}