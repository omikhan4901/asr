// ============================================================
// Bangla Subtitle Generator — frontend logic
//
// Expects a backend exposing (see web/backend/app.py):
//   POST /jobs                    multipart file=<video>  -> { job_id }
//   GET  /jobs/{job_id}           -> { status, stage, progress, error, result }
//   GET  /jobs/{job_id}/download  -> the .srt file
//
// status: "queued" | "processing" | "done" | "error"
// stage:  "upload" | "audio" | "chunk" | "asr" | "rescore" | "srt" | null
// result: { n_segments, duration_seconds, srt_preview } | null
// ============================================================

const API_BASE = window.ASR_API_BASE || "https://mehboobehsankhan--bangla-subtitle-generator-web-dev.modal.run"; // set window.ASR_API_BASE before this script loads to point at your deployed Modal endpoint

const els = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  filebar: document.getElementById("filebar"),
  fileName: document.getElementById("fileName"),
  startBtn: document.getElementById("startBtn"),

  uploadSection: document.getElementById("uploadSection"),
  processingSection: document.getElementById("processingSection"),
  resultSection: document.getElementById("resultSection"),
  errorSection: document.getElementById("errorSection"),

  steps: document.getElementById("steps"),
  procCaptionText: document.getElementById("procCaptionText"),

  metaSegments: document.getElementById("metaSegments"),
  metaDuration: document.getElementById("metaDuration"),
  downloadBtn: document.getElementById("downloadBtn"),
  srtPreview: document.getElementById("srtPreview"),
  resetBtn: document.getElementById("resetBtn"),

  errorMsg: document.getElementById("errorMsg"),
  errorResetBtn: document.getElementById("errorResetBtn"),
};

let selectedFile = null;
let pollTimer = null;

const STEP_ORDER = ["upload", "audio", "chunk", "asr", "rescore", "srt"];

const PROCESSING_CAPTIONS = [
  "শুরু হচ্ছে…",
  "অডিও আলাদা করা হচ্ছে…",
  "কথা ভাগ করা হচ্ছে…",
  "প্রতিলিপি তৈরি হচ্ছে…",
  "সেরা প্রার্থী বাছাই হচ্ছে…",
  "সাবটাইটেল তৈরি হচ্ছে…",
];
let captionIdx = 0;
let captionTimer = null;

function cycleProcessingCaption() {
  els.procCaptionText.style.opacity = 0;
  setTimeout(() => {
    captionIdx = (captionIdx + 1) % PROCESSING_CAPTIONS.length;
    els.procCaptionText.textContent = PROCESSING_CAPTIONS[captionIdx];
    els.procCaptionText.style.opacity = 1;
  }, 220);
}

function showSection(name) {
  ["uploadSection", "processingSection", "resultSection", "errorSection"].forEach((s) => {
    els[s].hidden = s !== name;
  });
}

// ---------- file selection ----------

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  els.fileName.textContent = `${file.name} — ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
  els.filebar.hidden = false;
}

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
});
els.fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => { e.preventDefault(); els.dropzone.classList.add("drag-over"); })
);
["dragleave", "drop"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag-over"); })
);
els.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  setFile(file);
});

// ---------- start job ----------

els.startBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  showSection("processingSection");
  resetSteps();
  setStepState("upload", "active");
  captionIdx = 0;
  els.procCaptionText.textContent = PROCESSING_CAPTIONS[0];
  captionTimer = setInterval(cycleProcessingCaption, 3200);

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch(`${API_BASE}/jobs`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    const { job_id } = await res.json();

    setStepState("upload", "done");
    pollJob(job_id);
  } catch (err) {
    showError(err.message || "Could not start the job. Check your connection and try again.");
  }
});

// ---------- polling ----------

function pollJob(jobId) {
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/jobs/${jobId}`);
      if (!res.ok) throw new Error(`Status check failed (${res.status})`);
      const data = await res.json();
      applyStatus(data, jobId);
    } catch (err) {
      clearInterval(pollTimer);
      clearInterval(captionTimer);
      showError(err.message || "Lost connection while checking job status.");
    }
  }, 2000);
}

function applyStatus(data, jobId) {
  if (data.stage) {
    const idx = STEP_ORDER.indexOf(data.stage);
    STEP_ORDER.forEach((s, i) => {
      if (i < idx) setStepState(s, "done");
      else if (i === idx) setStepState(s, "active");
    });
  }

  if (data.status === "done") {
    clearInterval(pollTimer);
    clearInterval(captionTimer);
    STEP_ORDER.forEach((s) => setStepState(s, "done"));
    setTimeout(() => showResult(data.result, jobId), 400);
  } else if (data.status === "error") {
    clearInterval(pollTimer);
    clearInterval(captionTimer);
    showError(data.error || "Processing failed unexpectedly.");
  }
}

function resetSteps() {
  els.steps.querySelectorAll("li").forEach((li) => li.classList.remove("active", "done"));
}
function setStepState(step, state) {
  const li = els.steps.querySelector(`[data-step="${step}"]`);
  if (!li) return;
  li.classList.remove("active", "done");
  li.classList.add(state);
}

// ---------- result ----------

function showResult(result, jobId) {
  els.metaSegments.textContent = result?.n_segments ?? "—";
  els.metaDuration.textContent = result?.duration_seconds
    ? formatDuration(result.duration_seconds)
    : "—";
  els.srtPreview.textContent = result?.srt_preview || "";
  els.downloadBtn.href = `${API_BASE}/jobs/${jobId}/download`;
  showSection("resultSection");
}

function formatDuration(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}m ${s}s`;
}

// ---------- error / reset ----------

function showError(msg) {
  els.errorMsg.textContent = msg;
  showSection("errorSection");
}

function resetAll() {
  selectedFile = null;
  els.fileInput.value = "";
  els.filebar.hidden = true;
  showSection("uploadSection");
}

els.resetBtn.addEventListener("click", resetAll);
els.errorResetBtn.addEventListener("click", resetAll);
