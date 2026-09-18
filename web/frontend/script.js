/*
 * Frontend Controller for Bangla ASR Subtitle Generator.
 */

// Dynamically connect to local FastAPI backend or custom remote host
const API_BASE = window.ASR_API_BASE || (
  ["http:", "https:"].includes(window.location.protocol)
    ? window.location.origin
    : "http://127.0.0.1:8000"
);

const STEP_ORDER = ["upload", "audio", "chunk", "asr", "rescore", "srt"];

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const filebar = document.getElementById("filebar");
const fileName = document.getElementById("fileName");
const startBtn = document.getElementById("startBtn");

const uploadSection = document.getElementById("uploadSection");
const processingSection = document.getElementById("processingSection");
const procCaptionText = document.getElementById("procCaptionText");
const procNote = document.getElementById("procNote");
const stepItems = document.querySelectorAll("#steps li");

const resultSection = document.getElementById("resultSection");
const metaSegments = document.getElementById("metaSegments");
const metaDuration = document.getElementById("metaDuration");
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");
const srtPreview = document.getElementById("srtPreview");

const errorSection = document.getElementById("errorSection");
const errorMsg = document.getElementById("errorMsg");
const errorResetBtn = document.getElementById("errorResetBtn");

let selectedFile = null;
let currentJobId = null;
let pollTimer = null;
let stageStartTimes = {};

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = `${file.name} (${formatBytes(file.size)})`;
  filebar.hidden = false;
  startBtn.disabled = false;
  dropzone.style.borderColor = "var(--accent)";
}

function resetToUpload() {
  if (pollTimer) clearInterval(pollTimer);
  selectedFile = null;
  currentJobId = null;
  stageStartTimes = {};
  fileInput.value = "";

  dropzone.style.borderColor = "";
  filebar.hidden = true;
  uploadSection.hidden = false;
  processingSection.hidden = true;
  resultSection.hidden = true;
  errorSection.hidden = true;

  // Reset steps
  stepItems.forEach((li) => {
    li.classList.remove("active", "done");
    const timeSpan = li.querySelector(".step-time");
    if (timeSpan) timeSpan.textContent = "";
  });
}

function updateStepStatus(activeStep, isComplete = false) {
  let pastActive = false;

  STEP_ORDER.forEach((stepId) => {
    const li = document.querySelector(`#steps li[data-step="${stepId}"]`);
    if (!li) return;

    if (isComplete) {
      li.classList.remove("active");
      li.classList.add("done");
    } else if (stepId === activeStep) {
      li.classList.add("active");
      li.classList.remove("done");
      pastActive = true;
    } else if (!pastActive) {
      li.classList.remove("active");
      li.classList.add("done");
    } else {
      li.classList.remove("active", "done");
    }
  });

  const banglaLabels = {
    upload: "আপলোড হচ্ছে...",
    audio: "অডিও আলাদা করা হচ্ছে...",
    chunk: "স্পিচ সেগমেন্ট করা হচ্ছে...",
    asr: "ট্রান্সক্রিপশন চলছে...",
    rescore: "MBR রিস্কোরিং চলছে...",
    srt: "সাবটাইটেল তৈরি হচ্ছে...",
  };

  if (procCaptionText) {
    procCaptionText.textContent = isComplete
      ? "সম্পূর্ণ হয়েছে!"
      : (banglaLabels[activeStep] || "কাজ চলছে...");
  }
}

function showError(msg) {
  if (pollTimer) clearInterval(pollTimer);
  uploadSection.hidden = true;
  processingSection.hidden = true;
  resultSection.hidden = true;
  errorSection.hidden = false;
  errorMsg.textContent = msg || "An unexpected error occurred.";
}

async function uploadAndProcess() {
  if (!selectedFile) return;

  uploadSection.hidden = true;
  processingSection.hidden = false;
  resultSection.hidden = true;
  errorSection.hidden = true;

  updateStepStatus("upload");
  stageStartTimes["upload"] = Date.now();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/jobs`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned status ${res.status}`);
    }

    const data = await res.json();
    currentJobId = data.job_id;
    startPolling(currentJobId);
  } catch (err) {
    showError(`Could not start job: ${err.message}. Is the server running on ${API_BASE}?`);
  }
}

function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/jobs/${jobId}`);
      if (!res.ok) return;

      const job = await res.json();

      if (job.status === "processing" || job.status === "pending") {
        const stage = job.stage || "upload";
        updateStepStatus(stage);

        const prog = job.progress;
        if (prog && prog.total > 0 && procNote) {
          procNote.textContent = `${prog.stage.toUpperCase()}: ${prog.current} of ${prog.total} items processed (${prog.percent}%).`;
        }
      } else if (job.status === "completed") {
        clearInterval(pollTimer);
        updateStepStatus("srt", true);
        showResults(job.result);
      } else if (job.status === "failed") {
        clearInterval(pollTimer);
        showError(job.error || "Subtitle generation failed on server.");
      }
    } catch (e) {
      console.warn("Polling error:", e);
    }
  }, 1500);
}

function showResults(result) {
  processingSection.hidden = true;
  resultSection.hidden = false;

  metaSegments.textContent = result.num_chunks ?? "-";
  metaDuration.textContent = result.total_speech_s ? `${result.total_speech_s}s` : "-";
  srtPreview.textContent = result.preview || "(Empty subtitle output)";

  downloadBtn.href = `${API_BASE}/jobs/${currentJobId}/download`;
  downloadBtn.setAttribute("download", `${selectedFile.name.replace(/\.[^/.]+$/, "")}.srt`);
}

// Event Listeners
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) {
    handleFile(e.target.files[0]);
  }
});

startBtn.addEventListener("click", uploadAndProcess);
resetBtn.addEventListener("click", resetToUpload);
errorResetBtn.addEventListener("click", resetToUpload);
