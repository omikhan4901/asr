/*
 * Frontend for the Bangla subtitle demo. One card moves through five states: pick -> ready -> working -> done | error.
 * Backend contract (web/backend/app.py on Modal, web/backend/local_server.py locally):
 *   POST /jobs (multipart "file") -> {job_id};  GET /jobs/{id} -> {status, stage, progress, result};  GET /jobs/{id}/download
 */

// The local server (same origin) on this PC; the Modal app on any public host (Vercel, a custom domain)
const MODAL_API = "https://mehboobehsankhan2--bangla-subtitle-generator-web.modal.run";
const IS_LOCAL = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
const API_BASE = window.ASR_API_BASE || (
  window.location.protocol === "file:" ? "http://127.0.0.1:8000" : IS_LOCAL ? window.location.origin : MODAL_API
);
const MAX_MB = 300;
const MAX_MINUTES = 15;
const STEPS = ["upload", "audio", "chunk", "asr", "srt"];

const TEXT = {
  en: {
    skip: "Skip to the upload", brandSub: "Bangla subtitles",
    title: "Bangla subtitles from your video",
    lede: "Upload a video or audio file. Our open Bangla speech recognition model writes an .srt subtitle file with timestamps.",
    dropTitle: "Choose a file or drop it here",
    limits: `MP4, MOV, MKV, MP3 or WAV · up to ${MAX_MINUTES}\u00a0minutes · up to ${MAX_MB}\u00a0MB`,
    change: "Change file", start: "Make subtitles",
    workingTitle: "Making subtitles",
    stepUpload: "Uploading", stepAudio: "Extracting the audio", stepChunk: "Finding the speech",
    stepAsr: "Transcribing", stepSrt: "Writing the .srt file",
    coldStart: "The first file after a quiet period waits about a minute while the server starts.",
    statusChecking: "Checking the server", statusReady: "Server ready",
    statusAsleep: "Server asleep · about 1 min to start", statusOffline: "Server offline",
    doneTitle: "Your subtitles are ready", download: "Download .srt", cuesLabel: "Subtitles", again: "Subtitle another file",
    errorTitle: "That didn't work", retry: "Try again",
    howTitle: "How it works",
    how1: "Voice detection finds the speech and cuts it into pieces of up to 25 seconds.",
    how2: "Each piece is transcribed by our Whisper-medium model, fine-tuned on about 300 hours of Bangla speech.",
    how3: "On whole media recordings our fine-tuned models get about 14% of words wrong, against 26% for the model they started from.",
    how4: "Your video is deleted from the server as soon as the subtitles are made.",
    research: "Research and code", model: "Model (Apache-2.0)", credit: "A CSE499 project at North South University",
    minutes: (m) => `${m} min`,
    summary: (n, s) => `${n} subtitles · ${s} of speech`,
    count: (c, t) => `${c} / ${t}`,
    elapsed: (s) => `${s}`,
    errType: "Please choose a video or audio file.",
    errSize: (mb) => `This file is ${mb} MB. The demo accepts up to ${MAX_MB} MB.`,
    errLong: (m) => `This file is ${m} minutes long. The demo accepts up to ${MAX_MINUTES} minutes.`,
    errNet: "We couldn't reach the server. Check your connection and try again.",
    errFail: (msg) => `The subtitles could not be made: ${msg}`,
    empty: "No speech was found in this file.",
  },
  bn: {
    skip: "আপলোডে যান", brandSub: "বাংলা সাবটাইটেল",
    title: "আপনার ভিডিও থেকে বাংলা সাবটাইটেল",
    lede: "একটি ভিডিও বা অডিও ফাইল আপলোড করুন। আমাদের উন্মুক্ত বাংলা স্পিচ রিকগনিশন মডেল টাইমস্ট্যাম্পসহ একটি .srt সাবটাইটেল ফাইল তৈরি করে দেবে।",
    dropTitle: "ফাইল বেছে নিন বা এখানে ছেড়ে দিন",
    limits: `MP4, MOV, MKV, MP3 বা WAV · সর্বোচ্চ\u00a0${bn(MAX_MINUTES)}\u00a0মিনিট · সর্বোচ্চ\u00a0${bn(MAX_MB)}\u00a0MB`,
    change: "ফাইল বদলান", start: "সাবটাইটেল তৈরি করুন",
    workingTitle: "সাবটাইটেল তৈরি হচ্ছে",
    stepUpload: "আপলোড হচ্ছে", stepAudio: "অডিও আলাদা করা হচ্ছে", stepChunk: "কথা খোঁজা হচ্ছে",
    stepAsr: "লেখা হচ্ছে", stepSrt: ".srt ফাইল তৈরি হচ্ছে",
    coldStart: "কিছুক্ষণ ব্যবহার না হলে প্রথম ফাইলে সার্ভার চালু হতে প্রায় এক মিনিট লাগে।",
    statusChecking: "সার্ভার দেখা হচ্ছে", statusReady: "সার্ভার প্রস্তুত",
    statusAsleep: "সার্ভার ঘুমিয়ে · চালু হতে প্রায় ১ মিনিট", statusOffline: "সার্ভার বন্ধ",
    doneTitle: "আপনার সাবটাইটেল তৈরি", download: ".srt ডাউনলোড করুন", cuesLabel: "সাবটাইটেল", again: "আরেকটি ফাইলের সাবটাইটেল",
    errorTitle: "কাজটি হয়নি", retry: "আবার চেষ্টা করুন",
    howTitle: "কীভাবে কাজ করে",
    how1: "ভয়েস ডিটেকশন কথা খুঁজে সর্বোচ্চ ২৫ সেকেন্ডের অংশে ভাগ করে।",
    how2: "প্রতিটি অংশ লেখে আমাদের Whisper-medium মডেল, যা প্রায় ৩০০ ঘণ্টা বাংলা কথায় প্রশিক্ষিত।",
    how3: "পূর্ণ মিডিয়া রেকর্ডিংয়ে আমাদের প্রশিক্ষিত মডেলগুলোর শব্দ ভুলের হার প্রায় ১৪%, যে মডেল থেকে শুরু হয়েছিল তার ২৬%।",
    how4: "সাবটাইটেল তৈরি হলেই আপনার ভিডিও সার্ভার থেকে মুছে ফেলা হয়।",
    research: "গবেষণা ও কোড", model: "মডেল (Apache-2.0)", credit: "নর্থ সাউথ ইউনিভার্সিটির একটি CSE499 প্রকল্প",
    minutes: (m) => `${bn(m)} মিনিট`,
    summary: (n, s) => `${bn(n)}টি সাবটাইটেল · ${s} কথা`,
    count: (c, t) => `${bn(c)} / ${bn(t)}`,
    elapsed: (s) => bn(s),
    errType: "একটি ভিডিও বা অডিও ফাইল বেছে নিন।",
    errSize: (mb) => `ফাইলটি ${bn(mb)} MB। এই ডেমোতে সর্বোচ্চ ${bn(MAX_MB)} MB চলে।`,
    errLong: (m) => `ফাইলটি ${bn(m)} মিনিট লম্বা। এই ডেমোতে সর্বোচ্চ ${bn(MAX_MINUTES)} মিনিট চলে।`,
    errNet: "সার্ভারে পৌঁছানো যায়নি। ইন্টারনেট সংযোগ দেখে আবার চেষ্টা করুন।",
    errFail: (msg) => `সাবটাইটেল তৈরি করা যায়নি: ${msg}`,
    empty: "এই ফাইলে কোনো কথা পাওয়া যায়নি।",
  },
};

function bn(x) { return String(x).replace(/\d/g, (d) => "০১২৩৪৫৬৭৮৯"[d]); }

const $ = (id) => document.getElementById(id);
const card = $("app");
let lang = "en";
let file = null, jobId = null, poll = null, clock = null, startedAt = 0;
let lastError = null, lastResult = null;
let serverStatus = "checking";  // checking | ready | asleep | offline  // kept so a language switch can re-render them

function t(key, ...args) { const v = TEXT[lang][key]; return typeof v === "function" ? v(...args) : v; }

function setLang(next) {
  lang = next;
  document.documentElement.lang = next;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === next)));
  try { localStorage.setItem("lang", next); } catch (e) { /* private mode */ }
  $("cues").setAttribute("aria-label", t("cuesLabel"));
  showStatus();
  if (file) showFile();
  if (lastResult) showSummary();
  if (lastError) $("errorMsg").textContent = t(lastError.key, ...lastError.args);
}

function show(state) {
  card.querySelectorAll(".state").forEach((s) => { s.hidden = s.dataset.state !== state; });
}

function fmtDuration(seconds) {
  const m = Math.floor(seconds / 60), s = Math.round(seconds % 60);
  const txt = m ? `${m}:${String(s).padStart(2, "0")} min` : `${s} s`;
  return lang === "bn" ? bn(txt).replace("min", "মিনিট").replace(" s", " সেকেন্ড") : txt;
}

function fail(key, ...args) {
  stopTimers();
  lastError = { key, args };
  $("errorMsg").textContent = t(key, ...args);
  show("error");
}

function stopTimers() { clearInterval(poll); clearInterval(clock); poll = clock = null; }

/* server status: is the API reachable, and is a GPU already up (a file starts at once) or asleep (about a minute) */
function showStatus() {
  const key = { checking: "statusChecking", ready: "statusReady", asleep: "statusAsleep", offline: "statusOffline" }[serverStatus];
  $("status").dataset.status = serverStatus;
  $("statusText").textContent = t(key);
  $("coldNote").hidden = serverStatus === "ready";
}

async function checkServer() {
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store", signal: AbortSignal.timeout(25000) });
    const body = res.ok ? await res.json() : {};
    serverStatus = body.status === "ready" ? "ready" : res.ok ? "asleep" : "offline";
  } catch (e) {
    serverStatus = "offline";
  }
  showStatus();
}

/* choosing a file: type, size and length are checked before anything is uploaded */
function mediaMinutes(f) {
  return new Promise((resolve) => {
    const el = document.createElement(f.type.startsWith("audio") ? "audio" : "video");
    const url = URL.createObjectURL(f);
    const done = (v) => { URL.revokeObjectURL(url); resolve(v); };
    el.preload = "metadata";
    el.onloadedmetadata = () => done(isFinite(el.duration) ? el.duration / 60 : null);
    el.onerror = () => done(null);  // the browser cannot read it (e.g. some MKV): the server still checks the length
    el.src = url;
    setTimeout(() => done(null), 4000);
  });
}

async function pick(f) {
  if (!f) return;
  const looksMedia = /^(video|audio)\//.test(f.type) || /\.(mkv|mp3|wav|m4a|mp4|mov|webm|ogg|flac)$/i.test(f.name);
  if (!looksMedia) return fail("errType");
  const mb = f.size / (1024 * 1024);
  if (mb > MAX_MB) return fail("errSize", Math.ceil(mb));
  const minutes = await mediaMinutes(f);
  if (minutes && minutes > MAX_MINUTES) return fail("errLong", Math.ceil(minutes));
  file = f;
  file.minutes = minutes;
  showFile();
  show("ready");
  $("startBtn").focus();
}

function showFile() {
  $("fileName").textContent = file.name;
  const size = file.size < 1024 * 1024 ? `${(file.size / 1024).toFixed(0)} KB` : `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
  const parts = [lang === "bn" ? bn(size) : size];
  if (file.minutes) parts.push(fmtDuration(file.minutes * 60));
  $("fileMeta").textContent = parts.join(" · ");
}

/* progress */
function setStep(stage, progress) {
  if (!STEPS.includes(stage)) stage = "upload";  // "queued": waiting for the GPU
  const i = STEPS.indexOf(stage);
  document.querySelectorAll("#steps li").forEach((li) => {
    const j = STEPS.indexOf(li.dataset.step);
    li.classList.toggle("done", j < i);
    li.classList.toggle("active", j === i);
  });
  const share = progress && progress.total ? progress.current / progress.total : 0;
  // overall bar: upload and audio are quick, transcription is most of the work
  const weights = { upload: [0, 10], audio: [10, 15], chunk: [15, 20], asr: [20, 95], srt: [95, 100] };
  const [a, b] = weights[stage] || [0, 0];
  const pct = Math.round(a + (b - a) * share);
  $("barFill").style.width = `${pct}%`;
  $("bar").setAttribute("aria-valuenow", String(pct));
  $("asrCount").textContent = stage === "asr" && progress && progress.total ? t("count", progress.current, progress.total) : "";
}

function tickClock() {
  const s = Math.round((Date.now() - startedAt) / 1000);
  const txt = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  $("elapsed").textContent = t("elapsed", txt);
}

async function start() {
  if (!file) return;
  show("working");
  setStep("upload");
  startedAt = Date.now();
  tickClock();
  clock = setInterval(tickClock, 1000);

  const body = new FormData();
  body.append("file", file);
  let res;
  try {
    res = await fetch(`${API_BASE}/jobs`, { method: "POST", body });
  } catch (e) {
    return fail("errNet");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    return fail("errFail", err.detail || `HTTP ${res.status}`);
  }
  jobId = (await res.json()).job_id;
  poll = setInterval(check, 1500);
}

async function check() {
  let job;
  try {
    const res = await fetch(`${API_BASE}/jobs/${jobId}`);
    if (!res.ok) return;
    job = await res.json();
  } catch (e) {
    return;  // a missed poll is retried on the next tick
  }
  if (job.status === "pending" || job.status === "processing") return setStep(job.stage || "upload", job.progress);
  if (job.status === "failed") return fail("errFail", job.error || "unknown error");
  if (job.status === "completed") done(job.result || {});
}

function done(result) {
  stopTimers();
  setStep("srt", { current: 1, total: 1 });
  const n = result.num_chunks || 0;
  if (!n) return fail("empty");
  lastResult = result;
  showSummary();
  showCues(result.cues || []);
  const link = $("downloadBtn");
  link.href = `${API_BASE}/jobs/${jobId}/download`;
  link.setAttribute("download", `${file.name.replace(/\.[^/.]+$/, "")}.srt`);
  show("done");
  link.focus();
}

function hms(seconds) {
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = Math.floor(seconds % 60);
  return (h ? `${h}:${String(m).padStart(2, "0")}` : `${m}`) + `:${String(s).padStart(2, "0")}`;
}

function showCues(cues) {
  const list = $("cues");
  list.replaceChildren(...cues.map((c) => {
    const li = document.createElement("li");
    const time = document.createElement("span");
    time.className = "cue-time";
    time.textContent = hms(c.start);
    const text = document.createElement("span");
    text.className = "cue-text";
    text.textContent = c.text;
    li.append(time, text);
    return li;
  }));
  list.scrollTop = 0;
}

function showSummary() {
  $("summary").textContent = t("summary", lastResult.num_chunks, fmtDuration(lastResult.total_speech_s || 0));
}

function reset() {
  stopTimers();
  file = jobId = lastResult = lastError = null;
  $("fileInput").value = "";
  show("pick");
}

/* events */
const drop = $("drop");
$("fileInput").addEventListener("change", (e) => pick(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); pick(e.dataTransfer.files[0]); });
$("changeBtn").addEventListener("click", () => { reset(); $("fileInput").click(); });
$("startBtn").addEventListener("click", start);
$("againBtn").addEventListener("click", reset);
$("retryBtn").addEventListener("click", () => (file && lastError && lastError.key !== "errType" ? start() : reset()));
document.querySelectorAll(".lang button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));

checkServer();
setInterval(() => { if (!document.hidden) checkServer(); }, 30000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) checkServer(); });

let saved = null;
try { saved = localStorage.getItem("lang"); } catch (e) { /* private mode */ }
setLang(saved || ((navigator.language || "").startsWith("bn") ? "bn" : "en"));
