const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");
const uploadProgress = document.getElementById("upload-progress");
const uploadProgressBar = document.getElementById("upload-progress-bar");
const dropOverlay = document.getElementById("drop-overlay");
const jobsListEl = document.getElementById("jobs-list");
const jobsPrevBtn = document.getElementById("jobs-prev");
const jobsNextBtn = document.getElementById("jobs-next");
const jobsPageInfo = document.getElementById("jobs-page-info");
const resultPanel = document.getElementById("result-panel");
const player = document.getElementById("player");
const metricsEl = document.getElementById("metrics");
const transcriptEl = document.getElementById("transcript");
const errorPanel = document.getElementById("error-panel");

const STAGE_LABELS = {
  preprocess: "Препроцессинг аудио...",
  asr: "Распознавание речи (ASR)...",
  diarization: "Диаризация (кто говорит)...",
  postprocess: "Роли и экспорт...",
};

const STATUS_LABELS = {
  uploaded: "Загружено",
  queued: "В очереди",
  done: "Готово",
  error: "Ошибка",
};

const JOBS_PAGE_SIZE = 10;

let jobsPage = 0;
let jobsTotal = 0;
let jobsPollTimer = null;
let selectedFileId = null;
let turns = [];

function showError(message) {
  errorPanel.textContent = message;
  errorPanel.classList.remove("hidden");
}

function clearError() {
  errorPanel.classList.add("hidden");
  errorPanel.textContent = "";
}

function jobStatusLabel(job) {
  if (job.status === "processing") {
    const label = STAGE_LABELS[job.stage] || "Обработка...";
    if (job.stage === "preprocess" && job.progress != null) {
      return `${label} ${Math.round(job.progress)}%`;
    }
    return label;
  }
  return STATUS_LABELS[job.status] || job.status;
}

function formatTimestamp(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("ru-RU", {
    day: "2-digit", month: "2-digit", year: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function formatSeconds(sec) {
  if (sec == null) return "-";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}м ${s}с` : `${s}с`;
}

// --- Загрузка файла (XHR, не fetch — только у XHR есть событие прогресса отправки) --------
// Очередь, а не параллельные запросы: и выбор в инпуте, и drag-and-drop могут дать сразу
// несколько файлов, а UI-элемент прогресса один общий — грузим по одному, по порядку.

let uploadQueue = [];
let uploading = false;

function enqueueUploads(files) {
  // f.type пустой, если браузер не смог определить MIME по расширению — не блокируем в
  // этом случае сами, пусть решает бекенд/ffmpeg; отсеиваем только явно НЕ аудио/видео
  const audioVideo = files.filter((f) => !f.type || /^(audio|video)\//.test(f.type));
  if (audioVideo.length === 0 && files.length > 0) {
    showError("Файл не похож на аудио/видео");
    return;
  }
  uploadQueue.push(...audioVideo);
  processUploadQueue();
}

async function processUploadQueue() {
  if (uploading || uploadQueue.length === 0) return;
  uploading = true;
  uploadBtn.disabled = true;
  while (uploadQueue.length > 0) {
    const file = uploadQueue.shift();
    await uploadFile(file, uploadQueue.length);
  }
  uploading = false;
  uploadBtn.disabled = false;
}

function uploadFile(file, remainingAfter) {
  return new Promise((resolve) => {
    // clearError() здесь не вызываем: при батче из нескольких файлов это стирало бы
    // ошибку предыдущего файла в тот момент, когда стартует следующий — очищаем один раз
    // за весь батч, в обработчике клика/drop, до enqueueUploads().
    const suffix = remainingAfter > 0 ? ` (ещё ${remainingAfter} в очереди)` : "";
    uploadStatus.textContent = `Загрузка «${file.name}»... 0%${suffix}`;
    uploadProgress.classList.remove("hidden");
    uploadProgressBar.style.width = "0%";

    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");

    xhr.upload.addEventListener("progress", (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      uploadProgressBar.style.width = `${pct}%`;
      uploadStatus.textContent = `Загрузка «${file.name}»... ${pct}%${suffix}`;
    });

    xhr.addEventListener("load", () => {
      uploadProgress.classList.add("hidden");
      if (xhr.status >= 200 && xhr.status < 300) {
        uploadStatus.textContent = "Загружено";
        jobsPage = 0; // новая задача — самая свежая, всегда на первой странице
        loadJobsList();
      } else {
        let message = `Upload failed: ${xhr.status}`;
        try {
          const detail = JSON.parse(xhr.responseText);
          if (detail.detail) message = detail.detail;
        } catch (_) {
          // тело ответа не JSON — оставляем дефолтное сообщение
        }
        showError(`«${file.name}»: ${message}`);
      }
      resolve();
    });

    xhr.addEventListener("error", () => {
      uploadProgress.classList.add("hidden");
      showError(`«${file.name}»: ошибка сети при загрузке файла`);
      resolve();
    });

    xhr.send(form);
  });
}

uploadBtn.addEventListener("click", () => {
  clearError();
  const files = Array.from(fileInput.files);
  if (files.length === 0) {
    uploadStatus.textContent = "Выберите файл";
    return;
  }
  fileInput.value = ""; // сразу можно выбрать следующий файл
  enqueueUploads(files);
});

// --- Drag-and-drop файла в любое место страницы --------------------------------------------

let dragCounter = 0;

window.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer || !e.dataTransfer.types.includes("Files")) return;
  e.preventDefault();
  dragCounter++;
  dropOverlay.classList.remove("hidden");
});

window.addEventListener("dragover", (e) => {
  // без preventDefault здесь событие drop вообще не сработает — так задумано в DnD API
  if (e.dataTransfer && e.dataTransfer.types.includes("Files")) e.preventDefault();
});

window.addEventListener("dragleave", (e) => {
  if (!e.dataTransfer || !e.dataTransfer.types.includes("Files")) return;
  e.preventDefault();
  dragCounter = Math.max(0, dragCounter - 1);
  if (dragCounter === 0) dropOverlay.classList.add("hidden");
});

window.addEventListener("drop", (e) => {
  if (!e.dataTransfer || !e.dataTransfer.types.includes("Files")) return;
  e.preventDefault();
  dragCounter = 0;
  dropOverlay.classList.add("hidden");
  clearError();
  enqueueUploads(Array.from(e.dataTransfer.files));
});

// --- Список задач (пагинация + автообновление статусов) -----------------------------------

async function loadJobsList() {
  const offset = jobsPage * JOBS_PAGE_SIZE;
  try {
    const res = await fetch(`/api/jobs?limit=${JOBS_PAGE_SIZE}&offset=${offset}`);
    if (!res.ok) return;
    const data = await res.json();
    jobsTotal = data.total;
    renderJobsList(data.jobs);
    updatePaginationControls();
  } catch (e) {
    // сеть/бекенд временно недоступны — молча пробуем на следующем тике опроса
  }
}

function renderJobsList(jobs) {
  jobsListEl.innerHTML = "";
  if (jobs.length === 0) {
    jobsListEl.innerHTML = `<div class="jobs-empty">Пока нет загруженных файлов</div>`;
    return;
  }
  for (const job of jobs) {
    jobsListEl.appendChild(renderJobRow(job));
  }
  tickElapsedTimers();
}

// Для asr/diarization точный процент честно недоступен без хрупких хаков во внутренности
// Whisper/NeMo — вместо фейковых цифр просто тикаем "сколько времени прошло" от
// updated_at (момент начала этапа), между 3-секундными опросами списка, чтобы было видно,
// что ничего не зависло, без лишних запросов к бекенду.
function tickElapsedTimers() {
  document.querySelectorAll(".job-row[data-stage-started-at]").forEach((row) => {
    const started = new Date(row.dataset.stageStartedAt).getTime();
    if (Number.isNaN(started)) return;
    const elapsedSec = Math.max(0, (Date.now() - started) / 1000);
    const el = row.querySelector(".job-elapsed");
    if (el) el.textContent = formatSeconds(elapsedSec);
  });
}

setInterval(tickElapsedTimers, 1000);

function renderJobRow(job) {
  const row = document.createElement("div");
  row.className = "job-row";
  row.dataset.fileId = job.file_id;

  const main = document.createElement("div");
  main.className = "job-main";

  const name = document.createElement("span");
  name.className = "job-filename";
  name.textContent = job.audio_filename || `${job.file_id.slice(0, 8)}...`;
  main.appendChild(name);

  const badge = document.createElement("span");
  badge.className = `badge job-status-badge status-${job.status}`;
  if (job.status === "processing") {
    const spinner = document.createElement("span");
    spinner.className = "spinner-sm";
    badge.appendChild(spinner);
  }
  badge.appendChild(document.createTextNode(jobStatusLabel(job)));

  const isTimedStage = job.status === "processing" && (job.stage === "asr" || job.stage === "diarization");
  if (isTimedStage) {
    badge.appendChild(document.createTextNode(" "));
    const elapsed = document.createElement("span");
    elapsed.className = "job-elapsed";
    badge.appendChild(elapsed);
    row.dataset.stageStartedAt = job.updated_at;
  }
  main.appendChild(badge);

  if (job.status === "done" && job.metrics) {
    const summary = document.createElement("span");
    summary.className = "status-text";
    summary.textContent = `${formatSeconds(job.metrics.audio_duration_sec)} · RTF ${job.metrics.total_rtf.toFixed(3)}`;
    main.appendChild(summary);
  }

  row.appendChild(main);

  if (job.status === "processing" && job.stage === "preprocess" && job.progress != null) {
    const track = document.createElement("div");
    track.className = "job-progress";
    const bar = document.createElement("div");
    bar.className = "job-progress-bar";
    bar.style.width = `${job.progress}%`;
    track.appendChild(bar);
    row.appendChild(track);
  } else if (isTimedStage) {
    const track = document.createElement("div");
    track.className = "job-progress indeterminate";
    const bar = document.createElement("div");
    bar.className = "job-progress-bar";
    track.appendChild(bar);
    row.appendChild(track);
  }

  const meta = document.createElement("div");
  meta.className = "job-meta";

  const time = document.createElement("span");
  time.className = "status-text";
  time.textContent = formatTimestamp(job.created_at);
  meta.appendChild(time);

  if (job.status === "uploaded" || job.status === "error") {
    const btn = document.createElement("button");
    btn.textContent = job.status === "error" ? "Повторить" : "Транскрибировать";
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      startTranscribe(job.file_id, btn);
    });
    meta.appendChild(btn);
  }

  row.appendChild(meta);

  if (job.status === "error" && job.error) {
    const errText = document.createElement("div");
    errText.className = "job-error-text";
    errText.textContent = job.error.split("\n")[0];
    errText.title = job.error;
    row.appendChild(errText);
  }

  if (job.status === "done") {
    row.classList.add("clickable");
    row.addEventListener("click", () => openJobDetail(job));
    if (job.file_id === selectedFileId) row.classList.add("active");
  }

  return row;
}

async function startTranscribe(fileId, btnEl) {
  if (btnEl) btnEl.disabled = true;
  clearError();
  try {
    const res = await fetch(`/api/transcribe/${fileId}`, { method: "POST" });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Transcribe failed: ${res.status}`);
    }
    await loadJobsList();
  } catch (e) {
    showError(String(e));
    if (btnEl) btnEl.disabled = false;
  }
}

function updatePaginationControls() {
  const totalPages = Math.max(1, Math.ceil(jobsTotal / JOBS_PAGE_SIZE));
  jobsPageInfo.textContent = `Страница ${jobsPage + 1} из ${totalPages} (всего ${jobsTotal})`;
  jobsPrevBtn.disabled = jobsPage === 0;
  jobsNextBtn.disabled = jobsPage + 1 >= totalPages;
}

jobsPrevBtn.addEventListener("click", () => {
  if (jobsPage === 0) return;
  jobsPage--;
  loadJobsList();
});

jobsNextBtn.addEventListener("click", () => {
  jobsPage++;
  loadJobsList();
});

function startJobsPolling() {
  if (jobsPollTimer) clearInterval(jobsPollTimer);
  jobsPollTimer = setInterval(loadJobsList, 3000);
}

// --- Детали готовой задачи: плеер + метрики + транскрипт --------------------------------

async function openJobDetail(job) {
  clearError();
  try {
    const res = await fetch(`/api/transcript/${job.file_id}`);
    if (!res.ok) {
      showError("Не удалось загрузить транскрипт");
      return;
    }
    const data = await res.json();
    turns = data.turns || [];
    selectedFileId = job.file_id;

    player.src = `/api/audio/${job.file_id}`;
    renderMetrics(job.metrics);
    renderTranscript();
    resultPanel.classList.remove("hidden");
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    showError(String(e));
  }
}

function renderMetrics(metrics) {
  metricsEl.innerHTML = "";
  if (!metrics) return;
  const tiles = [
    ["Препроцессинг", metrics.preprocess_time_sec, metrics.preprocess_rtf],
    ["ASR", metrics.asr_time_sec, metrics.asr_rtf],
    ["Диаризация", metrics.diarization_time_sec, metrics.diarization_rtf],
    ["Постобработка", metrics.postprocess_time_sec, metrics.postprocess_rtf],
    ["Итого", metrics.total_time_sec, metrics.total_rtf],
  ];
  for (const [label, sec, rtf] of tiles) {
    const tile = document.createElement("div");
    tile.className = "metric-tile";
    tile.innerHTML = `<div class="value">${formatSeconds(sec)}</div><div class="label">${label} · RTF ${rtf.toFixed(3)}</div>`;
    metricsEl.appendChild(tile);
  }
}

function roleClass(role) {
  if (role.startsWith("Преподаватель")) return "role-teacher";
  if (role.startsWith("Студент")) return "role-student";
  return "role-other";
}

function renderTranscript() {
  transcriptEl.innerHTML = "";
  for (const turn of turns) {
    const div = document.createElement("div");
    div.className = "turn";
    div.dataset.turnId = turn.id;
    div.dataset.start = turn.start;
    div.dataset.end = turn.end;

    const badge = document.createElement("span");
    badge.className = `badge ${roleClass(turn.role)}`;
    badge.textContent = turn.role;

    const text = document.createElement("span");
    text.className = "text";
    turn.words.forEach((w, i) => {
      const wordSpan = document.createElement("span");
      wordSpan.className = "word";
      wordSpan.dataset.start = w.start;
      wordSpan.dataset.end = w.end;
      wordSpan.textContent = w.word + (i < turn.words.length - 1 ? " " : "");
      text.appendChild(wordSpan);
    });

    div.appendChild(badge);
    div.appendChild(text);
    div.addEventListener("click", () => {
      player.currentTime = turn.start;
      player.play();
    });
    transcriptEl.appendChild(div);
  }
}

player.addEventListener("timeupdate", () => {
  const t = player.currentTime;
  const activeTurn = turns.find((turn) => t >= turn.start && t < turn.end);

  document.querySelectorAll(".turn.active").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".word.current").forEach((el) => el.classList.remove("current"));

  if (!activeTurn) return;

  const turnEl = transcriptEl.querySelector(`.turn[data-turn-id="${activeTurn.id}"]`);
  if (turnEl) {
    turnEl.classList.add("active");
    turnEl.scrollIntoView({ block: "nearest", behavior: "smooth" });

    const activeWord = activeTurn.words.find((w) => t >= w.start && t < w.end);
    if (activeWord) {
      const wordEl = turnEl.querySelector(`.word[data-start="${activeWord.start}"]`);
      if (wordEl) wordEl.classList.add("current");
    }
  }
});

loadJobsList();
startJobsPolling();
