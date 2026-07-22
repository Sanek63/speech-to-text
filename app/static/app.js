const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");
const uploadProgress = document.getElementById("upload-progress");
const uploadProgressBar = document.getElementById("upload-progress-bar");
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
  if (job.status === "processing") return STAGE_LABELS[job.stage] || "Обработка...";
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

uploadBtn.addEventListener("click", () => {
  clearError();
  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Выберите файл";
    return;
  }
  uploadStatus.textContent = "Загрузка... 0%";
  uploadBtn.disabled = true;
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
    uploadStatus.textContent = `Загрузка... ${pct}%`;
  });

  xhr.addEventListener("load", () => {
    uploadBtn.disabled = false;
    uploadProgress.classList.add("hidden");
    if (xhr.status >= 200 && xhr.status < 300) {
      uploadStatus.textContent = "Загружено";
      fileInput.value = ""; // сразу можно выбрать следующий файл
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
      showError(message);
    }
  });

  xhr.addEventListener("error", () => {
    uploadBtn.disabled = false;
    uploadProgress.classList.add("hidden");
    showError("Ошибка сети при загрузке файла");
  });

  xhr.send(form);
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
}

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
  main.appendChild(badge);

  if (job.status === "done" && job.metrics) {
    const summary = document.createElement("span");
    summary.className = "status-text";
    summary.textContent = `${formatSeconds(job.metrics.audio_duration_sec)} · RTF ${job.metrics.total_rtf.toFixed(3)}`;
    main.appendChild(summary);
  }

  row.appendChild(main);

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
