const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const transcribeBtn = document.getElementById("transcribe-btn");
const uploadStatus = document.getElementById("upload-status");
const uploadProgress = document.getElementById("upload-progress");
const uploadProgressBar = document.getElementById("upload-progress-bar");
const progressPanel = document.getElementById("progress-panel");
const progressText = document.getElementById("progress-text");
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

const STORAGE_KEY = "speech-to-text:file_id";

let fileId = null;
let pollTimer = null;
let turns = [];

function showError(message) {
  errorPanel.textContent = message;
  errorPanel.classList.remove("hidden");
}

function clearError() {
  errorPanel.classList.add("hidden");
  errorPanel.textContent = "";
}

function saveJobState(id) {
  localStorage.setItem(STORAGE_KEY, id);
}

// Восстанавливаем незавершённую (или уже готовую) задачу после перезагрузки вкладки —
// сервер (Postgres) остаётся единственным источником истины о статусе, localStorage хранит
// только "какой file_id проверить", ничего не додумывает сам.
async function resumeJobIfAny() {
  const savedId = localStorage.getItem(STORAGE_KEY);
  if (!savedId) return;

  try {
    const res = await fetch(`/api/status/${savedId}`);
    if (!res.ok) {
      localStorage.removeItem(STORAGE_KEY);
      return;
    }
    const st = await res.json();
    fileId = savedId;
    uploadStatus.textContent = `Восстановлено (id: ${fileId.slice(0, 8)}...)`;
    transcribeBtn.disabled = false;

    if (st.status === "done") {
      await showResult(st.metrics);
    } else if (st.status === "error") {
      showError(st.error || "Неизвестная ошибка обработки");
    } else if (st.status === "queued" || st.status === "processing") {
      transcribeBtn.disabled = true;
      progressPanel.classList.remove("hidden");
      progressText.textContent = STAGE_LABELS[st.stage] || `Статус: ${st.status}...`;
      pollStatus();
    }
  } catch (e) {
    // страница открылась без сети/бекенда — просто не восстанавливаем, не критично
  }
}

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

  // XHR, не fetch — только у XHR есть событие прогресса именно отправки (upload.progress),
  // что и нужно для полосы прогресса при загрузке большого файла.
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
      const data = JSON.parse(xhr.responseText);
      fileId = data.file_id;
      saveJobState(fileId);
      uploadStatus.textContent = `Загружено (id: ${fileId.slice(0, 8)}...)`;
      transcribeBtn.disabled = false;
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

transcribeBtn.addEventListener("click", async () => {
  if (!fileId) return;
  clearError();
  resultPanel.classList.add("hidden");
  transcribeBtn.disabled = true;

  try {
    const res = await fetch(`/api/transcribe/${fileId}`, { method: "POST" });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Transcribe failed: ${res.status}`);
    }
    progressPanel.classList.remove("hidden");
    pollStatus();
  } catch (e) {
    showError(String(e));
    transcribeBtn.disabled = false;
  }
});

function pollStatus() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/status/${fileId}`);
      const st = await res.json();

      if (st.status === "error") {
        progressPanel.classList.add("hidden");
        showError(st.error || "Неизвестная ошибка обработки");
        transcribeBtn.disabled = false;
        return;
      }

      if (st.status === "done") {
        progressPanel.classList.add("hidden");
        transcribeBtn.disabled = false;
        await showResult(st.metrics);
        return;
      }

      progressText.textContent = STAGE_LABELS[st.stage] || `Статус: ${st.status}...`;
      pollStatus();
    } catch (e) {
      showError(String(e));
      transcribeBtn.disabled = false;
    }
  }, 2000);
}

async function showResult(metrics) {
  const res = await fetch(`/api/transcript/${fileId}`);
  if (!res.ok) {
    showError("Не удалось загрузить транскрипт");
    return;
  }
  const data = await res.json();
  turns = data.turns || [];

  player.src = `/api/audio/${fileId}`;
  renderMetrics(metrics);
  renderTranscript();
  resultPanel.classList.remove("hidden");
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

function formatSeconds(sec) {
  if (sec == null) return "-";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return m > 0 ? `${m}м ${s}с` : `${s}с`;
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

resumeJobIfAny();
