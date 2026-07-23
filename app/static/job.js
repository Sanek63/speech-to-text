const player = document.getElementById("player");
const metricsEl = document.getElementById("metrics");
const transcriptEl = document.getElementById("transcript");
const errorPanel = document.getElementById("error-panel");
const jobTitleEl = document.getElementById("job-title");

let turns = [];

function showError(message) {
  errorPanel.textContent = message;
  errorPanel.classList.remove("hidden");
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

async function init() {
  const fileId = new URLSearchParams(location.search).get("file_id");
  if (!fileId) {
    showError("В ссылке не указан file_id");
    return;
  }

  try {
    const [statusRes, transcriptRes] = await Promise.all([
      fetch(`/api/status/${fileId}`),
      fetch(`/api/transcript/${fileId}`),
    ]);
    if (!statusRes.ok || !transcriptRes.ok) {
      showError("Не удалось загрузить задачу — возможно, она ещё не готова или не существует");
      return;
    }
    const status = await statusRes.json();
    const data = await transcriptRes.json();
    turns = data.turns || [];

    const title = status.audio_filename || `Задача ${fileId.slice(0, 8)}...`;
    jobTitleEl.textContent = title;
    document.title = title;

    player.src = `/api/audio/${fileId}`;
    renderMetrics(status.metrics);
    renderTranscript();
  } catch (e) {
    showError(String(e));
  }
}

init();
