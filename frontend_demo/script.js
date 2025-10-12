const PIPE_INDEX_PATH = "../mock_data/pipes/index.json";
const PIPE_PATH = (id) => `../mock_data/pipes/${id}.json`;
const ARTIFACT_INDEX_PATH = "../mock_data/artifacts/index.json";
const ARTIFACT_DRAFT_PATH = (pipeId) => `../mock_data/artifacts/${pipeId}/sample_draft.md`;
const ARTIFACT_REPORT_PATH = (pipeId) => `../mock_data/artifacts/${pipeId}/sample_report.json`;
const ARTIFACT_JSONLD_PATH = (pipeId) => `../mock_data/artifacts/${pipeId}/sample_jsonld.json`;

const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");
const pipeSelect = document.getElementById("pipe-select");
const pipesList = document.getElementById("pipes-list");
const artifactsList = document.getElementById("artifacts-list");
const briefForm = document.getElementById("brief-form");
const progressOverlay = document.getElementById("progress-overlay");
const draftView = document.getElementById("draft-view");
const reportView = document.getElementById("report-view");
const resultTitle = document.getElementById("result-title");
const resultMeta = document.getElementById("result-meta");
const downloadMdBtn = document.getElementById("download-md");
const downloadReportBtn = document.getElementById("download-report");

const state = {
  pipes: new Map(),
  artifacts: [],
  currentResult: null,
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

function switchTab(tabId) {
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
  panels.forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
}

async function loadPipes() {
  const response = await fetch(PIPE_INDEX_PATH);
  const ids = await response.json();
  const pipeEntries = await Promise.all(
    ids.map(async (id) => {
      const data = await fetch(PIPE_PATH(id)).then((res) => res.json());
      return [id, data];
    })
  );
  pipeEntries.forEach(([id, data], idx) => {
    state.pipes.set(id, data);
    const option = document.createElement("option");
    option.value = id;
    option.textContent = data.name;
    if (idx === 0) {
      option.selected = true;
    }
    pipeSelect.append(option);
  });
  renderPipeCards(pipeEntries.map(([, data]) => data));
}

function renderPipeCards(pipes) {
  if (!pipes.length) {
    pipesList.innerHTML = '<div class="empty-state">Тематики пока не добавлены.</div>';
    return;
  }

  pipesList.innerHTML = "";
  pipes.forEach((pipe) => {
    const card = document.createElement("article");
    card.className = "pipe-card";
    card.innerHTML = `
      <div>
        <h3>${pipe.name}</h3>
        <p>${pipe.description}</p>
      </div>
      <div class="pipe-meta">Тон: <strong>${pipe.tone}</strong></div>
      <div class="pipe-keywords">
        ${pipe.keywords.map((kw) => `<span>${kw}</span>`).join("")}
      </div>
    `;
    pipesList.append(card);
  });
}

async function loadArtifacts() {
  const response = await fetch(ARTIFACT_INDEX_PATH);
  state.artifacts = await response.json();
  renderArtifacts();
}

function renderArtifacts() {
  if (!state.artifacts.length) {
    artifactsList.innerHTML = '<div class="empty-state">Пока нет сгенерированных материалов.</div>';
    return;
  }

  const template = document.getElementById("artifact-card-template");
  artifactsList.innerHTML = "";

  state.artifacts.forEach((artifact) => {
    const pipe = state.pipes.get(artifact.pipe_id);
    const card = template.content.firstElementChild.cloneNode(true);
    card.querySelector(".card-title").textContent = artifact.name;
    card.querySelector(".status").textContent = artifact.status;
    card.querySelector(".status").dataset.status = artifact.status;
    card.querySelector(".card-topic").textContent = `${pipe?.name ?? artifact.pipe_id} · ${artifact.topic}`;
    card.querySelector(".card-meta").textContent = `Обновлено ${artifact.updated_at}`;
    card.addEventListener("click", () => {
      showResult({
        source: "artifact",
        artifactId: artifact.id,
        pipeId: artifact.pipe_id,
        topic: artifact.topic,
        tone: pipe?.tone ?? "",
      });
    });
    artifactsList.append(card);
  });
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  let html = "";
  let inUl = false;
  let inOl = false;

  const closeLists = () => {
    if (inUl) {
      html += "</ul>";
      inUl = false;
    }
    if (inOl) {
      html += "</ol>";
      inOl = false;
    }
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trimEnd();
    if (!line.trim()) {
      closeLists();
      html += "";
      return;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      closeLists();
      const level = headingMatch[1].length;
      const text = headingMatch[2];
      html += `<h${level}>${inlineFormat(text)}</h${level}>`;
      return;
    }

    const olMatch = line.match(/^\d+\.\s+(.*)$/);
    if (olMatch) {
      if (!inOl) {
        closeLists();
        html += "<ol>";
        inOl = true;
      }
      html += `<li>${inlineFormat(olMatch[1])}</li>`;
      return;
    }

    const ulMatch = line.match(/^[-*]\s+(.*)$/);
    if (ulMatch) {
      if (!inUl) {
        closeLists();
        html += "<ul>";
        inUl = true;
      }
      html += `<li>${inlineFormat(ulMatch[1])}</li>`;
      return;
    }

    closeLists();
    html += `<p>${inlineFormat(line)}</p>`;
  });

  closeLists();
  return html;
}

function inlineFormat(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

async function showResult({ source, pipeId, topic, tone }) {
  const pipe = state.pipes.get(pipeId);
  const draftPath = ARTIFACT_DRAFT_PATH(pipeId);
  const reportPath = ARTIFACT_REPORT_PATH(pipeId);

  const [draftContent, reportContent] = await Promise.all([
    fetch(draftPath).then((res) => res.text()),
    fetch(reportPath).then((res) => res.json()),
  ]);

  state.currentResult = {
    pipeId,
    topic: topic || reportContent.topic,
    tone: tone || reportContent.tone || pipe?.tone,
    draftPath,
    reportPath,
    jsonldPath: ARTIFACT_JSONLD_PATH(pipeId),
    generatedAt: reportContent.generated_at,
  };

  resultTitle.textContent = state.currentResult.topic || "Результат генерации";
  const metaParts = [];
  if (pipe) metaParts.push(`Тематика: ${pipe.name}`);
  if (state.currentResult.tone) metaParts.push(`Тон: ${state.currentResult.tone}`);
  if (state.currentResult.generatedAt) metaParts.push(`Дата: ${formatDate(state.currentResult.generatedAt)}`);
  resultMeta.textContent = metaParts.join(" • ");

  draftView.innerHTML = markdownToHtml(draftContent);
  renderReport(reportContent);

  downloadMdBtn.disabled = false;
  downloadReportBtn.disabled = false;
  downloadMdBtn.onclick = () => downloadFile(draftPath, `${pipeId}.md`);
  downloadReportBtn.onclick = () => downloadFile(reportPath, `${pipeId}-report.json`);

  switchTab("result");
}

function renderReport(report) {
  if (!report) {
    reportView.innerHTML = '<div class="empty-state">Отчет отсутствует.</div>';
    return;
  }

  const keywords = report.metrics?.keywords ?? { found: [], missing: [] };
  const length = report.metrics?.length;
  const structure = report.metrics?.structure ?? {};
  const jsonld = report.metrics?.jsonld;

  reportView.innerHTML = "";

  const keywordsSection = document.createElement("div");
  keywordsSection.className = "report-section";
  keywordsSection.innerHTML = `
    <h4>Ключевые слова</h4>
    <div class="metric-row">
      <span>Найдены</span>
      <span class="badge ok">${keywords.found.length}</span>
    </div>
    <ul>${keywords.found.map((kw) => `<li>${kw}</li>`).join("") || "<li>—</li>"}</ul>
    <div class="metric-row">
      <span>Отсутствуют</span>
      <span class="badge ${keywords.missing.length ? "missing" : "ok"}">${keywords.missing.length}</span>
    </div>
    <ul>${keywords.missing.map((kw) => `<li>${kw}</li>`).join("") || "<li>—</li>"}</ul>
  `;

  const lengthSection = document.createElement("div");
  lengthSection.className = "report-section";
  const lengthText = length
    ? `${length.actual.toLocaleString()} ${length.unit} / цель ${length.target.toLocaleString()} ${length.unit}`
    : "Нет данных";
  lengthSection.innerHTML = `
    <h4>Длина текста</h4>
    <div class="metric-row">
      <span>Фактическая</span>
      <span>${length ? length.actual.toLocaleString() : "—"}</span>
    </div>
    <div class="metric-row">
      <span>Цель</span>
      <span>${length ? length.target.toLocaleString() : "—"}</span>
    </div>
    <p class="card-meta">${lengthText}</p>
  `;

  const structureSection = document.createElement("div");
  structureSection.className = "report-section";
  structureSection.innerHTML = `
    <h4>Структура</h4>
    <div class="metric-row"><span>H1</span><span>${structure.h1 ?? "—"}</span></div>
    <div class="metric-row"><span>H2</span><span>${structure.h2 ?? "—"}</span></div>
    <div class="metric-row"><span>FAQ</span><span>${structure.faq ?? "—"}</span></div>
    <div class="metric-row"><span>Таблицы</span><span>${structure.table ? "Да" : "Нет"}</span></div>
  `;

  const jsonldSection = document.createElement("div");
  jsonldSection.className = "report-section";
  jsonldSection.innerHTML = `
    <h4>JSON-LD</h4>
    <span class="badge ${jsonld ? "ok" : "pending"}">${jsonld ? "Готово" : "Нужно доработать"}</span>
  `;

  reportView.append(keywordsSection, lengthSection, structureSection, jsonldSection);
}

async function downloadFile(path, suggestedName) {
  const response = await fetch(path);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function formatDate(date) {
  try {
    const parsed = new Date(date);
    return parsed.toLocaleString("ru-RU", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch (e) {
    return date;
  }
}

briefForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!pipeSelect.value) return;

  progressOverlay.classList.remove("hidden");

  const formData = new FormData(briefForm);
  const pipeId = formData.get("pipe-select") || pipeSelect.value;
  const topic = document.getElementById("topic-input").value.trim();
  const tone = document.getElementById("tone-select").value;

  setTimeout(async () => {
    progressOverlay.classList.add("hidden");
    await showResult({
      source: "brief",
      pipeId,
      topic: topic || state.pipes.get(pipeId)?.sample_topics?.[0],
      tone,
    });
  }, 1200 + Math.random() * 500);
});

function initFormDefaults() {
  pipeSelect.name = "pipe-select";
  document.getElementById("tone-select").value = "экспертный";
}

async function bootstrap() {
  initFormDefaults();
  await loadPipes();
  await loadArtifacts();
}

bootstrap().catch((error) => {
  console.error("Не удалось инициализировать демо", error);
});
