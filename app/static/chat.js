const collectionId = window.location.pathname.split("/").pop();
const messagesEl = document.getElementById("messages");
let history = [];
let pollTimer = null;
let busy = false;

// ---- documents sidebar ----

async function loadCollection() {
  try {
    const c = await api(`/api/collections/${collectionId}`);
    document.title = `${c.name} · RAG Generator`;
    document.getElementById("title").textContent = c.name;
    document.getElementById("collection-name").textContent = c.name;
    document.getElementById("collection-desc").textContent = c.description;
  } catch (err) {
    messagesEl.innerHTML = `<div class="welcome"><h2>App not found</h2><p><a href="/">Back to all apps</a></p></div>`;
    document.getElementById("ask-form").remove();
    throw err;
  }
  const origin = window.location.origin;
  document.getElementById("api-snippet").textContent =
    `curl -X POST ${origin}/api/collections/${collectionId}/query \\\n  -H "Content-Type: application/json" \\\n  -d '{"question": "..."}'`;
}

async function loadDocuments() {
  const docs = await api(`/api/collections/${collectionId}/documents`);
  const list = document.getElementById("docs");
  list.innerHTML = docs.length ? docs.map((d) => `
    <li>
      <div class="name">
        ${escapeHtml(d.filename)}
        <div class="sub"><span class="badge ${d.status}">${d.status}</span>
          ${d.status === "ready" ? `${d.chunk_count} chunks · ` : ""}${formatBytes(d.size_bytes)}</div>
        ${d.error ? `<div class="err">${escapeHtml(d.error)}</div>` : ""}
      </div>
      <button class="icon-btn" data-id="${d.id}" title="Remove document">×</button>
    </li>`).join("") : `<li class="sub">No documents yet.</li>`;

  list.querySelectorAll(".icon-btn").forEach((b) => {
    b.addEventListener("click", async () => {
      b.disabled = true;
      await api(`/api/collections/${collectionId}/documents/${b.dataset.id}`, { method: "DELETE" });
      loadDocuments();
    });
  });

  // Keep polling while anything is still being indexed.
  clearTimeout(pollTimer);
  if (docs.some((d) => d.status === "processing")) pollTimer = setTimeout(loadDocuments, 1500);
}

setupDropzone(document.getElementById("dropzone"), async (files) => {
  const error = document.getElementById("upload-error");
  error.textContent = "";
  try {
    const result = await uploadFiles(collectionId, files);
    const problems = result.documents.filter((d) => d.status === "rejected").map((d) => `${d.filename}: ${d.error}`);
    error.textContent = problems.join(" ");
  } catch (err) {
    error.textContent = err.message;
  }
  loadDocuments();
});

// ---- rendering answers ----

// Minimal, safe Markdown: escape first, then add paragraphs, lists, bold, inline code and citations.
function renderAnswer(text, messageId) {
  const lines = escapeHtml(text).split("\n");
  let html = "";
  let list = null;
  for (const raw of lines) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)/);
    const numbered = line.match(/^\d+[.)]\s+(.*)/);
    if (bullet || numbered) {
      const tag = bullet ? "ul" : "ol";
      if (list !== tag) { if (list) html += `</${list}>`; html += `<${tag}>`; list = tag; }
      html += `<li>${(bullet || numbered)[1]}</li>`;
      continue;
    }
    if (list) { html += `</${list}>`; list = null; }
    if (line) html += `<p>${line}</p>`;
  }
  if (list) html += `</${list}>`;
  return html
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d+)\]/g, `<span class="cite" data-msg="${messageId}" data-n="$1">$1</span>`);
}

function renderSources(sources, cited, messageId) {
  if (!sources.length) return "";
  const citedSet = new Set(cited);
  const items = sources.map((s) => `
    <div class="source ${citedSet.has(s.number) ? "cited" : ""}" id="src-${messageId}-${s.number}">
      <div class="head">[${s.number}] ${escapeHtml(s.filename)}${s.page ? `, page ${s.page}` : ""}
        <span class="score">relevance ${s.score.toFixed(2)}</span></div>
      <div class="text">${escapeHtml(s.text)}</div>
    </div>`).join("");
  const label = cited.length ? `${cited.length} cited of ${sources.length} sources retrieved` : `${sources.length} sources retrieved`;
  return `<details class="sources"><summary>${label}</summary>${items}</details>`;
}

messagesEl.addEventListener("click", (e) => {
  const cite = e.target.closest(".cite");
  if (!cite) return;
  const source = document.getElementById(`src-${cite.dataset.msg}-${cite.dataset.n}`);
  if (!source) return;
  source.closest("details").open = true;
  source.classList.remove("flash");
  void source.offsetWidth; // restart the highlight animation
  source.classList.add("flash");
  source.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

function addMessage(role, html) {
  document.getElementById("welcome")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="bubble">${html}</div>`;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

// ---- asking questions (streamed over Server-Sent Events) ----

async function ask(question) {
  busy = true;
  document.getElementById("send").disabled = true;
  addMessage("user", escapeHtml(question));
  const messageId = Date.now();
  const el = addMessage("assistant", `<span class="status-line typing">Searching documents</span>`);
  const bubble = el.querySelector(".bubble");

  let answer = "";
  let sources = [];
  let done = null;
  try {
    const response = await fetch(`/api/collections/${collectionId}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: history.slice(-6), stream: true }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const chunk of events) {
        if (!chunk.startsWith("data: ")) continue;
        const event = JSON.parse(chunk.slice(6));
        if (event.type === "sources") {
          sources = event.sources;
          bubble.innerHTML = `<span class="status-line typing">Found ${sources.length} relevant passages · writing answer</span>`;
        } else if (event.type === "token") {
          answer += event.text;
          bubble.innerHTML = `<div class="typing">${renderAnswer(answer, messageId)}</div>`;
        } else if (event.type === "done") {
          done = event;
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  } catch (err) {
    bubble.innerHTML = `<span class="error">Error: ${escapeHtml(err.message)}</span>`;
  }

  if (done) {
    bubble.classList.toggle("not-found", !done.grounded);
    bubble.innerHTML = renderAnswer(done.answer, messageId) + renderSources(sources, done.cited, messageId);
    history.push({ role: "user", content: question }, { role: "assistant", content: done.answer });
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
  busy = false;
  document.getElementById("send").disabled = false;
}

const questionEl = document.getElementById("question");
document.getElementById("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  if (!question || busy) return;
  questionEl.value = "";
  questionEl.style.height = "";
  ask(question);
});
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("ask-form").requestSubmit();
  }
});
questionEl.addEventListener("input", () => {
  questionEl.style.height = "auto";
  questionEl.style.height = `${questionEl.scrollHeight}px`;
});

document.getElementById("clear").addEventListener("click", () => {
  history = [];
  messagesEl.innerHTML = `<div class="welcome" id="welcome"><h2>New conversation</h2><p>Ask a question about these documents.</p></div>`;
});

showModelInfo();
loadCollection().then(loadDocuments).catch(() => {});
