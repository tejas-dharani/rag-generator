// Shared helpers for the RAG Generator UI.

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

const ACCEPTED = ".pdf,.docx,.md,.markdown,.txt,.html,.htm";

// Wire a drop zone element (containing a hidden <input type=file>) to a callback receiving File[].
function setupDropzone(zone, onFiles) {
  const input = zone.querySelector("input[type=file]");
  input.accept = ACCEPTED;
  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { onFiles([...input.files]); input.value = ""; });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    onFiles([...e.dataTransfer.files]);
  });
}

async function uploadFiles(collectionId, files) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return api(`/api/collections/${collectionId}/documents`, { method: "POST", body: form });
}

// Two-step delete button instead of a blocking confirm() dialog.
function armedDelete(button, action, label = "Delete") {
  let timer;
  button.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!button.classList.contains("armed")) {
      button.classList.add("armed");
      button.textContent = "Confirm?";
      timer = setTimeout(() => { button.classList.remove("armed"); button.textContent = label; }, 3000);
      return;
    }
    clearTimeout(timer);
    button.disabled = true;
    await action();
  });
}

async function showModelInfo() {
  try {
    const h = await api("/health");
    document.getElementById("model").textContent = `${h.llm_provider} · ${h.llm_model} · ${h.embedding_model}`;
  } catch (_) { /* header info is optional */ }
}
