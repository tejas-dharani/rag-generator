let selectedFiles = [];

function renderFileList() {
  document.getElementById("filelist").innerHTML = selectedFiles
    .map((f, i) => `<li>${escapeHtml(f.name)} · ${formatBytes(f.size)} <button type="button" class="icon-btn" data-i="${i}" title="Remove">×</button></li>`)
    .join("");
  document.querySelectorAll("#filelist .icon-btn").forEach((b) =>
    b.addEventListener("click", () => { selectedFiles.splice(Number(b.dataset.i), 1); renderFileList(); })
  );
}

async function loadCollections() {
  const container = document.getElementById("collections");
  const collections = await api("/api/collections");
  if (!collections.length) {
    container.innerHTML = `<p class="empty">No apps yet. Create one above.</p>`;
    return;
  }
  container.innerHTML = collections.map((c) => `
    <div class="card collection-card">
      <h3><a href="/c/${c.id}">${escapeHtml(c.name)}</a></h3>
      <div class="desc">${escapeHtml(c.description) || "&nbsp;"}</div>
      <div class="meta">${c.document_count} document${c.document_count === 1 ? "" : "s"} · ${c.chunk_count} chunks</div>
      <div class="buttons">
        <a class="btn" href="/c/${c.id}">Open</a>
        <button class="danger" data-id="${c.id}">Delete</button>
      </div>
    </div>`).join("");
  container.querySelectorAll("button.danger").forEach((b) =>
    armedDelete(b, async () => { await api(`/api/collections/${b.dataset.id}`, { method: "DELETE" }); loadCollections(); })
  );
}

function numberOrNull(id) {
  const value = document.getElementById(id).value;
  return value === "" ? null : Number(value);
}

document.getElementById("create-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = document.getElementById("create-btn");
  const error = document.getElementById("create-error");
  error.textContent = "";
  if (!selectedFiles.length) {
    error.textContent = "Add at least one document.";
    return;
  }
  button.disabled = true;
  button.textContent = "Creating…";
  try {
    const collection = await api("/api/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: document.getElementById("name").value,
        description: document.getElementById("description").value,
        chunk_size: numberOrNull("chunk_size"),
        chunk_overlap: numberOrNull("chunk_overlap"),
        top_k: numberOrNull("top_k"),
      }),
    });
    button.textContent = "Uploading…";
    await uploadFiles(collection.id, selectedFiles);
    window.location.href = `/c/${collection.id}`;
  } catch (err) {
    error.textContent = err.message;
    button.disabled = false;
    button.textContent = "Create app";
  }
});

setupDropzone(document.getElementById("dropzone"), (files) => { selectedFiles.push(...files); renderFileList(); });
showModelInfo();
loadCollections().catch((err) => {
  document.getElementById("collections").innerHTML = `<p class="error">${escapeHtml(err.message)}</p>`;
});
