const drop = document.getElementById("drop");
const input = document.getElementById("file");
const preview = document.getElementById("preview");
const prompt = document.getElementById("prompt");
const result = document.getElementById("result");
const verdict = document.getElementById("verdict");
const band = document.getElementById("band");
const marker = document.getElementById("marker");
const detail = document.getElementById("detail");
const error = document.getElementById("error");

const VERDICTS = {
  likely_ai: "Likely AI-generated",
  uncertain: "Can’t tell",
  likely_real: "No sign of AI generation",
};

const ERRORS = {
  invalid_image: "Couldn’t read that image. JPEG, PNG, WebP and GIF work.",
  image_too_large: "That image is over the size limit.",
  rate_limited: "Too many checks in a minute. Wait a moment and try again.",
  detector_busy: "The detector is busy. Try again in a few seconds.",
};

const promptText = prompt.innerHTML;
let latest = 0;

async function check(file) {
  if (!file.type.startsWith("image/")) {
    showError("That file isn’t an image.");
    return;
  }

  const id = ++latest;
  URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  drop.classList.add("busy");
  prompt.textContent = "Checking…";
  error.hidden = true;

  const started = performance.now();
  try {
    const res = await fetch("/api/v1/verify", {
      method: "POST",
      headers: { "content-type": file.type },
      body: file,
    });
    const body = await res.json();
    if (id !== latest) return;
    if (!res.ok) {
      showError(
        ERRORS[body.error?.code] ?? body.error?.message ?? `Request failed (${res.status}).`,
      );
      return;
    }
    showResult(body, performance.now() - started);
  } catch {
    if (id === latest) showError("Couldn’t reach the server.");
  } finally {
    if (id === latest) {
      drop.classList.remove("busy");
      prompt.innerHTML = promptText;
    }
  }
}

function showResult(body, elapsed) {
  const percent = (p) => `${(p * 100).toFixed(2)}%`;
  result.dataset.verdict = body.verdict;
  verdict.textContent = VERDICTS[body.verdict];
  band.style.left = percent(body.thresholds.real);
  band.style.width = percent(body.thresholds.ai - body.thresholds.real);
  marker.style.left = percent(body.aiProbability);

  const source = body.cached ? "from the cache" : "by the detector";
  detail.innerHTML = "";
  detail.append(
    `AI score ${body.aiProbability.toFixed(3)}, answered in ${Math.round(elapsed)} ms ${source}.`,
    document.createElement("br"),
    `Model ${body.model}. SHA-256 `,
    Object.assign(document.createElement("code"), { textContent: body.sha256 }),
  );
  result.hidden = false;
}

function showError(message) {
  error.textContent = message;
  error.hidden = false;
  result.hidden = true;
}

function firstImage(files) {
  return [...files].find((file) => file.type.startsWith("image/")) ?? files[0];
}

input.addEventListener("change", () => {
  if (input.files.length) check(input.files[0]);
  input.value = "";
});

document.addEventListener("paste", (event) => {
  const file = firstImage(event.clipboardData.files);
  if (file) check(file);
});

// Dropping anywhere on the page counts, and stops the browser from opening the file.
window.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("dragging");
});
window.addEventListener("dragleave", (event) => {
  if (!event.relatedTarget) drop.classList.remove("dragging");
});
window.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("dragging");
  const file = firstImage(event.dataTransfer.files);
  if (file) check(file);
});
