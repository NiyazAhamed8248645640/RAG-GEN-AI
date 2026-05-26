/* ── Configuration ─────────────────────────────────────────────────────── */
const API_BASE = window.location.origin;
const STORAGE_KEY = "rag_session_id";

/* ── DOM refs ──────────────────────────────────────────────────────────── */
const messagesEl = document.getElementById("chatMessages");
const inputEl    = document.getElementById("userInput");
const sendBtn    = document.getElementById("sendBtn");
const loadingEl  = document.getElementById("loadingIndicator");
const statusEl   = document.getElementById("statusText");
const charCount  = document.getElementById("charCount");
const newChatBtn = document.getElementById("newChatBtn");
const sessionDisplay = document.getElementById("sessionDisplay");

/* ── Session management ─────────────────────────────────────────────────── */
function generateSessionId() {
  return "sess_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
}

function getSessionId() {
  let id = localStorage.getItem(STORAGE_KEY);
  if (!id) {
    id = generateSessionId();
    localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}

function refreshSession() {
  const id = generateSessionId();
  localStorage.setItem(STORAGE_KEY, id);
  sessionDisplay.textContent = id.slice(-8);
  return id;
}

// Init session on load
let sessionId = getSessionId();
sessionDisplay.textContent = sessionId.slice(-8);

/* ── Marked.js config ───────────────────────────────────────────────────── */
if (typeof marked !== "undefined") {
  marked.setOptions({ breaks: true, gfm: true });
}

/* ── Helpers ────────────────────────────────────────────────────────────── */
function renderMarkdown(text) {
  if (typeof marked !== "undefined") {
    return marked.parse(text);
  }
  return text.replace(/\n/g, "<br>");
}

function formatTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "smooth" });
}

function setStatus(msg, type = "") {
  statusEl.textContent = msg;
  statusEl.className = "status-text " + type;
  if (type !== "error" && msg) {
    setTimeout(() => { statusEl.textContent = ""; statusEl.className = "status-text"; }, 3000);
  }
}

/* ── Message rendering ──────────────────────────────────────────────────── */
function addMessage({ role, text, chunks = null, isError = false }) {
  // Remove welcome message on first real message
  const welcome = document.getElementById("welcomeMsg");
  if (welcome && messagesEl.contains(welcome)) welcome.remove();

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}-message${isError ? " error-message" : ""}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}-avatar`;
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubbleWrap = document.createElement("div");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = role === "assistant" ? renderMarkdown(text) : escapeHtml(text);

  // Chunk badge for assistant messages
  if (role === "assistant" && chunks !== null && chunks > 0) {
    const badge = document.createElement("div");
    badge.className = "chunk-badge";
    badge.textContent = `📚 ${chunks} source chunk${chunks > 1 ? "s" : ""} retrieved`;
    bubble.appendChild(badge);
  }

  const time = document.createElement("div");
  time.className = "timestamp";
  time.textContent = formatTime();

  bubbleWrap.appendChild(bubble);
  bubbleWrap.appendChild(time);

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubbleWrap);

  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/\n/g, "<br>");
}

/* ── API call ───────────────────────────────────────────────────────────── */
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || sendBtn.disabled) return;

  // Optimistic UI
  addMessage({ role: "user", text });
  inputEl.value = "";
  charCount.textContent = "0 / 2000";
  autoResize();

  setLoading(true);
  setStatus("Retrieving relevant context…");

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, message: text }),
    });

    const data = await res.json();

    if (!res.ok) {
      const errMsg = data?.detail?.error || data?.error || "Something went wrong.";
      addMessage({ role: "assistant", text: `⚠️ ${errMsg}`, isError: true });
      setStatus("Error from server", "error");
      return;
    }

    addMessage({
      role: "assistant",
      text: data.reply,
      chunks: data.retrievedChunks,
    });

    const tokenInfo = data.tokensUsed ? `${data.tokensUsed} tokens` : "";
    setStatus(tokenInfo ? `✓ ${tokenInfo} used` : "✓ Response ready", "success");

  } catch (err) {
    console.error("Fetch error:", err);
    addMessage({
      role: "assistant",
      text: "⚠️ Could not reach the server. Please check your connection.",
      isError: true,
    });
    setStatus("Connection error", "error");
  } finally {
    setLoading(false);
  }
}

/* ── New chat ───────────────────────────────────────────────────────────── */
async function startNewChat() {
  // Tell server to clear this session's history
  try {
    await fetch(`${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
  } catch (_) { /* ignore */ }

  // Generate new session
  sessionId = refreshSession();

  // Clear UI
  messagesEl.innerHTML = `
    <div class="message assistant-message" id="welcomeMsg">
      <div class="avatar assistant-avatar">AI</div>
      <div class="bubble">
        <p>👋 New conversation started! How can I help you?</p>
      </div>
    </div>`;
  setStatus("New session started", "success");
}

/* ── Loading state ──────────────────────────────────────────────────────── */
function setLoading(on) {
  loadingEl.classList.toggle("hidden", !on);
  sendBtn.disabled = on;
  inputEl.disabled = on;
  if (on) scrollToBottom();
}

/* ── Auto-resize textarea ───────────────────────────────────────────────── */
function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
}

/* ── Event listeners ────────────────────────────────────────────────────── */
sendBtn.addEventListener("click", sendMessage);
newChatBtn.addEventListener("click", startNewChat);

inputEl.addEventListener("input", () => {
  autoResize();
  charCount.textContent = `${inputEl.value.length} / 2000`;
});

inputEl.addEventListener("keydown", (e) => {
  // Send on Enter; Shift+Enter = newline
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Focus input on load
window.addEventListener("load", () => inputEl.focus());
