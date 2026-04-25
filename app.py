"""
app.py — CodeBase RAG Assistant  ·  God-Mode Edition
Cinematic dark UI with matrix rain, glassmorphism, 3D cards,
particle system, typewriter reveals, and smooth transitions.
"""

import os
import streamlit as st
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx

from ingest import ingest, append_qa_to_store, save_store, log_conversation
from rag_chain import build_chain, ask
from config import GROQ_API_KEY

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG.DEV — CodeBase Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── God-Mode CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ─── Imports ────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Geist+Mono:wght@300;400;500&display=swap');

/* ─── Tokens ─────────────────────────────────────────────────────────────── */
:root {
  --bg:        #050710;
  --surface:   #0c0f1e;
  --panel:     #111427;
  --border:    rgba(99,102,241,.18);
  --border-hi: rgba(99,102,241,.55);
  --accent:    #6366f1;
  --accent2:   #22d3ee;
  --accent3:   #f472b6;
  --glow:      rgba(99,102,241,.35);
  --glow2:     rgba(34,211,238,.25);
  --text:      #e2e8f0;
  --muted:     #64748b;
  --code-bg:   #080a14;
  --r:         12px;
  --r-lg:      20px;
}

/* ─── Reset ──────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Syne', sans-serif;
  background: var(--bg) !important;
  color: var(--text);
}
code, pre, .stCodeBlock, .stCode { font-family: 'Geist Mono', monospace !important; }

/* ─── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 99px; }

/* ─── Streamlit chrome wipe ──────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1200px; }
.stApp { background: var(--bg) !important; }

/* ─── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
  backdrop-filter: blur(12px);
}
section[data-testid="stSidebar"] > div { padding: 1.6rem 1.2rem; }

/* Sidebar logo */
.sidebar-logo {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 1.8rem;
}
.sidebar-logo .hex {
  width: 40px; height: 40px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  clip-path: polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
  display: flex; align-items: center; justify-content: center;
  animation: hexSpin 8s linear infinite;
  font-size: 18px;
}
.sidebar-logo .wordmark {
  font-size: 1.15rem; font-weight: 800; letter-spacing: .04em;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sidebar-logo .sub {
  font-size: 0.68rem; color: var(--muted);
  font-family: 'Geist Mono', monospace;
  letter-spacing: .06em; text-transform: uppercase;
}

/* Sidebar input */
section[data-testid="stSidebar"] .stTextInput input {
  background: var(--code-bg) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--r) !important;
  font-family: 'Geist Mono', monospace !important;
  font-size: 0.82rem !important;
  padding: .55rem .9rem !important;
  transition: border-color .2s, box-shadow .2s;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--glow) !important;
}
section[data-testid="stSidebar"] .stTextInput label {
  color: var(--muted) !important;
  font-size: .72rem !important;
  font-family: 'Geist Mono', monospace !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
}

/* Sidebar checkbox */
section[data-testid="stSidebar"] .stCheckbox label {
  color: var(--muted) !important;
  font-size: .82rem !important;
}

/* Sidebar primary button */
section[data-testid="stSidebar"] .stButton:first-of-type > button,
.ingest-btn {
  background: linear-gradient(135deg, var(--accent) 0%, #4f46e5 50%, var(--accent2) 100%) !important;
  background-size: 200% 200% !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r) !important;
  padding: .65rem 1.4rem !important;
  font-weight: 700 !important;
  font-size: .9rem !important;
  width: 100% !important;
  letter-spacing: .04em !important;
  transition: all .3s ease !important;
  animation: gradientShift 4s ease infinite !important;
  box-shadow: 0 0 20px var(--glow) !important;
  position: relative; overflow: hidden;
}
section[data-testid="stSidebar"] .stButton:first-of-type > button::after {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent);
  transform: translateX(-100%);
  transition: transform .5s;
}
section[data-testid="stSidebar"] .stButton:first-of-type > button:hover::after {
  transform: translateX(100%);
}
section[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px var(--glow), 0 0 60px var(--glow2) !important;
}

/* Secondary sidebar buttons */
section[data-testid="stSidebar"] .stButton:not(:first-of-type) > button {
  background: transparent !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  padding: .5rem 1rem !important;
  font-size: .82rem !important;
  width: 100% !important;
  transition: all .2s !important;
}
section[data-testid="stSidebar"] .stButton:not(:first-of-type) > button:hover {
  border-color: var(--accent) !important;
  color: var(--text) !important;
  background: rgba(99,102,241,.08) !important;
}

/* Divider */
section[data-testid="stSidebar"] hr {
  border-color: var(--border) !important; margin: 1.2rem 0 !important;
}

/* ─── Stat cards ─────────────────────────────────────────────────────────── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: .5rem 0; }
.stat-card {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: .9rem .8rem;
  text-align: center;
  position: relative; overflow: hidden;
  transition: border-color .2s, box-shadow .2s;
}
.stat-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.stat-card:hover {
  border-color: var(--border-hi);
  box-shadow: 0 0 20px var(--glow);
}
.stat-val {
  font-size: 1.6rem; font-weight: 800;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1;
}
.stat-lbl {
  color: var(--muted); font-size: 0.7rem;
  font-family: 'Geist Mono', monospace;
  letter-spacing: .08em; text-transform: uppercase;
  margin-top: 4px;
}

/* ─── Hero ───────────────────────────────────────────────────────────────── */
.hero {
  position: relative; overflow: hidden;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 3.5rem 3rem 2.5rem;
  margin-bottom: 2rem;
  animation: heroIn .7s cubic-bezier(.16,1,.3,1) both;
}
/* Animated mesh gradient */
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 60% 50% at 80% 20%, rgba(99,102,241,.18) 0%, transparent 60%),
    radial-gradient(ellipse 40% 60% at 10% 80%, rgba(34,211,238,.12) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 50% 110%, rgba(244,114,182,.08) 0%, transparent 50%);
  animation: meshDrift 10s ease-in-out infinite;
}
/* Scanning line */
.hero::after {
  content: '';
  position: absolute; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent2), transparent);
  opacity: .5;
  animation: scanLine 4s linear infinite;
}
.hero-inner { position: relative; z-index: 1; }
.hero-eyebrow {
  font-family: 'Geist Mono', monospace;
  font-size: .72rem; letter-spacing: .2em; text-transform: uppercase;
  color: var(--accent2);
  margin-bottom: .8rem;
  display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
  content: '';
  width: 24px; height: 1px;
  background: var(--accent2);
  display: block;
}
.hero h1 {
  font-size: clamp(2rem, 4vw, 3.2rem);
  font-weight: 800;
  line-height: 1.1;
  margin: 0 0 1rem;
  color: var(--text);
}
.hero h1 .grad {
  background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
  background-size: 200%;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: gradientShift 4s linear infinite;
}
.hero-sub {
  color: var(--muted); font-size: 1rem; max-width: 520px;
  line-height: 1.6; margin-bottom: 1.8rem;
}
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; }
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(99,102,241,.08);
  color: var(--accent2);
  border: 1px solid rgba(99,102,241,.2);
  font-size: .75rem; font-weight: 600;
  font-family: 'Geist Mono', monospace;
  padding: 4px 12px; border-radius: 99px;
  letter-spacing: .04em;
  transition: all .2s;
}
.badge:hover {
  background: rgba(99,102,241,.18);
  border-color: var(--accent);
  box-shadow: 0 0 12px var(--glow);
  transform: translateY(-1px);
}

/* ─── Feature grid ───────────────────────────────────────────────────────── */
.feat-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin: 1.5rem 0;
}
.feat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 1.3rem;
  position: relative; overflow: hidden;
  transition: all .3s cubic-bezier(.16,1,.3,1);
  animation: cardIn .6s cubic-bezier(.16,1,.3,1) both;
}
.feat-card:nth-child(1) { animation-delay: .05s; }
.feat-card:nth-child(2) { animation-delay: .1s; }
.feat-card:nth-child(3) { animation-delay: .15s; }
.feat-card:nth-child(4) { animation-delay: .2s; }
.feat-card:nth-child(5) { animation-delay: .25s; }
.feat-card:nth-child(6) { animation-delay: .3s; }
.feat-card::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at var(--mx,50%) var(--my,50%), rgba(99,102,241,.1) 0%, transparent 60%);
  opacity: 0; transition: opacity .3s;
}
.feat-card:hover { border-color: var(--border-hi); transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,.4), 0 0 20px var(--glow); }
.feat-card:hover::before { opacity: 1; }
.feat-icon { font-size: 1.5rem; margin-bottom: .6rem; }
.feat-title { font-size: .88rem; font-weight: 700; color: var(--text); margin-bottom: .3rem; }
.feat-desc { font-size: .78rem; color: var(--muted); line-height: 1.5; font-family: 'Geist Mono', monospace; }

/* ─── Chat header ────────────────────────────────────────────────────────── */
.chat-hdr {
  display: flex; align-items: center; gap: 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1rem 1.5rem;
  margin-bottom: 1.5rem;
  position: relative; overflow: hidden;
  animation: heroIn .5s cubic-bezier(.16,1,.3,1) both;
}
.chat-hdr::before {
  content: '';
  position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: linear-gradient(180deg, var(--accent), var(--accent2));
}
.chat-hdr .pulse-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent2);
  box-shadow: 0 0 8px var(--accent2);
  animation: pulseDot 2s ease-in-out infinite;
  flex-shrink: 0;
}
.chat-hdr .repo {
  font-weight: 700; font-size: 1rem; color: var(--text); flex: 1;
}
.chat-hdr .repo code {
  font-family: 'Geist Mono', monospace !important;
  font-size: .88rem; color: var(--accent2);
  background: rgba(34,211,238,.08);
  padding: 2px 8px; border-radius: 6px;
}
.hdr-tag {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: .7rem; font-weight: 600;
  font-family: 'Geist Mono', monospace;
  padding: 3px 10px; border-radius: 99px;
  letter-spacing: .06em;
}
.tag-cached { background: rgba(99,102,241,.12); color: var(--accent); border: 1px solid rgba(99,102,241,.2); }
.tag-qa { background: rgba(34,211,238,.1); color: var(--accent2); border: 1px solid rgba(34,211,238,.2); }

/* ─── Chat messages ──────────────────────────────────────────────────────── */
.stChatMessage {
  border-radius: var(--r) !important;
  border: 1px solid var(--border) !important;
  margin-bottom: .75rem !important;
  background: var(--surface) !important;
  animation: msgIn .4s cubic-bezier(.16,1,.3,1) both !important;
  backdrop-filter: blur(8px) !important;
}
.stChatMessage[data-testid="chat-message-user"] {
  border-color: rgba(99,102,241,.25) !important;
  background: rgba(99,102,241,.06) !important;
}
.stChatMessage[data-testid="chat-message-assistant"] {
  border-color: rgba(34,211,238,.2) !important;
  background: rgba(34,211,238,.04) !important;
}

/* ─── Chat input ─────────────────────────────────────────────────────────── */
.stChatInputContainer {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  box-shadow: 0 0 30px rgba(0,0,0,.5), 0 0 60px var(--glow) !important;
  padding: .5rem !important;
  transition: box-shadow .3s, border-color .3s !important;
}
.stChatInputContainer:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 40px var(--glow), 0 0 80px var(--glow2) !important;
}
.stChatInputContainer textarea {
  background: transparent !important;
  color: var(--text) !important;
  font-family: 'Geist Mono', monospace !important;
  font-size: .9rem !important;
}
.stChatInputContainer button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  border-radius: 10px !important;
  box-shadow: 0 0 16px var(--glow) !important;
}

/* ─── Source chips ───────────────────────────────────────────────────────── */
.src-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: .6rem; }
.src-chip {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(99,102,241,.08);
  color: var(--accent);
  border: 1px solid rgba(99,102,241,.2);
  font-family: 'Geist Mono', monospace;
  font-size: .7rem;
  padding: 3px 10px; border-radius: 99px;
  transition: all .2s;
}
.src-chip:hover { background: rgba(99,102,241,.18); border-color: var(--accent); }
.src-chip .dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--accent2);
  box-shadow: 0 0 4px var(--accent2);
}

/* ─── Empty state ────────────────────────────────────────────────────────── */
.empty-state {
  text-align: center; padding: 3rem 1rem 2rem;
  animation: heroIn .8s cubic-bezier(.16,1,.3,1) .3s both;
}
.empty-orb {
  width: 80px; height: 80px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius: 50%;
  margin: 0 auto 1.2rem;
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem;
  box-shadow: 0 0 40px var(--glow), 0 0 80px var(--glow2);
  animation: orbFloat 4s ease-in-out infinite;
}
.empty-state h3 { font-size: 1.4rem; font-weight: 700; color: var(--text); margin-bottom: .5rem; }
.empty-state p { color: var(--muted); font-size: .88rem; font-family: 'Geist Mono', monospace; max-width: 360px; margin: 0 auto; }

/* ─── Section title ──────────────────────────────────────────────────────── */
.section-title {
  font-size: .72rem; font-weight: 700;
  font-family: 'Geist Mono', monospace;
  letter-spacing: .18em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 1rem;
  display: flex; align-items: center; gap: 8px;
}
.section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ─── Progress / spinner ─────────────────────────────────────────────────── */
.stProgress > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; border-radius: 99px !important; }

/* ─── Expander ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--code-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  color: var(--muted) !important;
  font-size: .8rem !important;
  font-family: 'Geist Mono', monospace !important;
}
.streamlit-expanderContent {
  background: var(--code-bg) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--r) var(--r) !important;
}

/* ─── Alerts ─────────────────────────────────────────────────────────────── */
.stAlert { border-radius: var(--r) !important; border-left-width: 3px !important; }

/* ─── Caption ────────────────────────────────────────────────────────────── */
.stCaption { color: var(--muted) !important; font-family: 'Geist Mono', monospace !important; font-size: .72rem !important; }

/* ─── Animations ─────────────────────────────────────────────────────────── */
@keyframes heroIn {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes cardIn {
  from { opacity: 0; transform: translateY(12px) scale(.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes msgIn {
  from { opacity: 0; transform: translateX(-8px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes gradientShift {
  0%,100% { background-position: 0% 50%; }
  50%     { background-position: 100% 50%; }
}
@keyframes scanLine {
  0%   { top: -1px; opacity: 0; }
  10%  { opacity: .5; }
  90%  { opacity: .5; }
  100% { top: calc(100% + 1px); opacity: 0; }
}
@keyframes meshDrift {
  0%,100% { opacity: .8; transform: scale(1); }
  50%     { opacity: 1; transform: scale(1.03); }
}
@keyframes hexSpin {
  from { filter: hue-rotate(0deg); }
  to   { filter: hue-rotate(360deg); }
}
@keyframes orbFloat {
  0%,100% { transform: translateY(0); box-shadow: 0 0 40px var(--glow), 0 0 80px var(--glow2); }
  50%     { transform: translateY(-10px); box-shadow: 0 15px 60px var(--glow), 0 0 120px var(--glow2); }
}
@keyframes pulseDot {
  0%,100% { opacity: 1; transform: scale(1); }
  50%     { opacity: .5; transform: scale(.7); }
}

/* ─── Typewriter cursor ──────────────────────────────────────────────────── */
.tw { border-right: 2px solid var(--accent2); animation: blink .8s step-end infinite; padding-right: 2px; }
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

/* ─── Terminal-style sidebar caption ─────────────────────────────────────── */
.terminal-line {
  font-family: 'Geist Mono', monospace;
  font-size: .68rem; color: var(--muted);
  margin: 0; padding: 0;
}
.terminal-line .prompt { color: var(--accent); }
.terminal-line .cmd { color: var(--accent2); }
.terminal-line .arg { color: var(--text); }

/* ─── Matrix canvas (positioned fixed) ───────────────────────────────────── */
#matrix-canvas {
  position: fixed; top: 0; left: 0; pointer-events: none; z-index: 0; opacity: .04;
}
</style>
""", unsafe_allow_html=True)

# ── Matrix rain JS ────────────────────────────────────────────────────────────

st.markdown("""
<canvas id="matrix-canvas"></canvas>
<script>
(function(){
  const c = document.getElementById('matrix-canvas');
  const ctx = c.getContext('2d');
  function resize(){ c.width = window.innerWidth; c.height = window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);
  const cols = () => Math.floor(c.width / 14);
  let drops = [];
  const chars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
  function init(){ drops = Array.from({length: cols()}, () => Math.random() * -100); }
  init();
  window.addEventListener('resize', init);
  function draw(){
    ctx.fillStyle = 'rgba(5,7,16,0.04)';
    ctx.fillRect(0,0,c.width,c.height);
    ctx.fillStyle = '#6366f1';
    ctx.font = '12px "Geist Mono",monospace';
    drops.forEach((y,i)=>{
      const ch = chars[Math.floor(Math.random()*chars.length)];
      ctx.fillStyle = i % 5 === 0 ? '#22d3ee' : '#6366f1';
      ctx.fillText(ch, i*14, y*14);
      if(y*14 > c.height && Math.random() > 0.975) drops[i] = 0;
      drops[i]++;
    });
  }
  setInterval(draw, 50);
})();
</script>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

_DEFAULTS = {
    "chain": None, "messages": [], "repo_url": "", "ingested": False,
    "stats": {}, "from_cache": False, "store": None, "qa_count": 0,
    "ingesting": False, "ingest_error": None, "ingest_progress": 0,
    "ingest_msg": "", "ingest_start_time": 0,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

if st.session_state.get("_needs_rerun"):
    st.session_state["_needs_rerun"] = False
    st.rerun()

# Auto-rebuild chain if it was lost across reruns (complex objects can be dropped)
if st.session_state.ingested and st.session_state.chain is None:
    try:
        from ingest import load_existing
        if st.session_state.store is None:
            st.session_state.store = load_existing()
        if st.session_state.store is not None:
            st.session_state.chain = build_chain(st.session_state.store)
    except Exception:
        st.session_state.ingested = False

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="hex">⬡</div>
      <div>
        <div class="wordmark">RAG.DEV</div>
        <div class="sub">CodeBase Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="terminal-line"><span class="prompt">$</span> <span class="cmd">connect</span> <span class="arg">--repo &lt;url&gt;</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    repo_url = st.text_input("Repository URL", placeholder="https://github.com/user/repo", label_visibility="collapsed")
    st.caption("Paste any public GitHub repo URL above")
    force = st.checkbox("⟳ Force re-ingest (skip cache)")

    ingest_btn = st.button("⚡ Ingest & Analyse")

    if st.session_state.ingesting:
        # Timeout check: 5 minutes max for ingestion
        if time.time() - st.session_state.get("ingest_start_time", 0) > 300:
            st.session_state.ingesting = False
            st.session_state.ingest_error = "Ingestion timed out (5m limit)."
            st.rerun()

        status = st.empty()
        pbar = st.progress(st.session_state.ingest_progress)
        status.info(f"◈ {st.session_state.ingest_msg}")
        pbar.progress(st.session_state.ingest_progress / 100)
        time.sleep(0.5)
        st.rerun()

    if st.session_state.ingest_error:
        st.error(f"✕ {st.session_state.ingest_error}")
        st.session_state.ingest_error = None

    if ingest_btn:
        if not repo_url.strip():
            st.error("⚠ No URL provided.")
        elif not GROQ_API_KEY:
            st.error("⚠ GROQ_API_KEY missing from .env")
        else:
            st.session_state.ingesting = True
            st.session_state.ingest_msg = "Starting ingestion ..."
            st.session_state.ingest_progress = 0
            st.session_state.ingest_start_time = time.time()

            def _bg_ingest():
                try:
                    def _cb(msg):
                        st.session_state.ingest_msg = msg
                        st.session_state.ingest_progress = min(st.session_state.ingest_progress + 15, 95)

                    store, stats, cached = ingest(repo_url, progress_cb=_cb, force=force)
                    chain = build_chain(store)
                    st.session_state.update(
                        chain=chain, store=store, ingested=True,
                        repo_url=repo_url, stats=stats,
                        from_cache=cached, messages=[], qa_count=0,
                        ingesting=False, ingest_progress=100
                    )
                except Exception as e:
                    st.session_state.ingesting = False
                    st.session_state.ingest_error = f"{type(e).__name__}: {e}"

            thread = threading.Thread(target=_bg_ingest, daemon=True)
            add_script_run_ctx(thread)
            thread.start()
            st.rerun()

    # Stats
    if st.session_state.ingested and st.session_state.stats:
        st.markdown("---")
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-val">{st.session_state.stats.get("files", "—")}</div>
            <div class="stat-lbl">Files</div>
          </div>
          <div class="stat-card">
            <div class="stat-val">{st.session_state.stats.get("chunks", "—")}</div>
            <div class="stat-lbl">Chunks</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.ingested:
        st.markdown("---")
        if st.button("⊘ Clear Chat"):
            st.session_state.messages = []
            st.session_state["_needs_rerun"] = True
        if st.button("↺ New Repository"):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.session_state["_needs_rerun"] = True

    st.markdown("---")
    st.markdown("""
    <div class="terminal-line"><span class="prompt">◈</span> <span class="cmd">LangChain</span></div>
    <div class="terminal-line"><span class="prompt">◈</span> <span class="cmd">FAISS</span> <span class="arg">vector store</span></div>
    <div class="terminal-line"><span class="prompt">◈</span> <span class="cmd">HuggingFace</span> <span class="arg">embeddings</span></div>
    <div class="terminal-line"><span class="prompt">◈</span> <span class="cmd">Groq</span> <span class="arg">LLaMA 3.1</span></div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────

if not st.session_state.ingested:

    # Hero
    st.markdown("""
    <div class="hero">
      <div class="hero-inner">
        <div class="hero-eyebrow">Retrieval Augmented Generation</div>
        <h1>Chat with any<br><span class="grad">GitHub codebase</span></h1>
        <p class="hero-sub">
          Ingest any public repository, then ask questions in plain English.
          Powered by FAISS vector search and Groq LLaMA 3.1 for instant,
          context-aware answers.
        </p>
        <div class="badge-row">
          <span class="badge">◈ LangChain</span>
          <span class="badge">▣ FAISS</span>
          <span class="badge">⬡ HuggingFace</span>
          <span class="badge">⚡ Groq LLaMA 3.1</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Empty state
    st.markdown("""
    <div class="empty-state">
      <div class="empty-orb">📡</div>
      <h3>Awaiting repository</h3>
      <p>Paste a GitHub URL in the sidebar panel and hit ⚡ Ingest & Analyse to connect</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    st.markdown('<div class="section-title">Capabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feat-grid">
      <div class="feat-card">
        <div class="feat-icon">🏗️</div>
        <div class="feat-title">Architecture Overview</div>
        <div class="feat-desc">Summarize project structure and folder organization</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🔐</div>
        <div class="feat-title">Auth Flow</div>
        <div class="feat-desc">Trace authentication and authorization logic</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🛣️</div>
        <div class="feat-title">API Endpoints</div>
        <div class="feat-desc">List all routes, handlers, and HTTP methods</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🧩</div>
        <div class="feat-title">Design Patterns</div>
        <div class="feat-desc">Identify patterns: MVC, singleton, factory, etc.</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">🗄️</div>
        <div class="feat-title">Data Layer</div>
        <div class="feat-desc">Locate DB config, schema, and ORM models</div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">📦</div>
        <div class="feat-title">Dependencies</div>
        <div class="feat-desc">Inspect libraries, versions, and package configs</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat header ───────────────────────────────────────────────────────────
    repo_name = st.session_state.repo_url.rstrip("/").split("/")[-1]
    tags_html = ""
    if st.session_state.from_cache:
        tags_html += '<span class="hdr-tag tag-cached">⚡ CACHED</span>'
    if st.session_state.qa_count:
        tags_html += f'<span class="hdr-tag tag-qa">◈ {st.session_state.qa_count} retained</span>'

    st.markdown(f"""
    <div class="chat-hdr">
      <div class="pulse-dot"></div>
      <div class="repo">Active repository: <code>{repo_name}</code></div>
      {tags_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Render previous messages ──────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🔮"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                chips = "".join(
                    f'<span class="src-chip"><span class="dot"></span>{s["file"]}</span>'
                    for s in msg["sources"]
                )
                st.markdown(f'<div class="src-chips">{chips}</div>', unsafe_allow_html=True)
                with st.expander("◈ View source snippets"):
                    for s in msg["sources"]:
                        lang = s["extension"].lstrip(".") or "text"
                        st.caption(f"**{s['file']}**")
                        st.code(s["snippet"][:300], language=lang)

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask anything about this codebase …"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🔮"):
            if st.session_state.chain is None:
                st.error("⚠ Chain not initialized — please re-ingest the repository.")
                answer, sources = "Error: chain not available.", []
            else:
                # ── Real Streaming Reveal ─────────────────────────────────────
                with st.spinner("◈ Searching vector store …"):
                    # Invoke chain.stream() for token-by-token response
                    # Note: source_documents are only available at the end of the stream
                    # or by a separate call. To keep it snappy AND get sources,
                    # we use a generator that yields tokens then the final dict.

                    # Container to capture results from the generator
                    res = {"full_answer": "", "sources": []}

                    def real_stream_gen():
                        try:
                            for chunk in st.session_state.chain.stream({"question": prompt}):
                                if isinstance(chunk, dict):
                                    if "answer" in chunk:
                                        res["full_answer"] += chunk["answer"]
                                        yield chunk["answer"]
                                    if "source_documents" in chunk:
                                        for doc in chunk["source_documents"]:
                                            src = doc.metadata.get("source", "unknown")
                                            res["sources"].append({
                                                "file": src,
                                                "filename": doc.metadata.get("filename", ""),
                                                "extension": doc.metadata.get("extension", ""),
                                                "snippet": doc.page_content[:300],
                                            })
                                elif isinstance(chunk, str):
                                    res["full_answer"] += chunk
                                    yield chunk
                        except Exception as e:
                            yield f"\n\n✕ Streaming Error: {e}"

                    st.write_stream(real_stream_gen())
                    answer = res["full_answer"]
                    sources = res["sources"]
            if sources:
                chips = "".join(
                    f'<span class="src-chip"><span class="dot"></span>{s["file"]}</span>'
                    for s in sources
                )
                st.markdown(f'<div class="src-chips">{chips}</div>', unsafe_allow_html=True)
                with st.expander("◈ View source snippets"):
                    for s in sources:
                        lang = s["extension"].lstrip(".") or "text"
                        st.caption(f"**{s['file']}**")
                        st.code(s["snippet"][:300], language=lang)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

        # Q&A retention
        if st.session_state.store:
            try:
                append_qa_to_store(st.session_state.store, prompt, answer, sources)
                log_conversation(prompt, answer, sources)
                st.session_state.qa_count += 1
                save_store(st.session_state.store)
            except Exception:
                pass
