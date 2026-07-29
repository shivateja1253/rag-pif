import streamlit as st
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter
import wikipediaapi
import re
import unicodedata
import time
import json
import os
import PyPDF2
import io
import requests as req

st.set_page_config(page_title="Chatbot · RAG-PIF", page_icon="💬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #F8FAFC; color: #0F172A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem; max-width: 750px; }
.chat-header { background: linear-gradient(135deg, #0F172A, #1E3A5F); border-radius: 14px; padding: 1.25rem 1.75rem; margin-bottom: 1.5rem; }
.chat-header h1 { color: white; font-size: 1.2rem; font-weight: 700; margin: 0; }
.chat-header p { color: #94A3B8; font-size: 0.8rem; margin: 0.25rem 0 0 0; }
.online-dot { width: 8px; height: 8px; background: #22C55E; border-radius: 50%; display: inline-block; margin-right: 6px; }
.msg-user { background: #1D4ED8; color: white; border-radius: 18px 18px 4px 18px; padding: 0.875rem 1.25rem; margin: 0.5rem 0 0.5rem 3rem; font-size: 0.9rem; line-height: 1.5; }
.msg-bot { background: white; border: 1.5px solid #E2E8F0; color: #0F172A; border-radius: 18px 18px 18px 4px; padding: 0.875rem 1.25rem; margin: 0.5rem 3rem 0.5rem 0; font-size: 0.9rem; line-height: 1.5; }
.msg-blocked { background: #FFF5F5; border: 1.5px solid #FECACA; border-left: 3px solid #EF4444; border-radius: 18px 18px 18px 4px; padding: 0.875rem 1.25rem; margin: 0.5rem 3rem 0.5rem 0; font-size: 0.9rem; color: #991B1B; }
.typing { background: white; border: 1.5px solid #E2E8F0; border-radius: 18px 18px 18px 4px; padding: 0.875rem 1.25rem; margin: 0.5rem 3rem 0.5rem 0; display: inline-block; }
.typing span { display: inline-block; width: 8px; height: 8px; background: #94A3B8; border-radius: 50%; margin: 0 2px; animation: bounce 1.2s infinite; }
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }
.stTextInput input { background: white !important; color: #0F172A !important; -webkit-text-fill-color: #0F172A !important; border: 1.5px solid #E2E8F0 !important; border-radius: 12px !important; font-size: 0.95rem !important; padding: 0.75rem 1rem !important; }
.stTextInput input:focus { border-color: #3B82F6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important; }
.stTextInput input::placeholder { color: #94A3B8 !important; -webkit-text-fill-color: #94A3B8 !important; }
.stButton > button { background: #0F172A !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; font-size: 0.9rem !important; }
.stButton > button:hover { background: #1E3A5F !important; }
section[data-testid="stSidebar"] { background: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
    <h1>💬 RAG-PIF Chatbot</h1>
    <p><span class="online-dot"></span>Protected by 3-layer firewall · Knowledge base active</p>
</div>
""", unsafe_allow_html=True)

BASE = '/content/drive/MyDrive/rag-pif'
LOG_PATH = f'{BASE}/data/security_log.json'

# ---------------------------------------------------------------------
# LAYER 1 — regex + context-aware suppression (v3)
# Base pattern list kept exactly as this demo already had it (includes
# extras beyond src/layer1_filter.py -- base64 markers, roleplay framing
# -- that were already tuned here separately). Context-suppression logic
# layered on top, same as the benchmarked layer1_filter_v3.
# ---------------------------------------------------------------------
INJECTION_PATTERNS = [
    r'ignore\s+(\w+\s+)*(instructions?|prompt|context|rules?|guidelines?)',
    r'forget\s+(everything|all|previous|your\s+instructions)',
    r'disregard\s+(the\s+|all\s+|previous\s+)?(above|instructions?|context)',
    r'you\s+are\s+now\s+(?!an?\s+assistant)',
    r'act\s+as\s+(?!an?\s+assistant)',
    r'pretend\s+(you\s+are|to\s+be)',
    r'(DAN|jailbreak|unrestricted)\s+mode',
    r'no\s+(restrictions?|filters?|limits?)',
    r'reveal\s+(your\s+)?(system\s+prompt|instructions|guidelines)',
    r'this\s+document\s+supersedes',
    r'new\s+instruction\s*:',
    r'(updated\s+policy|system\s+notice)\s*:',
    r'your\s+(new\s+|true\s+|real\s+)?(role|persona|identity)\s+is',
    r'(end\s+of\s+(task|instructions?|prompt)|\[done\]|\[end\]).*new\s+(task|instruction)',
    r'(-{3,}|#{3,}|={3,})\s*(system|instruction|prompt|override)',
    r'(repeat|print|show|display|output|list|reveal)\s+(all|every|the)\s+(text|instructions?|prompt|content|above|below)',
    r"(let\'s\s+play|imagine|suppose|hypothetically|roleplay).{0,30}(no\s+rules|unrestricted|no\s+limits|anything)",
    r'if\s+.{0,30}(reveal|ignore|forget|bypass|override)',
    r'aWdub3Jl|aWdub3JlIGFsbA|cHJldGVuZA|Zm9yZ2V0',
]
ACT_AS_PATTERN = r'act\s+as\s+(?!an?\s+assistant)'
NO_RESTRICTIONS_PATTERN = r'no\s+(restrictions?|filters?|limits?)'
GENERIC_PATTERNS = {ACT_AS_PATTERN, NO_RESTRICTIONS_PATTERN}

RESEARCH_PREFIXES = [
    r'(researchers?|scientists?|academics?|authors?)\s+(found|showed|discovered|studied|documented|analyzed|reported|noted|observed|described)\s+that\s+',
    r'(attackers?|hackers?|adversaries?|users?|people)\s+(use[sd]?|using|say[s]?|saying|type[sd]?|typing|write[s]?|writing|send[s]?|sending|try(ing|s)?|attempt(s|ing)?|employ(s|ing)?|exploit(s|ing)?)\s+',
    r'(example|e\.g\.|such as|including|like|called|known as|referred to as)\s*[:\-]?\s*',
    r'(the\s+)?(phrase|patterns?|technique|method|attack|prompt|command|string|text|keyword)\s+',
    r'(this|the)\s+(paper|study|research|work|course|tutorial|book|article|report|chapter|section)\s+(explains?|describes?|analyzes?|covers?|discusses?|presents?|shows?|examines?)\s+',
    r'(detect|detecting|detection of|blocking|blocked|prevents?|resists?|defend|defense against|guards? against)\s+',
    r'(test|testing|tested|evaluate|evaluating|benchmark)\s+',
    r'(warning|warns?|alert|caution|note|notice):\s*',
    r'(vulnerability|exploit|flaw|weakness|bug)\s+(report|found|discovered|involves?)\s+',
    r'(labeled?|annotated?|classified|categorized)\s+as\s+',
    r'(cannot|can\'t|refused?|rejected?|resisted?|blocked?)\s+(to\s+)?(comply|follow|accept|process)\s+',
    r'(in\s+)?(the\s+)?(novel|movie|film|short\s+story|story|fiction|game|book|comic\s+book|comic|screenplay|plot|fantasy|satire|thriller|animation|dystopian|sci-?fi|science\s+fiction|graphic\s+novel|cartoon)[\s:]+',
    r'(the\s+)?(classifier|model|system|firewall|filter|detector)\s+(was\s+)?(trained?|designed?|built?|created?)\s+',
    r'(contains?|includes?|has|with)\s+(examples?|samples?|instances?|cases?)\s+of\s+',
    r'(how|why|what|when|where)\s+(do|does|did|can|could|would|should|are|is)\s+',
    r'(our|the|this)\s+(dataset|corpus|benchmark|evaluation|test\s+set)\s+',
    r'(research|stud(y|ies)|analysis)\s+(on|into|of|about)\s+',
    r'(news|breaking(\s+news)?|report|investigation|analysis|feature|industry\s+report|tech\s+report|journalist\s+investigation|case\s+study)\s*[:\-]?\s*',
]
RESEARCH_SUFFIXES = [
    r'\s+(is|are|was|were)\s+(a|an|the)?\s*(common|known|typical|classic|standard|well-known|popular)\s+(attack|technique|method|pattern|prompt|example)',
    r'\s+(attack|technique|method|pattern|prompt|example|attempt)',
    r'\s+in\s+(the|our|this)\s+(paper|study|dataset|benchmark|literature|research)',
    r'\s+to\s+(test|evaluate|detect|identify|classify|recognize)',
    r'\s+style\s+(attack|prompt|injection|jailbreak)',
]
RESEARCH_SIGNALS = [
    r'\b(paper|study|research|tutorial|course|benchmark|dataset|classifier|detector|firewall)\b',
    r'\b(researchers?|academics?|scientists?)\b',
    r'\b(evaluated?|analyzed?|detected?|classified?|trained?)\b',
    r'\b(false\s+positive|true\s+positive|precision|recall|f1|accuracy)\b',
]
STRONG_SIGNALS = [
    r'\b(OWASP|CVE|vulnerability|penetration\s+test(ing)?|pen\s+test|red\s+team(ing)?|security\s+research(ers)?|honeypot)\b',
]
AI_CONTEXT_KEYWORDS = [
    r'\bai\b', r'\ba\.i\.\b', r'\bassistant\b', r'\bchatbot\b', r'\bbot\b',
    r'\bmodel\b', r'\bllm\b', r'\bgpt\b', r'\bprompt\b', r'\binstructions?\b',
    r'\bjailbreak\b', r'\bguidelines?\b', r'\bsystem\b', r'\bchat\b', r'\boverride\b',
]
BENIGN_ROLE_SUFFIX = r'^\s*(my|your|our|a|an|the)\s+(assistant|helper|guide|tour\s+guide|tutor|coach|translator|planner|advisor|consultant)\b'
QUOTE_CHARS = ['"', "'", '\u201c', '\u201d', '\u00ab', '\u00bb', '`']

L2_THRESHOLD = 0.50
L3_THRESHOLD = 0.50

def is_research_context(text, match_start, match_end, matched_pattern):
    prefix_window = text[max(0, match_start - 200):match_start].lower()
    suffix_window = text[match_end:min(len(text), match_end + 100)].lower()
    full_text_lower = text.lower()
    before_match = text[:match_start]

    for q in QUOTE_CHARS:
        if before_match.count(q) % 2 == 1:
            return True
    for prefix in RESEARCH_PREFIXES:
        if re.search(prefix, prefix_window):
            return True
    for suffix in RESEARCH_SUFFIXES:
        if re.search(suffix, suffix_window):
            return True
    if matched_pattern == ACT_AS_PATTERN and re.search(BENIGN_ROLE_SUFFIX, suffix_window):
        return True
    for sig in STRONG_SIGNALS:
        if re.search(sig, full_text_lower):
            return True
    signal_count = sum(1 for sig in RESEARCH_SIGNALS if re.search(sig, full_text_lower))
    if signal_count >= 2:
        return True
    if matched_pattern in GENERIC_PATTERNS:
        if not any(re.search(kw, full_text_lower) for kw in AI_CONTEXT_KEYWORDS):
            return True
    return False

def layer1_filter(text):
    normalized = unicodedata.normalize('NFKC', text).lower()
    for p in INJECTION_PATTERNS:
        m = re.search(p, normalized)
        if m:
            if is_research_context(text, m.start(), m.end(), p):
                continue
            return True, m.group()
    return False, None

def layer2_filter(text, model, injection_index):
    vec = model.encode([text], normalize_embeddings=True).astype('float32')
    D, _ = injection_index.search(vec, k=1)
    score = float(D[0][0])
    return score > L2_THRESHOLD, round(score, 4)

@torch.no_grad()
def layer3_filter(text, l3_model, l3_tokenizer, device):
    inputs = l3_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    logits = l3_model(**inputs).logits
    prob = float(torch.softmax(logits, dim=-1)[0][1])
    return prob > L3_THRESHOLD, round(prob, 4)

def save_log(entry):
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            try: log = json.load(f)
            except: log = []
    log.append(entry)
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f)

@st.cache_resource(show_spinner=False)
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    injection_index = faiss.read_index(f'{BASE}/data/injection_index.faiss')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    l3_tokenizer = RobertaTokenizerFast.from_pretrained(f'{BASE}/models/roberta_tokenizer_v2')
    l3_model = RobertaForSequenceClassification.from_pretrained(f'{BASE}/models/roberta_v2_hf')
    l3_model.to(device)
    l3_model.eval()

    return model, injection_index, l3_model, l3_tokenizer, device

@st.cache_data(show_spinner=False)
def load_knowledge_base():
    wiki = wikipediaapi.Wikipedia(language='en', user_agent='RAG-PIF/1.0')
    topics = [
        'Cybersecurity', 'Network security', 'Cryptography',
        'Firewall (computing)', 'Malware', 'Artificial intelligence',
        'Machine learning', 'Python (programming language)',
        'Computer network', 'Data science', 'Prompt injection',
        'Information security', 'Computer virus', 'Phishing',
        'SQL injection', 'Cross-site scripting', 'Ransomware',
        'Intrusion detection system', 'Virtual private network'
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    all_chunks = []
    for topic in topics:
        page = wiki.page(topic)
        if page.exists():
            all_chunks.extend(splitter.split_text(page.text[:6000]))
    return all_chunks

with st.spinner('Loading 3-layer firewall...'):
    model, injection_index, l3_model, l3_tokenizer, device = load_models()
    base_chunks = load_knowledge_base()

@st.cache_resource(show_spinner=False)
def build_retrieval_index(_model, chunks_tuple):
    chunks = list(chunks_tuple)
    vectors = _model.encode(chunks, normalize_embeddings=True, show_progress_bar=False).astype('float32')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

retrieval_index = build_retrieval_index(model, tuple(base_chunks))


def run_firewall(text):
    """Runs the full L1 -> L2 -> L3 cascade. Returns (blocked, layer, reason)."""
    b1, pattern = layer1_filter(text)
    if b1:
        return True, 1, f'Pattern: "{pattern}"'
    b2, score = layer2_filter(text, model, injection_index)
    if b2:
        return True, 2, f'Similarity score: {score}'
    b3, conf = layer3_filter(text, l3_model, l3_tokenizer, device)
    if b3:
        return True, 3, f'RoBERTa confidence: {conf}'
    return False, None, None


# Initialize firewall state
if 'firewall_on' not in st.session_state:
    st.session_state.firewall_on = True

# Firewall toggle
col_a, col_b = st.columns([3, 1])
with col_b:
    if st.session_state.firewall_on:
        if st.button("🔴 Disable Firewall", use_container_width=True):
            st.session_state.firewall_on = False
            st.rerun()
    else:
        if st.button("🟢 Enable Firewall", use_container_width=True):
            st.session_state.firewall_on = True
            st.rerun()

if not st.session_state.firewall_on:
    st.warning("⚠️ Firewall is OFF — all content passes through unfiltered. For demo purposes only.")

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_chunks' not in st.session_state:
    st.session_state.pdf_chunks = []
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
if 'pending_input' not in st.session_state:
    st.session_state.pending_input = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Display chat history
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    elif msg['role'] == 'blocked':
        st.markdown('<div class="msg-blocked">🛡️ This request was blocked by the firewall.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)

# Input — use key to force clear
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "",
        placeholder="Ask anything...",
        label_visibility="collapsed",
        key=f"input_{st.session_state.input_key}"
    )
with col2:
    send = st.button("Send", use_container_width=True)

if send and user_input.strip():
    query = user_input.strip()
    st.session_state.pending_input = query
    st.session_state.input_key += 1
    st.rerun()

# Process pending input
if st.session_state.pending_input:
    query = st.session_state.pending_input
    st.session_state.pending_input = None
    st.session_state.messages.append({'role': 'user', 'content': query})

    typing_placeholder = st.empty()
    typing_placeholder.markdown('<div class="typing"><span></span><span></span><span></span></div>', unsafe_allow_html=True)

    start = time.time()
    if st.session_state.firewall_on:
        blocked, layer, reason = run_firewall(query)
    else:
        blocked, layer, reason = False, None, None

    typing_placeholder.empty()

    if blocked:
        elapsed = round((time.time() - start) * 1000)
        type_map = {1: 'direct', 2: 'semantic', 3: 'semantic'}
        st.session_state.messages.append({'role': 'blocked', 'content': ''})
        save_log({'query': query, 'blocked': True, 'layer': layer, 'reason': reason,
                  'time_ms': elapsed, 'type': type_map.get(layer, 'semantic')})
    else:
        all_chunks = base_chunks + st.session_state.pdf_chunks
        if st.session_state.pdf_chunks:
            all_vectors = model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=False).astype('float32')
            temp_index = faiss.IndexFlatIP(all_vectors.shape[1])
            temp_index.add(all_vectors)
            search_index = temp_index
        else:
            search_index = retrieval_index

        q_vec = model.encode([query], normalize_embeddings=True).astype('float32')
        D, I = search_index.search(q_vec, k=6)
        retrieved = [all_chunks[i] for i in I[0] if i < len(all_chunks)]

        safe_chunks = []
        for chunk in retrieved:
            c_blocked, c_layer, c_reason = run_firewall(chunk)
            if c_blocked:
                save_log({'query': f'[Retrieved] {chunk[:60]}...', 'blocked': True, 'layer': c_layer,
                          'reason': c_reason, 'time_ms': 0, 'type': 'indirect'})
                continue
            safe_chunks.append(chunk)

        elapsed = round((time.time() - start) * 1000)
        answer = ' '.join(safe_chunks[:2])[:500] if safe_chunks else "I don't have enough information to answer that."
        st.session_state.messages.append({'role': 'bot', 'content': answer})
        save_log({'query': query, 'blocked': False, 'layer': None, 'reason': None, 'time_ms': elapsed, 'type': 'safe'})

    st.rerun()

# PDF / TXT Upload
st.markdown('<hr style="border:none;border-top:1.5px solid #E2E8F0;margin:1.5rem 0;">', unsafe_allow_html=True)
st.markdown("**📄 Upload a document to knowledge base (PDF or TXT):**")
uploaded_file = st.file_uploader("", type=['pdf', 'txt'], label_visibility="collapsed", key="file_uploader")

if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
    if st.button("➕ Add to Knowledge Base", use_container_width=False):
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            doc_text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
        else:
            doc_text = uploaded_file.read().decode('utf-8')

        if doc_text.strip():
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            new_chunks = splitter.split_text(doc_text)
            clean, flagged = [], 0
            with st.spinner(f'Scanning {len(new_chunks)} chunks through 3-layer firewall...'):
                for chunk in new_chunks:
                    c_blocked, c_layer, c_reason = run_firewall(chunk)
                    if c_blocked:
                        flagged += 1
                        save_log({'query': f'[PDF] {chunk[:60]}...', 'blocked': True, 'layer': c_layer,
                                  'reason': c_reason, 'time_ms': 0, 'type': 'indirect'})
                        continue
                    clean.append(chunk)
            st.session_state.pdf_chunks.extend(clean)
            st.session_state.processed_files.add(uploaded_file.name)
            if flagged > 0:
                st.error(f"⛔ {flagged} chunk(s) with injection detected and blocked. {len(clean)} safe chunks added.")
            else:
                st.success(f"✅ Document scanned. {len(clean)} chunks added to knowledge base.")

# URL Fetch
st.markdown('<hr style="border:none;border-top:1.5px solid #E2E8F0;margin:1rem 0;">', unsafe_allow_html=True)
st.markdown("**🌐 Fetch a URL into knowledge base:**")
url_col1, url_col2 = st.columns([4, 1])
with url_col1:
    url_input = st.text_input("", placeholder="https://example.com/article", label_visibility="collapsed", key="url_input")
with url_col2:
    fetch_btn = st.button("Fetch", use_container_width=True)

if fetch_btn and url_input.strip():
    try:
        response = req.get(url_input.strip(), timeout=10)
        clean_text = re.sub(r'<[^>]+>', ' ', response.text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if len(clean_text) > 100:
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            new_chunks = splitter.split_text(clean_text[:10000])
            clean, flagged = [], 0
            with st.spinner(f'Scanning {len(new_chunks)} chunks through 3-layer firewall...'):
                for chunk in new_chunks:
                    c_blocked, c_layer, c_reason = run_firewall(chunk)
                    if c_blocked:
                        flagged += 1
                        save_log({'query': f'[URL] {chunk[:60]}...', 'blocked': True, 'layer': c_layer,
                                  'reason': c_reason, 'time_ms': 0, 'type': 'indirect'})
                        continue
                    clean.append(chunk)
            st.session_state.pdf_chunks.extend(clean)
            if flagged > 0:
                st.error(f"⛔ {flagged} chunk(s) blocked from URL. {len(clean)} safe chunks added.")
            else:
                st.success(f"✅ URL scanned. {len(clean)} chunks added to knowledge base.")
        else:
            st.warning("Could not extract enough text from that URL.")
    except Exception as e:
        st.error(f"Could not fetch URL: {str(e)}")
