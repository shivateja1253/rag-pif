import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
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
    <p><span class="online-dot"></span>Protected by 2-layer firewall · Knowledge base active</p>
</div>
""", unsafe_allow_html=True)

LOG_PATH = '/content/drive/MyDrive/rag-pif/data/security_log.json'

INJECTION_PATTERNS = [
    r'ignore\s+(\w+\s+)*(instructions?|prompt|context|rules?)',
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
THRESHOLD = 0.45

def save_log(entry):
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            try: log = json.load(f)
            except: log = []
    log.append(entry)
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f)

def layer1_filter(text):
    normalized = unicodedata.normalize('NFKC', text).lower()
    for p in INJECTION_PATTERNS:
        m = re.search(p, normalized)
        if m: return True, m.group()
    return False, None

def layer2_filter(text, model, injection_index):
    vec = model.encode([text], normalize_embeddings=True).astype('float32')
    D, _ = injection_index.search(vec, k=1)
    score = float(D[0][0])
    return score > THRESHOLD, round(score, 4)

@st.cache_resource(show_spinner=False)
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    injection_index = faiss.read_index('/content/drive/MyDrive/rag-pif/data/injection_index.faiss')
    return model, injection_index

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

with st.spinner('Loading...'):
    model, injection_index = load_models()
    base_chunks = load_knowledge_base()

@st.cache_resource(show_spinner=False)
def build_retrieval_index(_model, chunks_tuple):
    chunks = list(chunks_tuple)
    vectors = _model.encode(chunks, normalize_embeddings=True, show_progress_bar=False).astype('float32')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

retrieval_index = build_retrieval_index(model, tuple(base_chunks))

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

    # Show typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown('<div class="typing"><span></span><span></span><span></span></div>', unsafe_allow_html=True)

    start = time.time()
    b1, pattern = layer1_filter(query)
    b2, score = layer2_filter(query, model, injection_index)

    typing_placeholder.empty()

    if b1 or b2:
        layer = 1 if b1 else 2
        reason = f'Pattern: "{pattern}"' if b1 else f'Similarity score: {score}'
        elapsed = round((time.time() - start) * 1000)
        st.session_state.messages.append({'role': 'blocked', 'content': ''})
        save_log({'query': query, 'blocked': True, 'layer': layer, 'reason': reason, 'time_ms': elapsed, 'type': 'direct' if b1 else 'semantic'})
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
            cb1, cp = layer1_filter(chunk)
            if cb1:
                save_log({'query': f'[Retrieved] {chunk[:60]}...', 'blocked': True, 'layer': 1, 'reason': f'Pattern: "{cp}"', 'time_ms': 0, 'type': 'indirect'})
                continue
            cb2, cs = layer2_filter(chunk, model, injection_index)
            if cb2:
                save_log({'query': f'[Retrieved] {chunk[:60]}...', 'blocked': True, 'layer': 2, 'reason': f'Similarity: {cs}', 'time_ms': 0, 'type': 'indirect'})
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
            for chunk in new_chunks:
                b1, cp = layer1_filter(chunk)
                if b1:
                    flagged += 1
                    save_log({'query': f'[PDF] {chunk[:60]}...', 'blocked': True, 'layer': 1, 'reason': f'Pattern: "{cp}"', 'time_ms': 0, 'type': 'indirect'})
                    continue
                b2, cs = layer2_filter(chunk, model, injection_index)
                if b2:
                    flagged += 1
                    save_log({'query': f'[PDF] {chunk[:60]}...', 'blocked': True, 'layer': 2, 'reason': f'Similarity: {cs}', 'time_ms': 0, 'type': 'indirect'})
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
            for chunk in new_chunks:
                b1, cp = layer1_filter(chunk)
                if b1:
                    flagged += 1
                    save_log({'query': f'[URL] {chunk[:60]}...', 'blocked': True, 'layer': 1, 'reason': f'Pattern: "{cp}"', 'time_ms': 0, 'type': 'indirect'})
                    continue
                b2, cs = layer2_filter(chunk, model, injection_index)
                if b2:
                    flagged += 1
                    save_log({'query': f'[URL] {chunk[:60]}...', 'blocked': True, 'layer': 2, 'reason': f'Similarity: {cs}', 'time_ms': 0, 'type': 'indirect'})
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
