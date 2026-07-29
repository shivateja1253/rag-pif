import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import wikipediaapi
import re
import unicodedata
import time

st.set_page_config(page_title="RAG-PIF", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #F8FAFC; color: #0F172A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem; max-width: 900px; }

.header { background: linear-gradient(135deg, #0F172A, #1E3A5F); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem; }
.header h1 { color: white; font-size: 1.6rem; font-weight: 700; margin: 0; }
.header p { color: #94A3B8; font-size: 0.875rem; margin: 0.4rem 0 0 0; }

.safe-chunk { background: white; border-left: 4px solid #22C55E; border: 1.5px solid #BBF7D0; border-left: 4px solid #22C55E; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; font-size: 0.875rem; line-height: 1.6; color: #0F172A; }
.blocked-chunk { background: #FFF5F5; border: 1.5px solid #FECACA; border-left: 4px solid #EF4444; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; font-size: 0.875rem; line-height: 1.6; color: #7F1D1D; }
.tag { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem; }
.tag-red { color: #EF4444; }
.tag-green { color: #16A34A; }
.tag-gray { color: #64748B; }

.answer { background: #EFF6FF; border: 1.5px solid #BFDBFE; border-radius: 12px; padding: 1.5rem; font-size: 0.9rem; color: #1E3A5F; line-height: 1.7; margin-top: 1.5rem; }

.stTextArea textarea {
    background: white !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    padding: 0.875rem 1rem !important;
}
.stTextArea textarea::placeholder {
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
}
.stTextArea textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}
.stButton > button {
    background: #0F172A !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
}
.stButton > button:hover { background: #1E3A5F !important; }
div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🛡️ RAG-PIF</h1>
    <p>Every message is scanned through the firewall before reaching the model. Try anything.</p>
</div>
""", unsafe_allow_html=True)

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
]
THRESHOLD = 0.45

def layer1_filter(text):
    normalized = unicodedata.normalize('NFKC', text).lower()
    for p in INJECTION_PATTERNS:
        m = re.search(p, normalized)
        if m:
            return True, m.group()
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
        'Computer network', 'Data science'
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    all_chunks = []
    for topic in topics:
        page = wiki.page(topic)
        if page.exists():
            all_chunks.extend(splitter.split_text(page.text[:6000]))
    return all_chunks

with st.spinner('Starting up...'):
    model, injection_index = load_models()
    base_chunks = load_knowledge_base()

if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'blocked': 0}

# Single input
user_input = st.text_area(
    "",
    placeholder="Type anything — a question, a document, a message. The firewall will scan it.",
    height=140,
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 1])
with col2:
    run = st.button("Scan & Answer →")

if run and user_input.strip():
    start = time.time()

    # Split input into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    input_chunks = splitter.split_text(user_input)

    # Also retrieve from knowledge base
    chunk_vectors = model.encode(base_chunks, normalize_embeddings=True, show_progress_bar=False).astype('float32')
    retrieval_index = faiss.IndexFlatIP(chunk_vectors.shape[1])
    retrieval_index.add(chunk_vectors)
    q_vec = model.encode([user_input], normalize_embeddings=True).astype('float32')
    D, I = retrieval_index.search(q_vec, k=5)
    retrieved = [base_chunks[i] for i in I[0] if i < len(base_chunks)]

    # Scan input chunks + retrieved chunks through firewall
    all_to_scan = input_chunks + retrieved
    safe, blocked = [], []

    for chunk in all_to_scan:
        b1, pattern = layer1_filter(chunk)
        if b1:
            blocked.append((chunk, f'Layer 1 — matched: "{pattern}"'))
            continue
        b2, score = layer2_filter(chunk, model, injection_index)
        if b2:
            blocked.append((chunk, f'Layer 2 — similarity score: {score}'))
            continue
        safe.append(chunk)

    elapsed = round((time.time() - start) * 1000)
    st.session_state.stats['total'] += 1
    st.session_state.stats['blocked'] += len(blocked)

    st.markdown("---")

    # Stats row
    c1, c2, c3 = st.columns(3)
    c1.metric("Scanned", f"{len(all_to_scan)} chunks")
    c2.metric("Blocked", f"{len(blocked)}", delta=f"{len(blocked)} threats" if blocked else "clean", delta_color="inverse")
    c3.metric("Time", f"{elapsed}ms")

    # Results
    if blocked:
        st.markdown("#### 🚫 Blocked")
        for chunk, reason in blocked:
            st.markdown(f"""
            <div class="blocked-chunk">
                <div class="tag tag-red">⛔ {reason}</div>
                <div>{chunk[:300]}{"..." if len(chunk)>300 else ""}</div>
            </div>
            """, unsafe_allow_html=True)

    if safe:
        st.markdown("#### ✅ Safe — passed to model")
        for chunk in safe[:3]:
            st.markdown(f"""
            <div class="safe-chunk">
                <div class="tag tag-green">✓ verified clean</div>
                <div>{chunk[:300]}{"..." if len(chunk)>300 else ""}</div>
            </div>
            """, unsafe_allow_html=True)

        context = " ".join(safe[:3])
        st.markdown(f"""
        <div class="answer">
            <div class="tag tag-gray" style="margin-bottom:0.75rem;">model response — based on verified chunks only</div>
            {context[:600]}{"..." if len(context)>600 else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("All content blocked. Nothing passed to the model.")
