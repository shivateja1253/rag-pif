import streamlit as st

st.set_page_config(page_title="RAG-PIF", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #F8FAFC; color: #0F172A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 3rem 2rem; max-width: 800px; }
.home-header { background: linear-gradient(135deg, #0F172A, #1E3A5F); border-radius: 16px; padding: 2.5rem; text-align: center; margin-bottom: 2rem; }
.home-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.03em; }
.home-header p { color: #94A3B8; font-size: 0.95rem; margin: 0.75rem 0 0 0; }
.card { background: white; border: 1.5px solid #E2E8F0; border-radius: 16px; padding: 2rem; margin-bottom: 1rem; }
.card h2 { font-size: 1.2rem; font-weight: 700; margin: 0 0 0.5rem 0; color: #0F172A; }
.card p { font-size: 0.875rem; color: #64748B; margin: 0; line-height: 1.5; }
.card-icon { font-size: 2rem; margin-bottom: 0.75rem; }
.stButton > button { background: #1D4ED8 !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important; padding: 0.6rem !important; }
.stButton > button:hover { background: #1E40AF !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="home-header">
    <h1>🛡️ RAG-PIF</h1>
    <p>Prompt Injection Firewall for Retrieval-Augmented Generation Systems</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">💬</div>
        <h2>AI Chatbot</h2>
        <p>Chat with the RAG-protected assistant. Injections are blocked silently before reaching the model.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Chatbot →", use_container_width=True, key="chat"):
        st.switch_page("pages/1_Chatbot.py")

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📊</div>
        <h2>Security Dashboard</h2>
        <p>Admin view. Every request, every blocked injection, confidence scores and live statistics.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Dashboard →", use_container_width=True, key="dash"):
        st.switch_page("pages/2_Dashboard.py")

st.markdown("""
<div style='text-align:center;margin-top:2rem;font-size:0.8rem;color:#94A3B8;'>
    RAG-PIF · IEEE Research Project · 2-Layer Prompt Injection Firewall
</div>
""", unsafe_allow_html=True)
