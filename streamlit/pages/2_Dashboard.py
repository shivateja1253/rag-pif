import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

st.set_page_config(page_title="Dashboard · RAG-PIF", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #F8FAFC; color: #0F172A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem; max-width: 1300px; }
.dash-header { background: linear-gradient(135deg, #0F172A, #1E3A5F); border-radius: 14px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; }
.dash-header h1 { color: white; font-size: 1.4rem; font-weight: 700; margin: 0; }
.dash-header p { color: #94A3B8; font-size: 0.85rem; margin: 0.25rem 0 0 0; }
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat-box { background: white; border: 1.5px solid #E2E8F0; border-radius: 12px; padding: 1.25rem 1.5rem; flex: 1; }
.stat-val { font-size: 2rem; font-weight: 700; }
.stat-lbl { font-size: 0.75rem; color: #64748B; font-weight: 500; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }
.log-blocked { background: #FFF5F5; border-left: 3px solid #EF4444; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
.log-safe { background: white; border: 1.5px solid #E2E8F0; border-left: 3px solid #22C55E; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
.log-query { font-weight: 500; color: #0F172A; margin-bottom: 0.3rem; font-size: 0.875rem; }
.log-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748B; }
.log-reason { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #EF4444; margin-top: 0.2rem; }
.log-owasp { font-size: 0.68rem; color: #6366F1; margin-top: 0.2rem; font-weight: 600; }
.tag { display: inline-block; font-size: 0.65rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 999px; margin-right: 0.3rem; text-transform: uppercase; }
.tag-l1 { background: #FEF3C7; color: #92400E; }
.tag-l2 { background: #FEE2E2; color: #991B1B; }
.tag-safe { background: #DCFCE7; color: #166534; }
.sev-critical { background: #FEE2E2; color: #991B1B; }
.sev-high { background: #FFEDD5; color: #9A3412; }
.sev-medium { background: #FEF9C3; color: #854D0E; }
</style>
""", unsafe_allow_html=True)

# OWASP + Severity mapping
OWASP_MAP = {
    'direct': 'LLM01:2025 — Prompt Injection',
    'semantic': 'LLM01:2025 — Prompt Injection',
    'indirect': 'LLM02:2025 — Insecure Output Handling',
    'safe': None
}

SEVERITY_MAP = {
    'direct': ('CRITICAL', 'sev-critical'),
    'semantic': ('HIGH', 'sev-high'),
    'indirect': ('MEDIUM', 'sev-medium'),
    'safe': ('NONE', 'tag-safe')
}

# Admin password
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div style='max-width:400px;margin:5rem auto;background:white;border:1.5px solid #E2E8F0;border-radius:16px;padding:2.5rem;text-align:center;'>
        <div style='font-size:2.5rem;margin-bottom:1rem;'>🔐</div>
        <h2 style='margin:0 0 0.5rem 0;font-size:1.3rem;color:#0F172A;'>Admin Access</h2>
        <p style='color:#64748B;font-size:0.875rem;margin-bottom:1.5rem;'>Enter the admin password to view the security dashboard.</p>
    </div>
    """, unsafe_allow_html=True)
    pwd = st.text_input("Password", type="password", placeholder="Enter admin password")
    if st.button("Login", use_container_width=False):
        if pwd == "ragpif2024":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.markdown("""
<div class="dash-header">
    <h1>📊 Security Dashboard</h1>
    <p>Live monitoring of all requests passing through RAG-PIF</p>
</div>
""", unsafe_allow_html=True)

# Load log
LOG_PATH = '/content/drive/MyDrive/rag-pif/data/security_log.json'
log = []
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'r') as f:
        try: log = json.load(f)
        except: log = []

total = len(log)
blocked = sum(1 for l in log if l['blocked'])
safe = total - blocked
block_rate = round((blocked / total * 100) if total > 0 else 0, 1)
l1 = sum(1 for l in log if l['blocked'] and l.get('layer') == 1)
l2 = sum(1 for l in log if l['blocked'] and l.get('layer') == 2)

# Stats
st.markdown(f"""
<div class="stat-row">
    <div class="stat-box"><div class="stat-val">{total}</div><div class="stat-lbl">Total Requests</div></div>
    <div class="stat-box"><div class="stat-val" style="color:#22C55E">{safe}</div><div class="stat-lbl">Safe Requests</div></div>
    <div class="stat-box"><div class="stat-val" style="color:#EF4444">{blocked}</div><div class="stat-lbl">Threats Blocked</div></div>
    <div class="stat-box"><div class="stat-val" style="color:#F59E0B">{block_rate}%</div><div class="stat-lbl">Block Rate</div></div>
    <div class="stat-box"><div class="stat-val" style="color:#D97706">{l1}</div><div class="stat-lbl">Layer 1 Blocks</div></div>
    <div class="stat-box"><div class="stat-val" style="color:#DC2626">{l2}</div><div class="stat-lbl">Layer 2 Blocks</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown("#### 🔴 Live Request Log")
    if not log:
        st.info("No requests yet. Go to the Chatbot and start chatting.")
    else:
        for entry in reversed(log[-30:]):
            ts = entry.get('timestamp', '')
            if entry['blocked']:
                sev_label, sev_class = SEVERITY_MAP.get(entry.get('type', 'direct'), ('HIGH', 'sev-high'))
                layer = entry.get('layer', '?')
                owasp = OWASP_MAP.get(entry.get('type', 'direct'), 'LLM01:2025 — Prompt Injection')
                reason = entry.get('reason', '')
                query = entry.get('query', '')[:100]
                st.markdown(f"""
                <div class="log-blocked">
                    <div class="log-query">⛔ {query}{'...' if len(entry.get('query',''))>100 else ''}</div>
                    <div class="log-meta">
                        <span class="tag tag-l{layer}">Layer {layer}</span>
                        <span class="tag {sev_class}">{sev_label}</span>
                        · {entry.get('time_ms', 0)}ms · {ts}
                    </div>
                    <div class="log-reason">{reason}</div>
                    <div class="log-owasp">⚠️ {owasp}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                query = entry.get('query', '')[:100]
                st.markdown(f"""
                <div class="log-safe">
                    <div class="log-query">✅ {query}{'...' if len(entry.get('query',''))>100 else ''}</div>
                    <div class="log-meta">
                        <span class="tag tag-safe">SAFE</span>
                        · {entry.get('time_ms', 0)}ms · {ts}
                    </div>
                </div>
                """, unsafe_allow_html=True)

with col_right:
    st.markdown("#### 📈 Threat Breakdown")
    if log:
        types = {}
        for entry in log:
            if entry['blocked']:
                t = entry.get('type', 'unknown')
                types[t] = types.get(t, 0) + 1
        if types:
            df_chart = pd.DataFrame(list(types.items()), columns=['Type', 'Count'])
            st.bar_chart(df_chart.set_index('Type'))

        st.markdown("---")
        st.markdown("#### 🔐 OWASP Coverage")
        st.markdown("""
        <div style='font-size:0.8rem;line-height:1.8;color:#475569;'>
        ✅ <b>LLM01:2025</b> — Prompt Injection<br>
        ✅ <b>LLM02:2025</b> — Insecure Output Handling<br>
        ⏳ <b>LLM06:2025</b> — Sensitive Info Disclosure
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ⬇️ Export")
        df_export = pd.DataFrame(log)
        csv = df_export.to_csv(index=False)
        st.download_button("Download Security Log CSV", data=csv, file_name="ragpif_log.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No data yet.")

st.markdown("---")
if st.button("🔄 Refresh", use_container_width=True):
    st.rerun()
