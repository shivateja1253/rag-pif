import streamlit as st
import pandas as pd
import time
import json
import os

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

.log-row-blocked { background: #FFF5F5; border-left: 3px solid #EF4444; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; }
.log-row-safe { background: white; border-left: 3px solid #22C55E; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; }
.log-query { font-weight: 500; color: #0F172A; margin-bottom: 0.3rem; }
.log-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #64748B; }
.log-reason { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #EF4444; margin-top: 0.2rem; }

.tag { display: inline-block; font-size: 0.65rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 999px; margin-right: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; }
.tag-l1 { background: #FEF3C7; color: #92400E; }
.tag-l2 { background: #FEE2E2; color: #991B1B; }
.tag-safe { background: #DCFCE7; color: #166534; }
.tag-direct { background: #EDE9FE; color: #5B21B6; }
.tag-indirect { background: #FFE4E6; color: #9F1239; }
.tag-semantic { background: #FEF9C3; color: #854D0E; }
</style>
""", unsafe_allow_html=True)


# Admin password protection
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div style='max-width:400px;margin:5rem auto;background:white;border:1.5px solid #E2E8F0;border-radius:16px;padding:2.5rem;text-align:center;'>
        <div style='font-size:2.5rem;margin-bottom:1rem;'>🔐</div>
        <h2 style='margin:0 0 0.5rem 0;font-size:1.3rem;'>Admin Access</h2>
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

# Get log from session state
LOG_PATH = '/content/drive/MyDrive/rag-pif/data/security_log.json'
log = []
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'r') as f:
        log = json.load(f)

total = len(log)
blocked = sum(1 for l in log if l['blocked'])
safe = total - blocked
block_rate = round((blocked / total * 100) if total > 0 else 0, 1)

# Stats
st.markdown(f"""
<div class="stat-row">
    <div class="stat-box">
        <div class="stat-val">{total}</div>
        <div class="stat-lbl">Total Requests</div>
    </div>
    <div class="stat-box">
        <div class="stat-val" style="color:#22C55E">{safe}</div>
        <div class="stat-lbl">Safe Requests</div>
    </div>
    <div class="stat-box">
        <div class="stat-val" style="color:#EF4444">{blocked}</div>
        <div class="stat-lbl">Threats Blocked</div>
    </div>
    <div class="stat-box">
        <div class="stat-val" style="color:#F59E0B">{block_rate}%</div>
        <div class="stat-lbl">Block Rate</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Layer breakdown
if blocked > 0:
    l1 = sum(1 for l in log if l['blocked'] and l['layer'] == 1)
    l2 = sum(1 for l in log if l['blocked'] and l['layer'] == 2)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box" style="margin-bottom:1.5rem;">
            <div class="stat-val" style="font-size:1.5rem;color:#D97706">{l1}</div>
            <div class="stat-lbl">Blocked by Layer 1 (Regex)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box" style="margin-bottom:1.5rem;">
            <div class="stat-val" style="font-size:1.5rem;color:#DC2626">{l2}</div>
            <div class="stat-lbl">Blocked by Layer 2 (Embedding)</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Live log
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("#### 🔴 Live Request Log")

    if not log:
        st.info("No requests yet. Go to the Chatbot and start chatting.")
    else:
        for entry in reversed(log):
            if entry['blocked']:
                layer_tag = f'<span class="tag tag-l{entry["layer"]}">Layer {entry["layer"]}</span>'
                type_tag = f'<span class="tag tag-{entry["type"]}">{entry["type"]}</span>'
                st.markdown(f"""
                <div class="log-row-blocked">
                    <div class="log-query">⛔ {entry['query'][:100]}{'...' if len(entry['query'])>100 else ''}</div>
                    <div class="log-meta">{layer_tag}{type_tag} · {entry['time_ms']}ms</div>
                    <div class="log-reason">{entry['reason']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="log-safe">
                    <div class="log-query">✅ {entry['query'][:100]}{'...' if len(entry['query'])>100 else ''}</div>
                    <div class="log-meta"><span class="tag tag-safe">safe</span> · {entry['time_ms']}ms</div>
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
            df = pd.DataFrame(list(types.items()), columns=['Type', 'Count'])
            st.bar_chart(df.set_index('Type'))
        else:
            st.info("No threats detected yet.")

        st.markdown("---")
        st.markdown("#### ⬇️ Export Log")
        if log:
            df_export = pd.DataFrame(log)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="ragpif_security_log.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("No data yet.")

# Auto refresh
st.markdown("---")
if st.button("🔄 Refresh Dashboard", use_container_width=True):
    st.rerun()
