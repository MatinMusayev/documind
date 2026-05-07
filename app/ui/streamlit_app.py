import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000"

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1a1a2e; }
    .sub-header { color: #666; margin-top: -10px; margin-bottom: 20px; }
    .source-tag { background: #e8f4f8; border-radius: 4px; padding: 2px 8px;
                  font-size: 0.8rem; color: #2980b9; margin-right: 4px; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="main-header">🧠 DocuMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI Research Assistant</div>', unsafe_allow_html=True)

    st.subheader("📁 Upload Documents")
    uploaded = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.uploaded_files:
                with st.spinner(f"Ingesting {f.name}..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/upload",
                            files={"file": (f.name, f.read(), f.type or "application/octet-stream")}
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(f"✅ {f.name}: {data['chunks']} chunks")
                            st.session_state.uploaded_files.append(f.name)
                        else:
                            st.error(f"❌ {resp.json().get('detail', 'Upload failed')}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    if st.session_state.uploaded_files:
        st.markdown("**Loaded documents:**")
        for name in st.session_state.uploaded_files:
            st.markdown(f"- 📄 `{name}`")

    use_agent = st.toggle("🤖 Agent Mode", help="Use AI agent with web search and tools")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
        st.rerun()

with col2:
    st.subheader("💬 Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(" ".join(
                    f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"]
                ), unsafe_allow_html=True)

    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(f"{API_URL}/query", json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                        "use_agent": use_agent
                    })
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(data["answer"])
                        if data.get("sources"):
                            st.markdown(" ".join(
                                f'<span class="source-tag">📄 {s}</span>' for s in data["sources"]
                            ), unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data.get("sources", [])
                        })
                    else:
                        st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Cannot connect to API. Is the server running? ({e})")
