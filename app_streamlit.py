# app_streamlit.py
# Run with: streamlit run app_streamlit.py

import streamlit as st
from src.rag import ask
from src.ingest import ingest
import os

st.set_page_config(page_title="Auto-Analyst: AI Research Assistant", page_icon="📚", layout="wide")

st.title("📚 Auto-Analyst: AI Research Assistant")
st.caption("RAG system with intent routing, citations, and evaluation metrics")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build / Update Knowledge Base"):
        os.makedirs("data/pdfs", exist_ok=True)
        for f in uploaded_files or []:
            with open(os.path.join("data/pdfs", f.name), "wb") as out:
                out.write(f.read())
        ingest()
        st.success("Knowledge base updated!")

    st.divider()
    st.markdown("**Tips**")
    st.markdown("- Ask research questions for citations (e.g., *What is the main objective?*)")
    st.markdown("- Ask casual questions to chat freely (e.g., *Are you okay?*)")

# Main chat UI
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask Auto-Analyst", placeholder="What is the main objective of this paper?")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        answer = ask(question)
        st.session_state.history.append((question, answer))

for q, a in reversed(st.session_state.history):
    with st.container():
        st.markdown(f"### 🧑‍💻 You\n{q}")
        st.markdown(f"### 🤖 Auto-Analyst\n{a}")
        st.divider()