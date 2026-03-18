import streamlit as st
from pathlib import Path
from H_21_rag_service import ask_rag

st.set_page_config(page_title="Chatbot RAG - The Queen's Gambit", page_icon="♟️", layout="wide")

st.title("♟️ Chatbot RAG — The Queen's Gambit")
st.caption("Pose une question, le bot répond avec les sources utilisées.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        st.info("Aucune source disponible.")
        return

    text_sources = [s for s in sources if s["type"] == "text"]
    kg_sources = [s for s in sources if s["type"] == "kg"]

    if text_sources:
        st.markdown("### Sources texte")
        for src in text_sources:
            with st.container(border=True):
                st.markdown(f"**{src['id']} — {src['title']}**")
                meta = f"`{src.get('domain', 'unknown')}`"
                if "score" in src:
                    meta += f" · score `{src['score']}`"
                st.caption(meta)

                st.markdown(f"[Ouvrir la source]({src['url']})")

                if src.get("snippet"):
                    st.write(src["snippet"])

    if kg_sources:
        st.markdown("### Faits du graphe de connaissances")
        for src in kg_sources:
            with st.container(border=True):
                st.markdown(f"**{src['id']} — {src['title']}**")
                if "score" in src:
                    st.caption(f"score `{src['score']}`")
                st.code(src["fact"], language="text")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources utilisées", expanded=False):
                render_sources(msg["sources"])


question = st.chat_input("Exemple : Who is Beth Harmon ?")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des sources..."):
            result = ask_rag(
                question=question,
                chunks_path=Path("data/rag/chunks.jsonl"),
                index_dir=Path("data/rag/index"),
                kg_path=Path("data/expanded_kb.ttl"),
                provider="ollama",
                model_name="phi3:mini",
                temperature=0.2,
                ollama_url="http://localhost:11434",
            )

        st.markdown(result["answer"])

        with st.expander("Sources utilisées", expanded=False):
            render_sources(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })