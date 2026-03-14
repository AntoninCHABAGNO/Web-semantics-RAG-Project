import streamlit as st
from pathlib import Path
from rag_service import ask_rag

st.set_page_config(page_title="Chatbot RAG - Jeu de dames", page_icon="♟️", layout="wide")

st.title("♟️ Chatbot RAG — Jeu de dames")
st.caption("Pose une question, le bot répond avec les sources utilisées.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources utilisées"):
                for src in msg["sources"]:
                    if src["type"] == "text":
                        st.markdown(f"**{src['id']} — {src['title']}**")
                        st.markdown(f"[Ouvrir la source]({src['url']})")
                        st.caption(src["snippet"])
                    elif src["type"] == "kg":
                        st.markdown(f"**{src['id']} — Knowledge Graph**")
                        st.code(src["fact"])

question = st.chat_input("Exemple : Quelles sont les règles de prise au jeu de dames ?")

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

        with st.expander("Sources utilisées"):
            for src in result["sources"]:
                if src["type"] == "text":
                    st.markdown(f"**{src['id']} — {src['title']}**")
                    st.markdown(f"[Ouvrir la source]({src['url']})")
                    st.caption(src["snippet"])
                elif src["type"] == "kg":
                    st.markdown(f"**{src['id']} — Knowledge Graph**")
                    st.code(src["fact"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })