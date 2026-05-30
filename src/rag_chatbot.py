"""
Partie 2 — Chatbot RAG avec interface Streamlit
Basé sur rag.py, corrigé et amélioré.

Améliorations :
  - Correction du typo dans le prompt (<quesrtion> → </question>)
  - Correction du retriever (search_kwargs au lieu de kwargs)
  - Historique de conversation multi-tours
  - Affichage des sources utilisées
  - Indicateur de progression détaillé
  - Chargement de la DB existante entre les sessions
"""

import os
import streamlit as st
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Constantes ────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_collection"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
Tu es un assistant expert qui répond aux questions en te basant UNIQUEMENT
sur le contexte fourni. Si la réponse n'est pas dans le contexte, dis
"Je ne trouve pas cette information dans les documents fournis."

<context>
{context}
</context>

Question : {question}
Réponse :
""")

# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def extract_text_from_pdfs(pdf_files) -> str:
    """Extrait tout le texte d'une liste de fichiers PDF."""
    full_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            full_text += page.extract_text() or ""
    return full_text


def build_vector_store(text: str) -> Chroma:
    """Découpe le texte, crée les embeddings et stocke dans ChromaDB."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vector_store, len(chunks)


def load_existing_store() -> Chroma | None:
    """Charge un vector store existant depuis le disque."""
    if os.path.exists(CHROMA_DIR):
        try:
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
            store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model,
                persist_directory=CHROMA_DIR,
            )
            if store._collection.count() > 0:
                return store
        except Exception:
            pass
    return None


def build_rag_chain(retriever):
    """Construit la chaîne RAG avec LCEL."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    def format_docs(docs):
        return "\n\n".join(
            f"[Source {i+1}] {doc.page_content}"
            for i, doc in enumerate(docs)
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return chain


# ── Interface Streamlit ───────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
    )

    # Initialisation de l'état de session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📂 Chargement des documents")

        # Tenter de charger une DB existante
        if st.session_state.retriever is None:
            existing = load_existing_store()
            if existing:
                count = existing._collection.count()
                st.success(f"✅ Base existante chargée ({count} chunks)")
                st.session_state.retriever = existing.as_retriever(
                    search_kwargs={"k": TOP_K}
                )
                st.session_state.doc_count = count

        pdf_docs = st.file_uploader(
            label="Charger des PDFs",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("📤 Indexer les documents", disabled=not pdf_docs):
            with st.spinner("Extraction du texte…"):
                text = extract_text_from_pdfs(pdf_docs)

            if not text.strip():
                st.error("Aucun texte extractible dans les PDFs.")
            else:
                with st.spinner("Création des embeddings et indexation…"):
                    store, n_chunks = build_vector_store(text)
                    st.session_state.retriever = store.as_retriever(
                        search_kwargs={"k": TOP_K}
                    )
                    st.session_state.doc_count = n_chunks
                    st.success(f"✅ {n_chunks} chunks indexés depuis {len(pdf_docs)} PDF(s)")

        if st.session_state.doc_count:
            st.metric("Chunks indexés", st.session_state.doc_count)

        st.divider()
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

        st.caption("Modèle : GPT-4o | Embeddings : text-embedding-3-small")

    # ── Zone principale ───────────────────────────────────────────────────────
    st.title("🤖 RAG Chatbot")
    st.caption("Posez vos questions sur les documents chargés")

    # Affichage de l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources utilisées"):
                    for i, src in enumerate(msg["sources"]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(src[:400] + "…" if len(src) > 400 else src)

    # Zone de saisie
    user_input = st.chat_input(
        "Votre question…",
        disabled=st.session_state.retriever is None,
    )

    if st.session_state.retriever is None:
        st.info("👆 Chargez des PDFs dans la barre latérale pour commencer.")

    if user_input:
        # Afficher le message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Récupérer le contexte
        ctx_docs = st.session_state.retriever.invoke(user_input)
        ctx_texts = [d.page_content for d in ctx_docs]

        # Générer la réponse en streaming
        with st.chat_message("assistant"):
            chain = build_rag_chain(st.session_state.retriever)
            response_placeholder = st.empty()
            full_response = ""

            for token in chain.stream(user_input):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            with st.expander("📄 Sources utilisées"):
                for i, src in enumerate(ctx_texts):
                    st.markdown(f"**Chunk {i+1}**")
                    st.text(src[:400] + "…" if len(src) > 400 else src)

        # Sauvegarder dans l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": ctx_texts,
        })


if __name__ == "__main__":
    main()
