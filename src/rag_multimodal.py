"""
Partie 3 — RAG Multimodal avec interface Streamlit

Fonctionnalités :
  - Ingestion de PDFs avec extraction de texte ET d'images
  - OCR sur les images extraites (via pytesseract)
  - Description des images par GPT-4o Vision
  - Questions texte ET images (upload d'image dans le chat)
  - Réponses enrichies avec contexte visuel
"""

import os
import io
import base64
import tempfile
import streamlit as st
from pypdf import PdfReader
from PIL import Image
from pdf2image import convert_from_bytes
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from dotenv import load_dotenv
import pytesseract  # OCR sur les images embarquées

load_dotenv(override=True)

# ── Constantes ────────────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db_multimodal"
COLLECTION_NAME = "multimodal_collection"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

# ── Fonctions : extraction ────────────────────────────────────────────────────

def image_to_base64(image: Image.Image) -> str:
    """Convertit une image PIL en base64 JPEG."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def describe_image_with_gpt4v(image: Image.Image, page_num: int) -> str:
    """Envoie une image à GPT-4o Vision et retourne la description."""
    client = OpenAI()
    b64 = image_to_base64(image)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Décris précisément le contenu de cette image extraite de la page {page_num} "
                            "d'un document PDF. Inclus les graphiques, tableaux, schémas, textes visibles "
                            "et toute information pertinente. Sois exhaustif."
                        ),
                    },
                ],
            }
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content


def extract_embedded_images_ocr(pdf_bytes: bytes, pdf_name: str) -> list:
    """Extrait les images embarquées dans le PDF et retourne leur texte OCR."""
    items = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages):
        for img_obj in page.images:
            try:
                image = Image.open(io.BytesIO(img_obj.data))
                ocr_text = pytesseract.image_to_string(image, lang="fra+eng").strip()
                if ocr_text:
                    items.append({
                        "type": "ocr_text",
                        "content": f"[OCR image] {ocr_text}",
                        "source": pdf_name,
                        "page": i + 1,
                    })
            except Exception:
                pass
    return items


def extract_multimodal_content(pdf_bytes: bytes, pdf_name: str, describe_images: bool = True):
    """
    Extrait texte + OCR des images embarquées + descriptions visuelles d'un PDF.
    Retourne une liste de dicts : {type, content, source, page}.
    """
    items = []

    # ── Texte ──────────────────────────────────────────────────────────────
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            items.append({
                "type": "text",
                "content": text,
                "source": pdf_name,
                "page": i + 1,
            })

    # ── OCR sur les images embarquées ──────────────────────────────────────
    items.extend(extract_embedded_images_ocr(pdf_bytes, pdf_name))

    # ── Images (rasterisation des pages) ──────────────────────────────────
    if describe_images:
        try:
            pages_as_images = convert_from_bytes(pdf_bytes, dpi=150)
        except Exception as e:
            err_msg = str(e)
            st.warning(
                f"⚠️ Impossible de rasteriser les pages PDF ({pdf_name}) : {err_msg}\n"
                "Installez Poppler et ajoutez-le au PATH pour activer la description visuelle."
            )
            return items
        for page_num, page_img in enumerate(pages_as_images, start=1):
            page_img.thumbnail((1200, 1600))
            description = describe_image_with_gpt4v(page_img, page_num)
            items.append({
                "type": "image_description",
                "content": f"[Page {page_num} — Contenu visuel] {description}",
                "source": pdf_name,
                "page": page_num,
            })

    return items


def build_multimodal_vector_store(all_items: list) -> tuple:
    """Crée l'index ChromaDB à partir des items texte + image."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    texts, metadatas = [], []
    for item in all_items:
        chunks = splitter.split_text(item["content"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "type": item["type"],
                "source": item["source"],
                "page": item["page"],
            })

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return store, len(texts)


# ── Fonctions : génération ────────────────────────────────────────────────────

def answer_text_question(question: str, retriever) -> tuple[str, list]:
    """Répond à une question textuelle via la chaîne RAG."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    prompt = ChatPromptTemplate.from_template("""
Tu es un assistant expert. Réponds à la question en te basant sur le contexte
fourni, qui peut inclure du texte brut et des descriptions visuelles de pages PDF.
Si la réponse est absente du contexte, dis "Je ne trouve pas cette information."

<context>
{context}
</context>

Question : {question}
Réponse :
""")

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('type','texte').upper()} | {doc.metadata.get('source','')} p.{doc.metadata.get('page','')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    ctx_docs = retriever.invoke(question)
    sources = [
        {
            "type": d.metadata.get("type", "text"),
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", ""),
            "content": d.page_content,
        }
        for d in ctx_docs
    ]
    return chain, sources


def answer_with_image(question: str, image: Image.Image, retriever) -> str:
    """Combine une image uploadée + contexte RAG pour répondre."""
    # Récupérer le contexte textuel
    ctx_docs = retriever.invoke(question)
    ctx_text = "\n\n".join(d.page_content for d in ctx_docs)

    # Appel GPT-4o avec image + contexte
    client = OpenAI()
    b64 = image_to_base64(image)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Réponds à la question en te basant sur :
1. L'image ci-jointe
2. Le contexte documentaire suivant :

<context>
{ctx_text}
</context>

Question : {question}
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


# ── Interface Streamlit ───────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="RAG Multimodal",
        page_icon="🖼️",
        layout="wide",
    )

    # État de session
    for key, default in [
        ("messages", []),
        ("retriever", None),
        ("doc_count", 0),
        ("processing", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🖼️ RAG Multimodal")
        st.caption("Texte + Contenu Visuel")

        pdf_docs = st.file_uploader(
            "Charger des PDFs",
            accept_multiple_files=True,
            type=["pdf"],
        )

        describe_images = st.toggle(
            "🔍 Décrire les images avec GPT-4o Vision",
            value=True,
            help="Active la description visuelle de chaque page. Plus lent mais plus précis.",
        )

        if st.button(
            "📤 Indexer (texte + images)",
            disabled=not pdf_docs or st.session_state.processing,
        ):
            st.session_state.processing = True
            all_items = []
            progress = st.progress(0)
            status = st.empty()

            for idx, pdf_file in enumerate(pdf_docs):
                status.write(f"⏳ Traitement : {pdf_file.name}…")
                pdf_bytes = pdf_file.read()
                items = extract_multimodal_content(
                    pdf_bytes, pdf_file.name, describe_images
                )
                all_items.extend(items)
                progress.progress((idx + 1) / len(pdf_docs))

            status.write("🔢 Création des embeddings…")
            store, n_chunks = build_multimodal_vector_store(all_items)
            st.session_state.retriever = store.as_retriever(
                search_kwargs={"k": TOP_K}
            )
            st.session_state.doc_count = n_chunks
            st.session_state.processing = False
            progress.empty()
            status.empty()
            st.success(f"✅ {n_chunks} chunks indexés depuis {len(pdf_docs)} PDF(s)")

        if st.session_state.doc_count:
            st.metric("Chunks indexés", st.session_state.doc_count)

        st.divider()
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

        st.caption("Modèle : GPT-4o + Vision | Embeddings : text-embedding-3-small")

    # ── Zone principale ───────────────────────────────────────────────────────
    st.title("🖼️ RAG Multimodal")
    st.caption("Interrogez vos PDFs — texte ET contenu visuel")

    # Historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image"):
                st.image(msg["image"], width=300)
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source(s) utilisée(s)"):
                    for src in msg["sources"]:
                        icon = "🖼️" if src["type"] == "image_description" else ("🔍" if src["type"] == "ocr_text" else "📝")
                        st.markdown(
                            f"{icon} **{src['source']}** — page {src['page']}"
                        )
                        st.text(src["content"][:300] + "…")

    # Upload d'image dans le chat
    uploaded_image = st.file_uploader(
        "📎 Joindre une image à votre question (optionnel)",
        type=["png", "jpg", "jpeg", "webp"],
        key="chat_image",
    )

    if st.session_state.retriever is None:
        st.info("👆 Chargez des PDFs dans la barre latérale pour commencer.")

    user_input = st.chat_input(
        "Votre question…",
        disabled=st.session_state.retriever is None,
    )

    if user_input:
        pil_image = Image.open(uploaded_image) if uploaded_image else None

        # Message utilisateur
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "image": pil_image,
        })
        with st.chat_message("user"):
            if pil_image:
                st.image(pil_image, width=300)
            st.write(user_input)

        # Réponse
        with st.chat_message("assistant"):
            if pil_image:
                # Mode multimodal : image + RAG
                with st.spinner("Analyse de l'image et des documents…"):
                    response = answer_with_image(
                        user_input, pil_image, st.session_state.retriever
                    )
                st.write(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })
            else:
                # Mode texte seul avec RAG
                chain, sources = answer_text_question(
                    user_input, st.session_state.retriever
                )
                placeholder = st.empty()
                full_response = ""
                for token in chain.stream(user_input):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                with st.expander(f"📄 {len(sources)} source(s) utilisée(s)"):
                    for src in sources:
                        icon = "🖼️" if src["type"] == "image_description" else ("🔍" if src["type"] == "ocr_text" else "📝")
                        st.markdown(f"{icon} **{src['source']}** — page {src['page']}")
                        st.text(src["content"][:300] + "…")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })


if __name__ == "__main__":
    main()
