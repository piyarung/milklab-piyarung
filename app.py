"""MilkLab RAG Chatbot (S3).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space
"""

import os
import re

import faiss
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
]


def _get_gemini_client(api_key: str | None = None):
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")

    load_dotenv()
    return genai.Client(api_key=key)


def _call_gemini(prompt: str, api_key: str | None = None, max_output_tokens: int = 256, temperature: float = 0.2) -> str:
    client = _get_gemini_client(api_key)
    preferred_model = os.environ.get(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    model_candidates = [preferred_model] if preferred_model else []
    for model in MODEL_FALLBACKS:
        if model not in model_candidates:
            model_candidates.append(model)

    last_error: Exception | None = None
    for model_name in model_candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    maxOutputTokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
            if response.text:
                return response.text.strip()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Gemini request failed for all models. Last error: {last_error}"
    )


def _chunk_text(text: str, max_words: int = 60) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        sentence_parts = [p.strip() for p in re.split(
            r"(?<=[。.?!])\s+|\n", paragraph) if p.strip()]
        current_chunk: list[str] = []
        current_count = 0
        for part in sentence_parts:
            part_count = len(part.split())
            if current_count + part_count > max_words and current_chunk:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = [part]
                current_count = part_count
            else:
                current_chunk.append(part)
                current_count += part_count

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

    return chunks


@st.cache_resource
def load_index():
    load_dotenv()
    kb_path = os.path.join(os.path.dirname(__file__), "menu_kb.md")
    with open(kb_path, encoding="utf-8") as f:
        text = f.read()

    chunks = _chunk_text(text)
    if not chunks:
        raise RuntimeError("Knowledge base is empty")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        chunks, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return model, index, chunks


def retrieve_top_k(query: str, model, index, chunks: list[str], k: int = 3) -> list[str]:
    query_embedding = model.encode(
        [query], convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(query_embedding)

    k = min(k, len(chunks))
    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(chunks[idx])
    return results


def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks).strip()
    prompt = f"""ข้อมูลต่อไปนี้เป็น knowledge base ของ MilkLab° ที่ตอบได้

Context:
{context}

คำถาม: {query}

ตอบเป็นภาษาไทยจากข้อมูลด้านบนเท่านั้น ถ้าคำตอบไม่มีใน context ให้ตอบว่า "ขอโทษครับ/ค่ะ ผมไม่ทราบข้อมูลนี้"."""
    return _call_gemini(prompt, max_output_tokens=256, temperature=0.2)


def main():
    st.set_page_config(page_title="MilkLab° RAG", page_icon="🥛")
    st.title("MilkLab° RAG Chatbot")
    st.caption("ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md")

    try:
        model, index, chunks = load_index()
    except Exception as exc:
        st.error(f"ไม่สามารถโหลดดัชนีข้อมูลได้: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("ถามอะไรเกี่ยวกับ MilkLab"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นข้อมูล..."):
                context = retrieve_top_k(prompt, model, index, chunks)
                answer = generate_answer(prompt, context)
            st.write(answer)
            with st.expander("Source chunks"):
                for i, c in enumerate(context, 1):
                    st.markdown(f"**[{i}]** {c}")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
