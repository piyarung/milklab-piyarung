"""FitMeal RAG Chatbot (Session 3).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space
"""

import os
import re
import time
import json
import streamlit as st
from dotenv import load_dotenv
from google import genai

# Try importing FAISS & SentenceTransformers, fallback to TfidfVectorizer if needed
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS_ST = True
except ImportError:
    HAS_FAISS_ST = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np


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


def _call_gemini(prompt: str, api_key: str | None = None, max_output_tokens: int = 512, temperature: float = 0.2) -> str:
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


def _chunk_text(text: str, max_words: int = 80) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        current_chunk: list[str] = []
        current_count = 0
        for line in lines:
            line_count = len(line.split())
            if current_count + line_count > max_words and current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = [line]
                current_count = line_count
            else:
                current_chunk.append(line)
                current_count += line_count

        if current_chunk:
            chunks.append("\n".join(current_chunk).strip())

    return chunks


@st.cache_resource
def load_index():
    load_dotenv()
    kb_path = os.path.join(os.path.dirname(__file__), "menu_kb.md")
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    with open(kb_path, encoding="utf-8") as f:
        text = f.read()

    chunks = _chunk_text(text)
    if not chunks:
        raise RuntimeError("Knowledge base is empty")

    if HAS_FAISS_ST:
        model_name = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        try:
            model = SentenceTransformer(model_name)
        except Exception:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = model.encode(
            chunks, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        return ("faiss", model, index, chunks)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(chunks)
        return ("tfidf", vectorizer, tfidf_matrix, chunks)


def retrieve_top_k(query: str, model_tuple, k: int = 3) -> tuple[list[str], list[float], dict]:
    start_time = time.time()
    mode = model_tuple[0]
    chunks = model_tuple[3]
    k = min(k, len(chunks))

    results = []
    scores = []

    if mode == "faiss":
        _, model, index, _ = model_tuple
        query_embedding = model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_embedding)

        distances, indices = index.search(query_embedding, k)
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append(chunks[idx])
            scores.append(float(score))
    else:
        _, vectorizer, tfidf_matrix, _ = model_tuple
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:k]
        for idx in top_indices:
            results.append(chunks[idx])
            scores.append(float(sims[idx]))

    duration_ms = round((time.time() - start_time) * 1000, 2)
    span_meta = {
        "span": "retrieve_top_k",
        "mode": mode,
        "k": k,
        "results_count": len(results),
        "duration_ms": duration_ms
    }
    return results, scores, span_meta


def generate_answer(query: str, context_chunks: list[str]) -> tuple[str, dict]:
    start_time = time.time()
    context = "\n\n".join(context_chunks).strip()
    prompt = f"""คุณคือผู้ช่วย AI ประจำร้าน FitMeal ร้านอาหารเพื่อสุขภาพ เชี่ยวชาญเมนูคาร์บต่ำ โปรตีนสูง คุมแคลอรี

ข้อมูล Knowledge Base ของ FitMeal:
{context}

คำถามจากลูกค้า: {query}

คำแนะนำในการตอบ:
1. ตอบเป็นภาษาไทยด้วยความสุภาพ อบอุ่น และเป็นมิตรเหมือนพนักงานร้านที่ดีใจที่ลูกค้าถาม
2. **ต้องระบุราคา (บาท) เสมอ** เมื่อพูดถึงเมนูอาหารใดก็ตาม
3. อธิบายรายละเอียดเมนูให้ครบ เช่น ส่วนผสมหลัก, แคลอรี, โปรตีน, คาร์บ, ไขมัน เพื่อให้ลูกค้าตัดสินใจได้ง่าย
4. หากมีหลายเมนูที่เกี่ยวข้อง ให้แสดงเป็นรายการ พร้อมราคาและข้อมูลโภชนาการของแต่ละเมนู
5. ปิดท้ายด้วยการแนะนำให้ลูกค้าสั่งซื้อหรือสอบถามเพิ่มเติมได้เสมอ
6. ใช้ข้อมูลจาก Knowledge Base เท่านั้น หากไม่มีข้อมูล ให้ตอบว่า "ขออภัยครับ/ค่ะ ร้าน FitMeal ยังไม่มีข้อมูลเรื่องนี้ในขณะนี้ สามารถสอบถามเพิ่มเติมทาง LINE OA (@FitMealThailand) ได้ครับ/ค่ะ\""""

    try:
        answer = _call_gemini(prompt, max_output_tokens=512, temperature=0.2)
    except Exception as exc:
        answer = f"ขออภัยครับ/ค่ะ ไม่สามารถเชื่อมต่อกับ Gemini API ได้ในขณะนี้ ({exc})"

    duration_ms = round((time.time() - start_time) * 1000, 2)
    span_meta = {
        "span": "generate_answer",
        "duration_ms": duration_ms
    }
    return answer, span_meta


def main():
    st.set_page_config(page_title="FitMeal RAG Chatbot 🥗", page_icon="🥗", layout="centered")
    
    st.title("🥗 FitMeal RAG Chatbot")
    st.caption("ผู้ช่วย AI ร้าน FitMeal - สอบถามเมนูสุขภาพ คาร์บต่ำ โปรตีนสูง คุมแคลอรี ข้อมูลแพ้อาหาร และการจัดส่ง")

    try:
        index_tuple = load_index()
    except Exception as exc:
        st.error(f"ไม่สามารถโหลดดัชนีข้อมูลได้: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "สวัสดีครับ! ยินดีต้อนรับสู่ FitMeal 🥗 ร้านอาหารเพื่อสุขภาพ คาร์บต่ำ โปรตีนสูง คุมแคลอรี มีอะไรให้ผู้ช่วย AI แนะนำเกี่ยวกับเมนู สารอาหาร หรือการจัดส่งไหมครับ?"
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("สอบถามเมนู แคลอรี สารอาหาร หรือค่าจัดส่ง..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาข้อมูลโภชนาการและเมนู..."):
                context, scores, r_span = retrieve_top_k(prompt, index_tuple, k=3)
                answer, g_span = generate_answer(prompt, context)
            
            st.write(answer)
            
            with st.expander("📌 Source Chunks & Trace Log (ข้อมูลอ้างอิง)"):
                st.markdown("#### Retrieved Context Chunks:")
                for i, (c, score) in enumerate(zip(context, scores), 1):
                    st.markdown(f"**[{i}] Similarity Score: {score:.4f}**\n\n```text\n{c}\n```")
                
                st.markdown("#### Observability Trace Spans:")
                st.json({
                    "trace_id": f"trace-{int(time.time()*1000)}",
                    "query": prompt,
                    "spans": [r_span, g_span]
                })
                
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
