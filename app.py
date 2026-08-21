"""FitMeal Interactive Web Application & RAG Chatbot.

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


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
MODEL_FALLBACKS = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-flash-latest",
]

# FitMeal Menu Data Catalog
MENU_ITEMS = [
    {
        "id": "menu-1",
        "name": "ข้าวอกไก่ย่างซอสเทอริยากิ",
        "english_name": "Low-Carb Teriyaki Chicken Rice Bowl",
        "category": "🍚 ข้าว & จานหลัก (Rice Bowls)",
        "badge": "Low-Carb",
        "price": 119,
        "calories": 350,
        "protein": 40,
        "carb": 25,
        "fat": 5,
        "ingredients": "อกไก่ลอกหนังย่าง, ข้าวบุกผสมข้าวไรซ์เบอร์รี, บรอกโคลีนึ่ง, แครอต, ซอสเทอริยากิโซเดียมต่ำ",
        "icon": "🍗",
        "highlight": "อกไก่ลอกหนังย่างฉ่ำซอสเทอริยากิโซเดียมต่ำ เสิร์ฟพร้อมข้าวบุกไรซ์เบอร์รี"
    },
    {
        "id": "menu-2",
        "name": "สลัดแซลมอนย่างอะโวคาโด",
        "english_name": "Keto Grilled Salmon Avocado Salad",
        "category": "🥗 สลัด & คีโต (Salad & Keto)",
        "badge": "Keto Special",
        "price": 159,
        "calories": 420,
        "protein": 35,
        "carb": 8,
        "fat": 28,
        "ingredients": "สเต๊กแซลมอนนอร์เวย์ย่าง, อะโวคาโดสด, ผักสลัดออร์แกนิก, มะเขือเทศเชอร์รี, น้ำสลัดงาญี่ปุ่นใสแคลอรีต่ำ",
        "icon": "🥗",
        "highlight": "แซลมอนนอร์เวย์ย่างหนังกรอบ อะโวคาโดสด คาร์บต่ำเพียง 8g เหมาะสำหรับสาย Keto"
    },
    {
        "id": "menu-3",
        "name": "สเต๊กอกไก่พริกไทยดำ + ผักนึ่ง",
        "english_name": "High-Protein Black Pepper Chicken Steak",
        "category": "🥩 สเต๊กโปรตีนสูง (High-Protein)",
        "badge": "High-Protein 45g",
        "price": 129,
        "calories": 280,
        "protein": 45,
        "carb": 5,
        "fat": 4,
        "ingredients": "อกไก่หมักพริกไทยดำย่าง 200 กรัม, บรอกโคลี, ฟักทองนึ่ง, หน่อไม้ฝรั่ง",
        "icon": "🥩",
        "highlight": "อกไก่ชิ้นโต 200g อัดแน่นโปรตีน 45g ไขมันต่ำ เหมาะสำหรับสายสร้างกล้ามเนื้อ"
    },
    {
        "id": "menu-4",
        "name": "ข้าวไรซ์เบอร์รีอกไก่ผัดพริกสด",
        "english_name": "Clean Stir-Fried Chicken with Fresh Chili",
        "category": "🍚 ข้าว & จานหลัก (Rice Bowls)",
        "badge": "Clean Plate",
        "price": 109,
        "calories": 320,
        "protein": 38,
        "carb": 30,
        "fat": 4,
        "ingredients": "อกไก่ผัดพริกสดใช้น้ำมันมะกอก 1 ช้อนชา, ข้าวไรซ์เบอร์รีออร์แกนิก, แตงกวาสด",
        "icon": "🌶️",
        "highlight": "รสชาติเผ็ดร้อนกำลังดี ใช้น้ำมันมะกอกแท้ ข้าวไรซ์เบอร์รีออร์แกนิกหอมนุ่ม"
    },
    {
        "id": "menu-5",
        "name": "ควินัวผักรวมไข่ต้ม",
        "english_name": "Superfood Quinoa Mix Bowl",
        "category": "🌱 มังสวิรัติ & Vegan (Plant-Based)",
        "badge": "Vegan Option",
        "price": 99,
        "calories": 290,
        "protein": 14,
        "carb": 35,
        "fat": 8,
        "ingredients": "เมล็ดควินัวต้ม, ถั่วแระญี่ปุ่น, ข้าวโพดหวาน, แครอต, ไข่ต้ม 1 ฟอง (เลือกไม่ใส่ไข่ต้มได้)",
        "icon": "🥑",
        "highlight": "ซูเปอร์ฟู้ดควินัวอุดมด้วยไฟเบอร์และวิตามิน เลือกไม่ใส่ไข่ต้มเพื่อเป็น Vegan ได้"
    },
    {
        "id": "menu-6",
        "name": "เวย์โปรตีนไอโซเลตปั่นผลไม้รวม",
        "english_name": "Whey Isolate Berry Blast Smoothie",
        "category": "🥤 เครื่องดื่มโปรตีน (Protein Drinks)",
        "badge": "Whey Isolate 25g",
        "price": 79,
        "calories": 180,
        "protein": 25,
        "carb": 12,
        "fat": 2,
        "ingredients": "เวย์โปรตีนไอโซเลต Whey Isolate, สตรอว์เบอร์รี, บลูเบอร์รี, นมพิสตาชิโอไร้น้ำตาล",
        "icon": "🥤",
        "highlight": "เวย์โปรตีนเกรดไอโซเลตดูดซึมไว ผสมมิกซ์เบอร์รีสดและนมพิสตาชิโอไร้น้ำตาล"
    }
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
    kb_path = os.path.join(os.path.dirname(__file__), "fitmeal_kb.md")
    if not os.path.exists(kb_path):
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


def apply_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600;700&family=Kanit:wght@300;400;500;600;700&display=swap');

        /* ── Global Reset ── */
        html, body, [class*="css"] {
            font-family: 'Kanit', 'Google Sans', sans-serif;
            background-color: #ffffff;
        }

        /* ── Hide default Streamlit header padding ── */
        .stAppHeader { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* ── Main container ── */
        .stMainBlockContainer {
            max-width: 860px !important;
            margin: 0 auto !important;
            padding: 0 16px 140px 16px !important;
        }

        /* ── Gradient App Header banner ── */
        .main-header {
            background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 55%, #40916c 100%);
            padding: 24px 32px;
            border-radius: 20px;
            color: #fff;
            margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(45,106,79,0.22);
            text-align: center;
        }
        .main-header h1 {
            font-size: 2.1rem;
            font-weight: 700;
            margin: 0;
            color: #fff !important;
        }
        .main-header p {
            font-size: 1rem;
            margin-top: 6px;
            color: #d8f3dc;
            font-weight: 300;
        }
        .badge-pill {
            display: inline-block;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.28);
            color: #e8f5e9;
            padding: 4px 14px;
            border-radius: 50px;
            font-size: 0.8rem;
            margin: 3px;
            font-weight: 500;
        }

        /* ── Tabs ── */
        button[data-baseweb="tab"] {
            font-family: 'Kanit', sans-serif !important;
            font-size: 0.93rem !important;
            font-weight: 500 !important;
            color: #5f6368 !important;
            border-radius: 0 !important;
            padding: 10px 20px !important;
            border-bottom: 3px solid transparent !important;
        }
        button[aria-selected="true"] {
            color: #1b4332 !important;
            border-bottom: 3px solid #2d6a4f !important;
            font-weight: 600 !important;
            background: transparent !important;
        }

        /* ── Gemini-style Chat Messages ── */
        div[data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 4px 0 !important;
            margin-bottom: 0 !important;
            box-shadow: none !important;
        }
        /* User bubble */
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            display: flex;
            justify-content: flex-end;
        }
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
            background: #e8f5e9 !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 10px 16px !important;
            color: #1b4332 !important;
            max-width: 78% !important;
            font-size: 0.95rem !important;
        }
        /* Assistant bubble */
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 4px 18px 18px 18px !important;
            padding: 10px 16px !important;
            max-width: 86% !important;
            font-size: 0.95rem !important;
        }

        /* ── Gemini-style Floating Chat Input ── */
        /* stBottom is Streamlit's native wrapper around st.chat_input */
        div[data-testid="stBottom"] {
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            z-index: 9998 !important;
            background: rgba(255,255,255,0.94) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            padding: 14px 20px 18px !important;
            border-top: 1px solid #e8eaed !important;
        }
        div[data-testid="stChatInput"] {
            background: #f1f3ff !important;
            border: 2px solid #8ab4f8 !important;
            border-radius: 24px !important;
            box-shadow: 0 2px 14px rgba(99,102,241,0.18) !important;
            padding: 4px 8px !important;
            max-width: 820px !important;
            margin: 0 auto !important;
        }
        div[data-testid="stChatInput"]:focus-within {
            border-color: #6366f1 !important;
            box-shadow: 0 4px 24px rgba(99,102,241,0.32) !important;
        }
        div[data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: #1e1e2e !important;
            font-size: 0.95rem !important;
            font-family: 'Kanit', sans-serif !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #9aa0a6 !important;
        }
        div[data-testid="stChatInput"] button {
            color: #6366f1 !important;
        }

        /* ── Bottom padding so messages aren't hidden under fixed input ── */
        .stMainBlockContainer {
            padding-bottom: 120px !important;
        }

        /* ── Menu Cards ── */
        .menu-card {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }
        .menu-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 22px rgba(45,106,79,0.12);
            border-color: #52b788;
        }
        .menu-card-title { font-size: 1.2rem; font-weight: 600; color: #1b4332; margin-bottom: 4px; }
        .menu-card-sub   { font-size: 0.87rem; color: #718096; margin-bottom: 10px; }
        .price-tag {
            font-size: 1.25rem; font-weight: 700; color: #2d6a4f;
            background: #e8f5e9; padding: 4px 12px;
            border-radius: 10px; display: inline-block;
        }
        .macro-chip {
            display: inline-block; padding: 3px 10px;
            border-radius: 8px; font-size: 0.78rem; font-weight: 600;
            margin-right: 5px; margin-top: 7px;
        }
        .macro-cal     { background: #fff5f5; color: #c53030; }
        .macro-protein { background: #ebf8ff; color: #2b6cb0; }
        .macro-carb    { background: #feefc3; color: #b7791f; }
        .macro-fat     { background: #f0fff4; color: #276749; }
        </style>
        """,
        unsafe_allow_html=True
    )


@st.dialog("🤖 AI Chatbot - FitMeal Assistant 🥗", width="large")
def open_chatbot_dialog(prompt: str, index_tuple):
    st.markdown(f"#### 💬 สอบถามเกี่ยวกับ: **{prompt}**")
    
    dialog_key = f"dialog_msgs_{abs(hash(prompt))}"
    if dialog_key not in st.session_state:
        with st.spinner("กำลังประมวลผลคำตอบจาก AI..."):
            context, scores, r_span = retrieve_top_k(prompt, index_tuple, k=3)
            answer, g_span = generate_answer(prompt, context)
            st.session_state[dialog_key] = [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": answer,
                    "context": context,
                    "scores": scores,
                    "r_span": r_span,
                    "g_span": g_span
                }
            ]

    for msg in st.session_state[dialog_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "context" in msg:
                with st.expander("📌 Source Chunks & Trace Log (ข้อมูลอ้างอิง)"):
                    st.markdown("#### Retrieved Context Chunks:")
                    for i, (c, score) in enumerate(zip(msg["context"], msg["scores"]), 1):
                        st.markdown(f"**[{i}] Similarity Score: {score:.4f}**\n\n```text\n{c}\n```")
                    st.markdown("#### Observability Trace Spans:")
                    st.json({
                        "trace_id": f"trace-{int(time.time()*1000)}",
                        "query": prompt,
                        "spans": [msg["r_span"], msg["g_span"]]
                    })

    if follow_up := st.chat_input("พิมพ์คำถามเพิ่มเติมตรงนี้..."):
        st.session_state[dialog_key].append({"role": "user", "content": follow_up})
        with st.chat_message("user"):
            st.write(follow_up)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นหาข้อมูล..."):
                c, s, r = retrieve_top_k(follow_up, index_tuple, k=3)
                ans, g = generate_answer(follow_up, c)
            st.write(ans)
            st.session_state[dialog_key].append({
                "role": "assistant",
                "content": ans,
                "context": c,
                "scores": s,
                "r_span": r,
                "g_span": g
            })


def render_menu_modal_dialog(index_tuple):
    """Render an interactive menu section for viewing & filtering FitMeal menu items."""
    st.markdown("### 🍱 เลือกดูรายการเมนูร้าน FitMeal แยกตามหมวดหมู่")
    
    categories = [
        "ทั้งหมด (All)",
        "🍚 ข้าว & จานหลัก (Rice Bowls)",
        "🥗 สลัด & คีโต (Salad & Keto)",
        "🥩 สเต๊กโปรตีนสูง (High-Protein)",
        "🌱 มังสวิรัติ & Vegan (Plant-Based)",
        "🥤 เครื่องดื่มโปรตีน (Protein Drinks)"
    ]
    
    selected_cat = st.selectbox("🎯 เลือกกรองตามหมวดหมู่เมนู:", categories)
    
    filtered_items = MENU_ITEMS
    if selected_cat != "ทั้งหมด (All)":
        filtered_items = [item for item in MENU_ITEMS if item["category"] == selected_cat]
        
    cols = st.columns(2)
    for idx, item in enumerate(filtered_items):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="menu-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-size: 1.8rem;">{item['icon']}</span>
                            <div class="menu-card-title">{item['name']}</div>
                            <div class="menu-card-sub">{item['english_name']}</div>
                        </div>
                        <span class="price-tag">{item['price']} ฿</span>
                    </div>
                    <p style="font-size: 0.88rem; color: #4a5568; margin: 8px 0;">{item['highlight']}</p>
                    <div>
                        <span class="macro-chip macro-cal">🔥 {item['calories']} kcal</span>
                        <span class="macro-chip macro-protein">💪 โปรตีน {item['protein']}g</span>
                        <span class="macro-chip macro-carb">🌾 คาร์บ {item['carb']}g</span>
                        <span class="macro-chip macro-fat">🥑 ไขมัน {item['fat']}g</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            b1, b2 = st.columns(2)
            with b1:
                if st.button(f"💬 ถาม AI เกี่ยวกับ {item['name']}", key=f"ask-{item['id']}"):
                    open_chatbot_dialog(f"ขอข้อมูลโภชนาการ แคลอรี และส่วนผสมของ {item['name']}", index_tuple)
            with b2:
                if st.button(f"🛒 สั่งซื้อ / บันทึกขาย 1 กล่อง", key=f"order-{item['id']}"):
                    open_chatbot_dialog(f"สั่งซื้อ {item['name']} 1 กล่อง ราคา {item['price']} บาท", index_tuple)


def main():
    st.set_page_config(
        page_title="FitMeal Health & Clean Food 🥗",
        page_icon="🥗",
        layout="wide"
    )
    
    apply_custom_styles()
    
    st.markdown(
        """
        <div class="main-header">
            <h1>🥗 FitMeal Health & Clean Food</h1>
            <p>อาหารเพื่อสุขภาพ คาร์บต่ำ โปรตีนสูง ควบคุมแคลอรี ตอบโจทย์คนรักสุขภาพและคนออกกำลังกาย</p>
            <div>
                <span class="badge-pill">🔥 Calorie-Controlled</span>
                <span class="badge-pill">💪 High-Protein</span>
                <span class="badge-pill">🥑 Keto Friendly</span>
                <span class="badge-pill">🌱 Vegan Option</span>
                <span class="badge-pill">📲 LINE OA: @FitMealThailand</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        index_tuple = load_index()
    except Exception as exc:
        st.error(f"ไม่สามารถโหลดดัชนีข้อมูลได้: {exc}")
        st.stop()

    # ── Active tab tracker ──────────────────────────────────────────────────
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat"

    tab_chat, tab_menu = st.tabs([
        "💬 สอบถาม AI Chatbot (RAG Assistant)",
        "🍱 หน้าต่างเลือกเมนูอาหาร (Menu Catalog & Selector)"
    ])

    with tab_menu:
        st.session_state.active_tab = "menu"
        render_menu_modal_dialog(index_tuple)

    with tab_chat:
        st.session_state.active_tab = "chat"

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "สวัสดีครับ! ยินดีต้อนรับสู่ FitMeal 🥗\n\nผมคือ AI Assistant ของร้านอาหารเพื่อสุขภาพ **FitMeal** พร้อมช่วยตอบคำถามเกี่ยวกับ:\n- 🍽️ รายละเอียดเมนูและส่วนผสม\n- 🔥 ข้อมูลโภชนาการ แคลอรี Protein/Carb/Fat\n- 🚴 คำแนะนำสำหรับนักกีฬาและคนควบคุมน้ำหนัก\n- 🚚 การสั่งซื้อและจัดส่ง\n\nมีอะไรให้ช่วยไหมครับ?"
                }
            ]

        # Message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "context" in msg:
                    with st.expander("📌 Source Chunks & Trace Log (ข้อมูลอ้างอิง)"):
                        st.markdown("**Retrieved Context Chunks:**")
                        for i, (c, score) in enumerate(zip(msg["context"], msg["scores"]), 1):
                            st.markdown(f"**[{i}] Similarity Score: {score:.4f}**\n\n```\n{c}\n```")
                        st.markdown("**Observability Trace Spans:**")
                        st.json({
                            "trace_id": f"trace-{int(time.time()*1000)}",
                            "query": msg.get("query", ""),
                            "spans": [msg["r_span"], msg["g_span"]]
                        })

    # ── Chat input lives at page level → Streamlit auto-pins it to bottom ──
    pending_prompt = st.session_state.pop("pending_prompt", None)
    user_input = pending_prompt or st.chat_input("สอบถามเมนู แคลอรี สารอาหาร หรือสั่งซื้ออาหาร...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("FitMeal AI กำลังค้นหาข้อมูล..."):
            context, scores, r_span = retrieve_top_k(user_input, index_tuple, k=3)
            answer, g_span = generate_answer(user_input, context)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "context": context,
            "scores": scores,
            "r_span": r_span,
            "g_span": g_span,
            "query": user_input
        })
        st.rerun()


if __name__ == "__main__":
    main()
