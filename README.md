# 🥗 FitMeal Solopreneur Starter (Session 3: RAG Chatbot + Evaluation)

โครงการระบบ AI Chatbot ร้าน **FitMeal** (อาหารเพื่อสุขภาพ คาร์บต่ำ โปรตีนสูง ควบคุมแคลอรี)

## 📌 ภาพรวมโครงสร้างระบบ (Session 3 Deliverables)

1. **RAG Knowledge Base (`fitmeal_kb.md`)**: ข้อมูลรายการอาหารสุขภาพ, โภชนาการ (Calories/Macros), สารอาหาร, Allergen (ข้อจำกัดนม/กลูเตน/ถั่ว), Keto/Vegan, ที่ตั้ง และ FAQ การจัดส่ง
2. **Pivot Documentation (`PIVOT.md`)**: เอกสารรายละเอียดการ Pivot สำหรับร้าน FitMeal
3. **Streamlit RAG Chatbot (`app.py`)**: เว็บแอปพลิเคชันสำหรับโต้ตอบลูกค้า ค้นหาด้วย FAISS Vector Search + SentenceTransformers และประมวลผลคำตอบด้วย Gemini API
4. **Mini Evaluation (`eval.ipynb` & `eval_rag.py`)**: การประเมินผล Retrieval Layer ด้วย 10 Ground-Truth Questions (คำนวณ Precision@3, Recall@3 และแสดง Similarity Score Histogram)

## 🚀 การใช้งาน (Usage)

### 1. การรัน Streamlit App ในท้องถิ่น (Local Run)
```bash
streamlit run app.py
```

### 2. การประเมินผลระบบ RAG (Evaluation)
```bash
python eval_rag.py
```
หรือเปิดรันผ่าน Jupyter Notebook `eval.ipynb`

### 3. เครื่องมือเสริมอื่นๆ
- `caption_generator.py`: สร้างแคปชั่นโปรโมตเมนูสุขภาพ
- `sales_logger.py`: บันทึกยอดขายลง Google Sheets
- `agent_harness.py`: ระบบตีความคำสั่งภาษาไทยเรียกใช้เครื่องมืออัตโนมัติ

## 🛠️ ความต้องการระบบ (Requirements)
- Python 3.11+
- `streamlit`
- `sentence-transformers`
- `faiss-cpu`
- `google-genai`
- `matplotlib`, `numpy`
