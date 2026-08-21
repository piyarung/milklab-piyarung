"""FitMeal RAG Evaluation Script (Session 3 Mini Eval).

Computes Precision@3 and Recall@3 over 10 ground truth questions
and generates a similarity score histogram plot.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from app import load_index, retrieve_top_k

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


GROUND_TRUTH = [
    {
        "id": 1,
        "question": "ร้าน FitMeal เปิดกี่โมงถึงกี่โมง และอยู่ที่ไหน",
        "expected_keywords": ["เปิดให้บริการทุกวัน", "08:00 ถึง 20:00", "ถนนมิตรภาพ", "FitZone"],
        "category": "store_info"
    },
    {
        "id": 2,
        "question": "ข้าวอกไก่ย่างซอสเทอริยากิ ราคาเท่าไหร่ มีกี่แคล",
        "expected_keywords": ["ข้าวอกไก่ย่างซอสเทอริยากิ", "119 บาท", "350 kcal", "โปรตีน 40g"],
        "category": "menu_nutrition"
    },
    {
        "id": 3,
        "question": "เมนูไหนเหมาะกับคนที่ทานคีโต (Keto)",
        "expected_keywords": ["สลัดแซลมอนย่างอะโวคาโด", "สเต๊กอกไก่พริกไทยดำ", "Keto"],
        "category": "dietary"
    },
    {
        "id": 4,
        "question": "คนแพ้นมวัวหรือแลคโตสทานเวย์โปรตีนได้ไหม",
        "expected_keywords": ["Whey Isolate", "แลคโตสต่ำมาก", "ไม่มีส่วนผสมของนมวัว"],
        "category": "allergen"
    },
    {
        "id": 5,
        "question": "มีเมนูมังสวิรัติหรือ Vegan ไหม",
        "expected_keywords": ["ควินัวผักรวม", "Vegan", "ระบุไม่ใส่ไข่ต้มได้"],
        "category": "dietary"
    },
    {
        "id": 6,
        "question": "สั่งอาหารส่งฟรีไหม ค่าส่งเท่าไหร่",
        "expected_keywords": ["จัดส่งฟรีเมื่อสั่งซื้อครบ 300 บาท", "ระยะทาง 5 กม", "GrabFood"],
        "category": "delivery"
    },
    {
        "id": 7,
        "question": "มีบริการคอร์สอาหารรายสัปดาห์ Meal Prep ไหม",
        "expected_keywords": ["คอร์สอาหารรายสัปดาห์", "5 วัน", "10 มื้อ", "1,100 บาท"],
        "category": "meal_prep"
    },
    {
        "id": 8,
        "question": "สามารถขอเปลี่ยนข้าวเป็นผักนึ่งหรือบุกได้ไหม",
        "expected_keywords": ["เปลี่ยนข้าวเป็นข้าวบุก", "ผักนึ่ง", "เพิ่มเงิน 10 บาท"],
        "category": "customization"
    },
    {
        "id": 9,
        "question": "คนแพ้ถั่วทานอาหารร้าน FitMeal ได้ไหม",
        "expected_keywords": ["Nut Allergies", "นมพิสตาชิโอ", "อาหารหลักทั้งหมดปลอดถั่ว"],
        "category": "allergen"
    },
    {
        "id": 10,
        "question": "ฉลากอาหารมีบอกค่าโภชนาการแคลอรีไหม",
        "expected_keywords": ["แปะฉลากโภชนาการ", "Calorie", "Protein", "Carb", "Fat"],
        "category": "nutrition"
    }
]


def evaluate_retrieval(k: int = 3):
    index_tuple = load_index()
    chunks = index_tuple[3]
    
    results = []
    precision_list = []
    recall_list = []
    top1_scores = []

    print("=" * 65)
    print(f"FitMeal RAG Retrieval Evaluation (top_k={k})")
    print("=" * 65)

    for item in GROUND_TRUTH:
        q = item["question"]
        expected_kw = item["expected_keywords"]
        
        retrieved_chunks, scores, _ = retrieve_top_k(q, index_tuple, k=k)
        
        relevant_retrieved = 0
        for chunk in retrieved_chunks:
            is_relevant = any(kw.lower() in chunk.lower() for kw in expected_kw)
            if is_relevant:
                relevant_retrieved += 1
        
        precision = relevant_retrieved / k
        all_relevant_in_kb = sum(1 for c in chunks if any(kw.lower() in c.lower() for kw in expected_kw))
        total_relevant = max(1, all_relevant_in_kb)
        recall = min(1.0, relevant_retrieved / total_relevant)

        precision_list.append(precision)
        recall_list.append(recall)
        if scores:
            top1_scores.append(scores[0])

        eval_entry = {
            "id": item["id"],
            "question": q,
            "precision@3": round(precision, 4),
            "recall@3": round(recall, 4),
            "top1_score": round(scores[0], 4) if scores else 0.0,
            "retrieved_count": len(retrieved_chunks)
        }
        results.append(eval_entry)

        print(f"Q{item['id']:<2}: {q}")
        print(f"    Precision@{k}: {precision:.2f} | Recall@{k}: {recall:.2f} | Top-1 Score: {scores[0]:.4f}")

    mean_precision = float(np.mean(precision_list))
    mean_recall = float(np.mean(recall_list))

    print("-" * 65)
    print(f"Mean Precision@{k}: {mean_precision:.4f}")
    print(f"Mean Recall@{k}:    {mean_recall:.4f}")
    print("=" * 65)

    # Plot histogram of Top-1 similarity scores
    plt.figure(figsize=(8, 5))
    plt.hist(top1_scores, bins=6, color="#2E7D32", edgecolor="white", alpha=0.85)
    plt.title("Distribution of Top-1 Retrieval Similarity Scores (FitMeal RAG)", fontsize=13, fontweight="bold")
    plt.xlabel("Similarity Score", fontsize=11)
    plt.ylabel("Query Count", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    chart_path = os.path.join(os.path.dirname(__file__), "retrieval_similarity_hist.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved histogram plot to: {chart_path}")

    summary = {
        "mean_precision_at_k": round(mean_precision, 4),
        "mean_recall_at_k": round(mean_recall, 4),
        "total_queries": len(GROUND_TRUTH),
        "k": k,
        "results": results
    }

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    evaluate_retrieval()
