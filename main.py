# main.py

from rag_query import build_diagnosis_agent_rag_pipeline, build_lab_test_evaluation_agent_rag_pipeline, build_lab_test_result_evaluation_agent_rag_pipeline
import json

def parse_lab_results(lab_results_str):
    """
    Chuyển chuỗi dạng:
    "CBC: Elevated WBC 15,000; CRP: 50 mg/L; Chest X-ray: infiltrate right lung"
    thành dict:
    {
      "CBC": "Elevated WBC 15,000",
      "CRP": "50 mg/L",
      "Chest X-ray": "infiltrate right lung"
    }
    """
    results = {}
    # Tách từng cặp xét nghiệm-kết quả theo dấu chấm phẩy
    parts = [p.strip() for p in lab_results_str.split(';') if p.strip()]
    for part in parts:
        if ':' in part:
            test, result = part.split(':', 1)
            results[test.strip()] = result.strip()
    return results

def ensure_list(obj):
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return []
    return obj

def run_diagnosis_agent():
    global patient_context
    # Chuẩn hóa context đầu vào
    if not patient_context.get("symptoms"):
        patient_context["symptoms"] = input("Nhập triệu chứng bệnh nhân: ").strip()
    if not patient_context.get("history"):
        patient_context["history"] = input("Nhập tiền sử bệnh (nếu có, có thể bỏ qua): ").strip()
    patient_context["user_differentials"] = input("Nhập danh sách chẩn đoán phân biệt (cách nhau bởi dấu phẩy): ").strip()
    # Các trường khác để rỗng
    patient_context["lab_orders"] = ""
    patient_context["lab_tests"] = ""
    patient_context["lab_results"] = ""
    patient_context["lab_test_results"] = ""
    patient_context["diagnoses"] = ""
    patient_context["final_diagnosis"] = ""
    patient_context["treatment_plan"] = ""
    agent = build_diagnosis_agent_rag_pipeline()
    result = agent.invoke({
        "symptoms": patient_context["symptoms"],
        "history": patient_context["history"],
        "user_differentials": patient_context["user_differentials"],
        "lab_orders": patient_context["lab_orders"],
        "lab_tests": patient_context["lab_tests"],
        "lab_results": patient_context["lab_results"],
        "lab_test_results": patient_context["lab_test_results"],
        "diagnoses": patient_context["diagnoses"],
        "final_diagnosis": patient_context["final_diagnosis"],
        "treatment_plan": patient_context["treatment_plan"]
    })
    result["top_5"] = ensure_list(result.get("top_5", []))
    result["user_evaluation"] = ensure_list(result.get("user_evaluation", []))
    result["follow_up_questions_for_missing"] = ensure_list(result.get("follow_up_questions_for_missing", []))
    result["follow_up_questions_for_top5"] = ensure_list(result.get("follow_up_questions_for_top5", []))
    if result["user_evaluation"]:
        print("\nUser differentials evaluation:")
        for ev in result["user_evaluation"]:
            print(f" - {ev.get('diagnosis', 'N/A')}: {ev.get('verdict', '')} — {ev.get('short_rationale', '')}")
    if result.get("missing_dangerous", "").lower().startswith("y"):
        print("\n⚠ Some potentially dangerous conditions may be missing from the user's list.")
        for q in result["follow_up_questions_for_missing"]:
            print("  •", q)
    if result["follow_up_questions_for_top5"]:
        print("\nFollow-up questions for other diagnoses:")
        for item in result["follow_up_questions_for_top5"]:
            print("  •", item.get("question", ""))

def run_lab_test_evaluation_agent():
    global patient_context
    # Đảm bảo context chung đã có
    if not patient_context.get("symptoms"):
        patient_context["symptoms"] = input("Nhập triệu chứng bệnh nhân: ").strip()
    if not patient_context.get("history"):
        patient_context["history"] = input("Nhập tiền sử bệnh (nếu có, có thể bỏ qua): ").strip()
    # Bổ sung các trường đặc thù cho bước này
    patient_context["lab_orders"] = input("Nhập danh sách chỉ định xét nghiệm (nếu có, có thể bỏ qua): ").strip()
    patient_context["lab_tests"] = input("Nhập danh sách xét nghiệm đã làm (cách nhau bởi dấu phẩy): ").strip()
    patient_context["diagnoses"] = input("Nhập danh sách chẩn đoán mục tiêu (cách nhau bởi dấu phẩy): ").strip()
    # Các trường khác để rỗng nếu chưa có
    if "user_differentials" not in patient_context:
        patient_context["user_differentials"] = ""
    patient_context["lab_results"] = ""
    patient_context["lab_test_results"] = ""
    patient_context["final_diagnosis"] = ""
    patient_context["treatment_plan"] = ""
    agent = build_lab_test_evaluation_agent_rag_pipeline()
    result = agent.invoke({
        "symptoms": patient_context["symptoms"],
        "history": patient_context["history"],
        "user_differentials": patient_context["user_differentials"],
        "lab_orders": patient_context["lab_orders"],
        "lab_tests": patient_context["lab_tests"],
        "lab_results": patient_context["lab_results"],
        "lab_test_results": patient_context["lab_test_results"],
        "diagnoses": patient_context["diagnoses"],
        "final_diagnosis": patient_context["final_diagnosis"],
        "treatment_plan": patient_context["treatment_plan"]
    })
    print("\nKết quả đánh giá xét nghiệm:")
    # Nếu là dạng list kết quả chi tiết từng chẩn đoán
    if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
        for i, entry in enumerate(result["results"], 1):
            print(f"--- Đánh giá {i} ---")
            print(f"Chẩn đoán: {entry.get('diagnosis', 'N/A')}")
            print(f"Kết luận: {entry.get('verdict', '')}")
            print(f"Giải thích: {entry.get('rationale', '')}")
            if entry.get("missing_tests"):
                print(f"Thiếu xét nghiệm: {', '.join(entry['missing_tests'])}")
            if entry.get("unnecessary_tests"):
                print(f"Xét nghiệm không cần thiết: {', '.join(entry['unnecessary_tests'])}")
            print()
    # Nếu là dict tổng hợp (overall)
    elif isinstance(result, dict):
        for k, v in result.items():
            print(f"{k.replace('_', ' ').capitalize()}: {v}")
    else:
        print(result)


def run_lab_test_result_evaluation_agent():
    global patient_context
    # Đảm bảo context chung đã có
    if not patient_context.get("symptoms"):
        patient_context["symptoms"] = input("Nhập triệu chứng bệnh nhân: ").strip()
    if not patient_context.get("history"):
        patient_context["history"] = input("Nhập tiền sử bệnh (nếu có, có thể bỏ qua): ").strip()
    # Bổ sung các trường đặc thù cho bước này
    lab_results_str = input(
        "Nhập kết quả xét nghiệm theo định dạng 'Tên xét nghiệm: kết quả; ...' (ví dụ: CBC: Elevated WBC 15,000; CRP: 50 mg/L):\n"
    ).strip()
    patient_context["lab_results"] = lab_results_str
    patient_context["lab_test_results"] = parse_lab_results(lab_results_str)
    patient_context["diagnoses"] = input("Nhập danh sách chẩn đoán mục tiêu (cách nhau bởi dấu phẩy): ").strip()
    # Các trường khác để rỗng nếu chưa có
    if "user_differentials" not in patient_context:
        patient_context["user_differentials"] = ""
    if "lab_orders" not in patient_context:
        patient_context["lab_orders"] = ""
    if "lab_tests" not in patient_context:
        patient_context["lab_tests"] = ""
    patient_context["final_diagnosis"] = ""
    patient_context["treatment_plan"] = ""
    agent = build_lab_test_result_evaluation_agent_rag_pipeline()
    result = agent.invoke({
        "symptoms": patient_context["symptoms"],
        "history": patient_context["history"],
        "user_differentials": patient_context["user_differentials"],
        "lab_orders": patient_context["lab_orders"],
        "lab_tests": patient_context["lab_tests"],
        "lab_results": patient_context["lab_results"],
        "lab_test_results": patient_context["lab_test_results"],
        "diagnoses": patient_context["diagnoses"],
        "final_diagnosis": patient_context["final_diagnosis"],
        "treatment_plan": patient_context["treatment_plan"]
    })
    print("\n========== KẾT QUẢ ĐÁNH GIÁ KẾT QUẢ XÉT NGHIỆM ==========")
    if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
        for i, entry in enumerate(result["results"], 1):
            print(f"\n--- ĐÁNH GIÁ {i} ---")
            print(f"Chẩn đoán:        {entry.get('diagnosis', 'N/A')}")
            print(f"Kết luận:         {entry.get('verdict', '')}")
            print(f"Giải thích:       {entry.get('rationale', '')}")
            if entry.get("missing_tests"):
                print(f"Thiếu xét nghiệm: {', '.join(entry['missing_tests']) if entry['missing_tests'] else 'Không'}")
            if entry.get("unnecessary_tests"):
                print(f"Không cần thiết:  {', '.join(entry['unnecessary_tests']) if entry['unnecessary_tests'] else 'Không'}")
    elif isinstance(result, dict):
        print("\n--- KẾT QUẢ TỔNG HỢP ---")
        for k, v in result.items():
            print(f"{k.replace('_', ' ').capitalize()}: {v}")
    else:
        print(result)

# Ví dụ input test khi chạy:
"""
Nhập kết quả xét nghiệm theo định dạng 'Tên xét nghiệm: kết quả; ...' (ví dụ: CBC: Elevated WBC 15,000; CRP: 50 mg/L):
CBC: Elevated WBC 15,000; CRP: 50 mg/L; Chest X-ray: infiltrate right lung
Nhập danh sách chẩn đoán mục tiêu (cách nhau bởi dấu phẩy): 
Community-acquired pneumonia, Acute myocardial infarction, Pulmonary embolism
"""

    

patient_context = {}

if __name__ == "__main__":
    print("Chọn chế độ:")
    print("1. Chẩn đoán bệnh (diagnosis)")
    print("2. Đánh giá tiền xét nghiệm (lab test evaluation)")
    print("3. Đánh giá kết quả xét nghiệm (lab test result evaluation)")
    print("4. Đánh giá chuẩn đoán cuối (final diagnosis evaluation)")
    print("5. Đánh giá phác đồ điều trị (treatment plan evaluation)")
    while True:
        mode = input("Nhập số : ").strip()
        if mode == "1":
            run_diagnosis_agent()
        elif mode == "2":
            run_lab_test_evaluation_agent()
        elif mode == "3":
            run_lab_test_result_evaluation_agent()
        elif mode == "4":
            print("Chức năng đánh giá chuẩn đoán cuối chưa được triển khai.")
        elif mode == "5":
            print("Chức năng đánh giá phác đồ điều trị chưa được triển khai.")
        else:
            print("Kết thúc.")
            break