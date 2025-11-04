from dotenv import load_dotenv
import os

load_dotenv()  # load các biến từ .env



from agents import (
    get_diagnosis_agent,
    get_lab_test_evaluation_agent,
    get_lab_test_result_evaluation_agent,
    get_final_diagnosis_evaluation_agent,
    get_treatment_plan_evaluation_agent,
)


def parse_lab_results(lab_results_str):
    """Chuyển chuỗi 'Tên xét nghiệm: kết quả; ...' thành dict"""
    results = {}
    parts = [p.strip() for p in lab_results_str.split(';') if p.strip()]
    for part in parts:
        if ':' in part:
            test, result = part.split(':', 1)
            results[test.strip()] = result.strip()
    return results


# ------------------ Diagnosis Agent ------------------
def run_diagnosis_agent():
    year = int(input("Nhập năm học của sinh viên (3, 4, 5): ").strip())
    symptoms = input("Nhập triệu chứng bệnh nhân: ").strip()
    user_differentials = input(
        "Nhập danh sách chẩn đoán phân biệt (cách nhau bởi dấu phẩy): "
    ).strip()

    agent = get_diagnosis_agent(year=year)
    prompt_inputs = {"symptoms": symptoms, "user_differentials": user_differentials}
    result_text = agent.invoke(prompt_inputs)

    print("\n========== KẾT QUẢ CHẨN ĐOÁN ==========\n")
    print(result_text)
    print("\n================================================\n")


# ------------------ Lab Test Evaluation Agent ------------------
def run_lab_test_evaluation_agent():
    year = int(input("Nhập năm học của sinh viên (3, 4, 5): ").strip())
    lab_tests = input("Nhập danh sách xét nghiệm đã làm (cách nhau bởi dấu phẩy): ").strip()
    diagnoses = input("Nhập danh sách chẩn đoán mục tiêu (cách nhau bởi dấu phẩy): ").strip()

    agent = get_lab_test_evaluation_agent(year=year)
    prompt_inputs = {"lab_tests": lab_tests, "diagnoses": diagnoses}
    result_text = agent.invoke(prompt_inputs)

    print("\n========== ĐÁNH GIÁ CHỈ ĐỊNH XÉT NGHIỆM ==========\n")
    print(result_text)
    print("\n================================================\n")


# ------------------ Lab Test Result Evaluation Agent ------------------
def run_lab_test_result_evaluation_agent():
    year = int(input("Nhập năm học của sinh viên (3, 4, 5): ").strip())
    lab_results_str = input(
        "Nhập kết quả xét nghiệm theo định dạng 'Tên xét nghiệm: kết quả; ...'\n"
    ).strip()
    diagnoses = input("Nhập danh sách chẩn đoán mục tiêu (cách nhau bởi dấu phẩy): ").strip()

    lab_results = parse_lab_results(lab_results_str)
    agent = get_lab_test_result_evaluation_agent(year=year)
    prompt_inputs = {"lab_test_results": lab_results, "diagnoses": diagnoses}
    result_text = agent.invoke(prompt_inputs)

    print("\n========== KẾT QUẢ ĐÁNH GIÁ XÉT NGHIỆM ==========\n")
    print(result_text)
    print("\n================================================\n")


# ------------------ Final Diagnosis Evaluation Agent ------------------
def run_final_diagnosis_agent():
    year = int(input("Nhập năm học của sinh viên (3, 4, 5): ").strip())
    symptoms = input("Nhập triệu chứng bệnh nhân: ").strip()
    lab_results_str = input(
        "Nhập kết quả xét nghiệm (nếu có) theo định dạng 'Tên xét nghiệm: kết quả; ...'\n"
    ).strip()
    user_final_diagnosis = input("Nhập chẩn đoán cuối của sinh viên: ").strip()

    lab_results = parse_lab_results(lab_results_str)
    agent = get_final_diagnosis_evaluation_agent(year=year)
    prompt_inputs = {
        "symptoms": symptoms,
        "lab_test_results": lab_results,
        "user_final_diagnoses": user_final_diagnosis,
    }
    result_text = agent.invoke(prompt_inputs)

    print("\n========== ĐÁNH GIÁ CHẨN ĐOÁN CUỐI ==========\n")
    print(result_text)
    print("\n================================================\n")


# ------------------ Treatment Plan Evaluation Agent ------------------
def run_treatment_plan_agent():
    year = int(input("Nhập năm học của sinh viên (3, 4, 5): ").strip())
    symptoms = input("Nhập triệu chứng bệnh nhân: ").strip()
    lab_results_str = input(
        "Nhập kết quả xét nghiệm (nếu có) theo định dạng 'Tên xét nghiệm: kết quả; ...'\n"
    ).strip()
    user_final_diagnosis = input("Nhập chẩn đoán cuối của sinh viên: ").strip()
    treatment_plan = input("Nhập phác đồ điều trị của sinh viên: ").strip()

    lab_results = parse_lab_results(lab_results_str)
    agent = get_treatment_plan_evaluation_agent(year=year)
    prompt_inputs = {
        "symptoms": symptoms,
        "lab_test_results": lab_results,
        "user_final_diagnoses": user_final_diagnosis,
        "treatment_plan": treatment_plan,
    }
    result_text = agent.invoke(prompt_inputs)

    print("\n========== ĐÁNH GIÁ PHÁC ĐỒ ĐIỀU TRỊ ==========\n")
    print(result_text)
    print("\n================================================\n")


# ------------------ Menu chính ------------------
if __name__ == "__main__":
    print("Chọn chế độ:")
    print("1. Chẩn đoán bệnh (diagnosis)")
    print("2. Đánh giá tiền xét nghiệm (lab test evaluation)")
    print("3. Đánh giá kết quả xét nghiệm (lab test result evaluation)")
    print("4. Đánh giá chẩn đoán cuối (final diagnosis evaluation)")
    print("5. Đánh giá phác đồ điều trị (treatment plan evaluation)")

    mode = input("Nhập số: ").strip()

    if mode == "1":
        run_diagnosis_agent()
    elif mode == "2":
        run_lab_test_evaluation_agent()
    elif mode == "3":
        run_lab_test_result_evaluation_agent()
    elif mode == "4":
        run_final_diagnosis_agent()
    elif mode == "5":
        run_treatment_plan_agent()
    else:
        print("Lựa chọn không hợp lệ.")
