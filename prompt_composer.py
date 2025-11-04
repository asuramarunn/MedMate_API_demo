# prompt_composer.py
from langchain.prompts import PromptTemplate
from prompts import (
    get_diagnosis_prompt_for_year,
    get_lab_test_evaluation_prompt_for_year,
    get_lab_test_result_evaluation_prompt_for_year,
    get_final_diagnosis_evaluation_prompt_for_year,
    get_treatment_plan_evaluation_prompt_for_year
    
)
from basic_prompt import (
    prompt_basic,
    basic_three_phase_prompt_template,
    PEARLS_prompt_template,
    LEARN_prompt_template,
    DEBRIEF_prompt_template,
    GREAT_prompt_template,
)

# =========================
# 1️⃣ Cấu hình mapping Agent
# =========================
AGENT_FUNCTIONS = {
    "diagnosis": get_diagnosis_prompt_for_year,
    "lab_test_evaluation": get_lab_test_evaluation_prompt_for_year,
    "lab_test_result_evaluation": get_lab_test_result_evaluation_prompt_for_year,
    "final_diagnosis_evaluation": get_final_diagnosis_evaluation_prompt_for_year,
    "treatment_plan_evaluation": get_treatment_plan_evaluation_prompt_for_year,
}

# =========================
# 2️⃣ Cấu hình mapping Format
# =========================
FORMAT_STYLES = {
    "basic": prompt_basic,
    "three_phase": basic_three_phase_prompt_template,
    "pearls": PEARLS_prompt_template,
    "learn": LEARN_prompt_template,
    "debrief": DEBRIEF_prompt_template,
    "great": GREAT_prompt_template,
}

# =========================
# 3️⃣ Hàm sinh prompt tổng hợp
# =========================
def get_year_instruction(year: int):
    if year == 3:
        return (
            "Focus your questions on basic symptom recognition and "
            "common diseases suitable for third-year medical students. Avoid management details."
        )
    elif year == 4:
        return (
            "Encourage differential reasoning and identify missing information. "
            "Ask clarifying questions without naming dangerous diseases."
        )
    elif year == 5:
        return (
            "Include advanced reasoning and management discussion. "
            "Provide suggestions for next diagnostic or treatment steps."
        )
    return ""