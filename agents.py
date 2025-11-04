# agents.py
from agent_base import GenericAgent
from schema import (
    diagnosis_response_schemas,
    lab_test_response_schemas,
    lab_test_result_response_schemas,
)
from prompt_composer import AGENT_FUNCTIONS, FORMAT_STYLES, get_year_instruction
import os

# Hàm chung lấy thư mục DB
def _get_db_dir(subfolder):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "db", subfolder)


# --------------------- Diagnosis Agent ---------------------
def get_diagnosis_agent(year, db_dir=None, format_style="three_phase"):
    db_dir = db_dir or _get_db_dir("chroma_db")

    func_prompt = AGENT_FUNCTIONS["diagnosis"](year)
    format_prompt = FORMAT_STYLES[format_style]
    year_instruction = get_year_instruction(year)
    system_instruction = f"{format_prompt.strip()}\n\n[Student Level Guidance]\n{year_instruction}"

    return GenericAgent(
        functional_prompt=func_prompt,
        use_rag=True,
        db_dir=db_dir,
        system_instruction=system_instruction,
        year=year
    )


# --------------------- Lab Test Evaluation Agent ---------------------
def get_lab_test_evaluation_agent(year, db_dir=None, format_style="learn"):
    db_dir = db_dir or _get_db_dir("chroma_db_1")

    func_prompt = AGENT_FUNCTIONS["lab_test_evaluation"](year)
    format_prompt = FORMAT_STYLES[format_style]
    year_instruction = get_year_instruction(year)
    system_instruction = f"{format_prompt.strip()}\n\n[Student Level Guidance]\n{year_instruction}"

    return GenericAgent(
        functional_prompt=func_prompt,
        use_rag=True,
        db_dir=db_dir,
        system_instruction=system_instruction,
        year=year
    )


# --------------------- Lab Test Result Evaluation Agent ---------------------
def get_lab_test_result_evaluation_agent(year, db_dir=None, format_style="pearls"):
    db_dir = db_dir or _get_db_dir("chroma_db_2")

    func_prompt = AGENT_FUNCTIONS["lab_test_result_evaluation"](year)
    format_prompt = FORMAT_STYLES[format_style]
    year_instruction = get_year_instruction(year)
    system_instruction = f"{format_prompt.strip()}\n\n[Student Level Guidance]\n{year_instruction}"

    return GenericAgent(
        functional_prompt=func_prompt,
        use_rag=True,
        db_dir=db_dir,
        system_instruction=system_instruction,
        year=year
    )
    
# --------------------- Final Diagnosis Evaluation Agent ---------------------
def get_final_diagnosis_evaluation_agent(year, db_dir=None, format_style="debrief"):
    db_dir = db_dir or _get_db_dir("chroma_db_3")

    func_prompt = AGENT_FUNCTIONS["final_diagnosis_evaluation"](year)
    format_prompt = FORMAT_STYLES[format_style]
    year_instruction = get_year_instruction(year)
    system_instruction = f"{format_prompt.strip()}\n\n[Student Level Guidance]\n{year_instruction}"

    return GenericAgent(
        functional_prompt=func_prompt,
        use_rag=True,
        db_dir=db_dir,
        system_instruction=system_instruction,
        year=year
    )
    
# --------------------- Treatment Plan Evaluation Agent ---------------------
def get_treatment_plan_evaluation_agent(year, db_dir=None, format_style="great"):
    db_dir = db_dir or _get_db_dir("chroma_db_4")

    func_prompt = AGENT_FUNCTIONS["treatment_plan_evaluation"](year)
    format_prompt = FORMAT_STYLES[format_style]
    year_instruction = get_year_instruction(year)
    system_instruction = f"{format_prompt.strip()}\n\n[Student Level Guidance]\n{year_instruction}"

    return GenericAgent(
        functional_prompt=func_prompt,
        use_rag=True,
        db_dir=db_dir,
        system_instruction=system_instruction,
        year=year
    )
