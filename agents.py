from agent_base import GenericAgent
from prompts import get_diagnosis_prompt, get_lab_test_evaluation_prompt, get_lab_test_result_evaluation_prompt
from schema import diagnosis_response_schemas, lab_test_response_schemas, lab_test_result_response_schemas
import os

def get_diagnosis_agent(db_dir=None):
    if db_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(current_dir, "db", "chroma_db")
    # prompt đã chuẩn hóa, agent chỉ cần nhận context mở rộng
    return GenericAgent(
        prompt_template=get_diagnosis_prompt(),
        response_schemas=diagnosis_response_schemas,
        use_rag=True,
        db_dir=db_dir
    )

def get_lab_test_evaluation_agent(db_dir=None):
    if db_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(current_dir, "db", "chroma_db_1")
    return GenericAgent(
        prompt_template=get_lab_test_evaluation_prompt(),
        response_schemas=lab_test_response_schemas,
        use_rag=True,
        db_dir=db_dir
    )

def get_lab_test_result_evaluation_agent(db_dir=None):
    if db_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(current_dir, "db", "chroma_db_2")
    return GenericAgent(
        prompt_template=get_lab_test_result_evaluation_prompt(),
        response_schemas=lab_test_result_response_schemas,
        use_rag=True,
        db_dir=db_dir
    )