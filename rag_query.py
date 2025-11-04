# rag_query.py
import os
from dotenv import load_dotenv
from agents import get_diagnosis_agent, get_lab_test_evaluation_agent, get_lab_test_result_evaluation_agent

load_dotenv()

def build_diagnosis_agent_rag_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db", "chroma_db")
    return get_diagnosis_agent(db_dir=db_dir)

def build_lab_test_evaluation_agent_rag_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db", "chroma_db_1")
    return get_lab_test_evaluation_agent(db_dir=db_dir)

def build_lab_test_result_evaluation_agent_rag_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db", "chroma_db_2")
    return get_lab_test_result_evaluation_agent(db_dir=db_dir)

