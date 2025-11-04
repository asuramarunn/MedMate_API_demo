from langchain.prompts import PromptTemplate

# ==============================
# 🧩 HÀM CHUNG: thêm hướng dẫn theo năm học
# ==============================
def get_prompt_for_year(base_template: str, year: int) -> PromptTemplate:
    """
    Thêm hướng dẫn phù hợp với năm học vào prompt gốc.
    """
    base = base_template.strip()

    if year == 3:
        base += "\nFocus your response on basic recognition and interpretation suitable for third-year medical students. Avoid advanced management details."
    elif year == 4:
        base += "\nAdditionally, for fourth-year students:\n- Emphasize reasoning, differential diagnosis, and identification of missing or unclear information.\n- Provide concise comparisons between possible diagnoses or interpretations.\n- If dangerous or critical conditions might be missing, DO NOT name them directly; instead, propose targeted follow-up questions."
    elif year == 5:
        base += "\nFocus on advanced reasoning, management, and clinical decision-making suitable for fifth-year students. Include treatment guidance or next steps where relevant."

    return PromptTemplate.from_template(base)


# ==============================
# 📋 PROMPT TEMPLATE CHÍNH
# ==============================
diagnosis_prompt_text_template = """ 
You will receive: 
- Retrieved medical context (may be empty) 
- Patient symptoms (may be incomplete) 
- A list of differential diagnoses provided by the user 

Task: 
1) Identify missing or unclear information in the patient history/symptoms. Provide 3–5 Socratic questions to help the student clarify or supplement missing data. 
2) Review the user's differential diagnoses and give reflective feedback for each: 
  - Include comments on appropriateness and reasoning. 
  - Suggest 1–2 follow-up questions per diagnosis to prompt deeper reflection. 
3) Suggest any potentially missing critical considerations in a safe way (without naming them). Provide 2–3 open-ended questions to encourage the student to think of them. 
4) Be concise, constructive, and friendly. Use bullet points or numbered lists for clarity. 
Output: 
- Text or markdown format, easy to read by a student. 
- Include sections: 
  - "Missing information questions" 
  - "Feedback on user differentials" 
  - "Follow-up questions for critical considerations" 
- Do NOT output JSON. 

Retrieved context: 
{context} 

Symptoms: 
{symptoms} 

User differentials (comma separated): 
{user_differentials} """ 

lab_test_evaluation_prompt_text_template = """ 
You will receive: 
- Retrieved medical context (may be empty) 
- A list of lab tests performed 
- A list of target diagnoses 

Task: 
1) Evaluate whether the performed tests are sufficient, partially sufficient, or insufficient. Provide 3–5 Socratic questions under "Overall assessment". 
2) Identify helpful tests. Provide 2–4 reflective questions under "Helpful tests". 
3) Identify unnecessary tests. Provide 2–4 reflective questions under "Low-value tests". 
4) Suggest missing tests that could improve diagnostic reasoning. Provide 2–4 Socratic questions under "Missing tests". 
5) Be concise, constructive, and student-friendly. 
Use bullet points or numbered lists. 

Output: 
- Text or markdown format, easy to read. 
- Do NOT output JSON. 

Retrieved context: 
{context} 

Lab tests performed: 
{lab_tests} 

Target diagnoses (comma separated): 
{diagnoses} """ 

lab_test_result_evaluation_prompt_text_template = """ 
You will receive: 
- Retrieved medical context (may be empty) 
- A list of lab tests performed with results 
- A list of target diagnoses 

Task: 
1) Evaluate whether current results confirm, rule out, or are insufficient for each diagnosis. Provide 2–4 reflective questions per diagnosis. 
2) Highlight additional tests that could improve differentiation, if any. 
3) Identify any performed tests that are low-value or unhelpful. 
4) Be concise, constructive, and friendly. 
Use bullet points or numbered lists. 

Output: 
- Text or markdown format. 
- Do NOT output JSON. 

Retrieved context: 
{context} 

Lab test results: 
{lab_test_results} 

Target diagnoses (comma separated): 
{diagnoses} """

final_diagnosis_evaluation_prompt_text_template = """
You will receive:
- Retrieved medical context (may be empty)
- Patient symptoms
- Lab test results (if any)
- A list of candidate final diagnoses provided by the user

Task:
1) Review the final diagnoses suggested by the user.
2) Evaluate each diagnosis in terms of:
   - Appropriateness given symptoms and test results.
   - Reasoning gaps or assumptions.
   - Confidence level (high, moderate, low) with explanation.
3) Suggest Socratic follow-up questions (2–4 per diagnosis) to guide deeper reflection and critical thinking.
4) Highlight any potential critical considerations that may have been overlooked, without naming dangerous conditions directly.
5) Be concise, constructive, and student-friendly. Use bullet points or numbered lists.

Output:
- Text or markdown format, easy to read.
- Include sections:
  - "Evaluation of final diagnoses"
  - "Follow-up questions for reflection"
  - "Potential critical considerations"
- Do NOT output JSON.

Retrieved context:
{context}

Symptoms:
{symptoms}

Lab test results:
{lab_test_results}

User final diagnoses (comma separated):
{user_final_diagnoses}
"""

treatment_plan_evaluation_prompt_text_template = """
You will receive:
- Retrieved medical context (may be empty)
- Patient symptoms
- Lab test results (if any)
- The user-proposed treatment plan

Task:
1) Evaluate the treatment plan in terms of:
   - Appropriateness for the final diagnosis.
   - Completeness (does it address all major problems?).
   - Safety and potential risks (without giving explicit dangerous advice).
2) Suggest 2–4 Socratic questions to help the student reflect on optimization or alternatives.
3) Highlight missing considerations or follow-up actions in a safe, constructive way.
4) Be concise, constructive, and student-friendly. Use bullet points or numbered lists.

Output:
- Text or markdown format.
- Include sections:
  - "Treatment plan evaluation"
  - "Reflective questions"
  - "Additional considerations"
- Do NOT output JSON.

Retrieved context:
{context}

Symptoms:
{symptoms}

Lab test results:
{lab_test_results}

User proposed treatment plan:
{treatment_plan}
"""


# ==============================
# 📦 CÁC HÀM TẠO PROMPT
# ==============================
def get_diagnosis_prompt_for_year(year):
  return get_prompt_for_year(diagnosis_prompt_text_template, year)

def get_lab_test_evaluation_prompt_for_year(year):
  return get_prompt_for_year(lab_test_evaluation_prompt_text_template, year)

def get_lab_test_result_evaluation_prompt_for_year(year):
  return get_prompt_for_year(lab_test_result_evaluation_prompt_text_template, year)

def get_final_diagnosis_evaluation_prompt_for_year(year):
  return get_prompt_for_year(final_diagnosis_evaluation_prompt_text_template, year)

def get_treatment_plan_evaluation_prompt_for_year(year):
  return get_prompt_for_year(treatment_plan_evaluation_prompt_text_template, year)