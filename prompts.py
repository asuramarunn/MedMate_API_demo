# prompts.py
from langchain.prompts import PromptTemplate

diagnosis_prompt_template = """
You are a medical assistant.

You will receive:
- Retrieved medical context (may be empty).
- Patient symptoms (symptoms).
- Patient history (history, may be empty).
- Preliminary differential diagnoses (user_differentials).
- Lab test orders (lab_orders, optional).
- Lab test results (lab_results, optional).
- Final diagnosis (final_diagnosis, optional).
- Treatment plan (treatment_plan, optional).

Use only the fields relevant to your current task. Ignore fields that are not provided or not relevant.

Task:
1) From the retrieved context and the symptoms/history, list up to 5 most relevant differential diagnoses, ranked from most to least likely.
2) Compare those top-5 with the user's differential list.
   For each user-provided diagnosis, give a short evaluation entry with:
     - diagnosis (string)
     - verdict (one of: "good", "partial", "missing", "unlikely")
     - short_rationale (1-2 sentences)
   Definitions:
     - "good": appears among the top-5 and consistent with the symptoms/context.
     - "partial": related but needs more specific evidence; state what extra info is needed.
     - "missing": a potentially important or dangerous condition is not in the user's list.
     - "unlikely": unlikely given symptoms/context.
3) If any potentially dangerous conditions are judged "missing", DO NOT NAME them. Instead, produce targeted follow-up question(s) that would help the user determine whether such dangerous conditions might be relevant (e.g. "Is there severe chest pain radiating to the arm?"). Do not reveal the missing diagnosis names.
4) For each of the top-5 diagnoses, produce one concise follow-up question that would help confirm or refute it.
5) Be concise.

Output: produce a JSON object that exactly matches the format instructions below.

Retrieved context:
{context}

Symptoms:
{symptoms}

History:
{history}

User differentials (comma separated):
{user_differentials}

Lab test orders (if any):
{lab_orders}

Lab test results (if any):
{lab_results}

Final diagnosis (if any):
{final_diagnosis}

Treatment plan (if any):
{treatment_plan}

{format_instructions}
"""

lab_test_evaluation_prompt_template = """
You are a medical assistant.

You will receive:
- Retrieved medical context from a trusted database (may be empty).
- Patient symptoms (symptoms).
- Patient history (history, may be empty).
- Preliminary differential diagnoses (user_differentials, optional).
- Lab test orders (lab_tests or lab_orders).
- Lab test results (lab_results, optional).
- Target diagnoses (diagnoses).
- Final diagnosis (final_diagnosis, optional).
- Treatment plan (treatment_plan, optional).

Use only the fields relevant to your current task. Ignore fields that are not provided or not relevant.

Task:
Evaluate the lab tests performed and determine:
1) Whether these tests are sufficient and appropriate to help distinguish or confirm the target diagnoses as a group.
2) Identify any lab tests that are unnecessary because they do not contribute to differentiating or confirming any diagnosis in the list.
3) Identify any important missing tests that, if performed, could improve diagnostic differentiation among the target diagnoses.
4) Provide an overall verdict summarizing the sufficiency of the performed tests (e.g., "sufficient", "partially sufficient", "insufficient").

Base your reasoning strictly on the retrieved context and accepted clinical guidelines.
Be concise.

Output: produce a JSON object that exactly matches the format instructions below.

Retrieved context:
{context}

Symptoms:
{symptoms}

History:
{history}

Preliminary differentials (if any):
{user_differentials}

Lab tests performed:
{lab_tests}

Lab test results (if any):
{lab_results}

Target diagnoses (comma separated):
{diagnoses}

Final diagnosis (if any):
{final_diagnosis}

Treatment plan (if any):
{treatment_plan}

{format_instructions}
"""

lab_test_result_evaluation_prompt_template = """
You are a medical assistant.

You will receive:
- Retrieved medical context from a trusted database (may be empty).
- Patient symptoms (symptoms).
- Patient history (history, may be empty).
- Preliminary differential diagnoses (user_differentials, optional).
- Lab test orders (lab_orders, optional).
- Lab test results (lab_test_results or lab_results).
- Target diagnoses (diagnoses).
- Final diagnosis (final_diagnosis, optional).
- Treatment plan (treatment_plan, optional).

Use only the fields relevant to your current task. Ignore fields that are not provided or not relevant.

Task:
1) For each target diagnosis:
   - Evaluate whether the current lab test results are sufficient to confirm or rule out the diagnosis.
   - Identify any additional lab tests needed to better differentiate between these diagnoses.
   - Identify any performed lab tests that are unlikely to contribute useful information for differentiating these diagnoses.
2) Base your reasoning strictly on the retrieved context and established clinical guidelines.
3) Provide concise, clear, and clinically relevant explanations.

Output: produce a JSON object with a single key "results", which is a list of objects with the following fields:
  - diagnosis (string)
  - verdict (one of: "confirmed", "ruled_out", "insufficient")
  - missing_tests (list of strings) — tests that should be ordered to clarify diagnosis
  - unnecessary_tests (list of strings) — tests performed but not helpful for diagnosis
  - rationale (string) — brief explanation supporting your evaluation for each diagnosis

Include any additional notes or recommendations if appropriate.

Retrieved context:
{context}

Symptoms:
{symptoms}

History:
{history}

Preliminary differentials (if any):
{user_differentials}

Lab test orders (if any):
{lab_orders}

Lab test results:
{lab_test_results}

Target diagnoses (comma separated):
{diagnoses}

Final diagnosis (if any):
{final_diagnosis}

Treatment plan (if any):
{treatment_plan}

{format_instructions}
"""

def get_diagnosis_prompt():
    return PromptTemplate.from_template(diagnosis_prompt_template)

def get_lab_test_evaluation_prompt():
    return PromptTemplate.from_template(lab_test_evaluation_prompt_template)

def get_lab_test_result_evaluation_prompt():
    return PromptTemplate.from_template(lab_test_result_evaluation_prompt_template)