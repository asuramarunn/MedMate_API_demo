from langchain.output_parsers import ResponseSchema


diagnosis_response_schemas = [
            ResponseSchema(
                name="top_5",
                description="A list of up to 5 objects with fields: diagnosis (string) and brief_reason (short string)."
            ),
            ResponseSchema(
                name="user_evaluation",
                description="A list of objects, one per user-provided diagnosis: {diagnosis, verdict, short_rationale}."
            ),
            ResponseSchema(
                name="missing_dangerous",
                description="Yes or No. Whether any potentially dangerous conditions are missing from the user's list."
            ),
            ResponseSchema(
                name="follow_up_questions_for_missing",
                description="If missing_dangerous is Yes, a list of targeted follow-up questions to prompt the user to consider those missing dangerous conditions. DO NOT name the missing conditions."
            ),
            ResponseSchema(
                name="follow_up_questions_for_top5",
                description="A list of objects for each top_5 item: {diagnosis, question} - one targeted question each."
            ),
            ResponseSchema(
                name="notes",
                description="Optional short note when context is empty or results uncertain."
            )
        ]

lab_test_response_schemas = [
    ResponseSchema(
        name="overall_verdict",
        description="A string verdict summarizing whether the performed tests are sufficient, partially sufficient, or insufficient."
    ),
    ResponseSchema(
        name="tests_helpful",
        description="A list of lab tests (strings) performed that help differentiate or confirm at least one diagnosis."
    ),
    ResponseSchema(
        name="tests_unnecessary",
        description="A list of lab tests (strings) performed that do not help differentiate or confirm any diagnosis."
    ),
    ResponseSchema(
        name="missing_tests",
        description="A list of lab tests (strings) recommended but not performed that could improve differentiation."
    )
]

lab_test_result_response_schemas = [
    ResponseSchema(
        name="results",
        description="List of evaluations for each diagnosis"
    ),
    ResponseSchema(
        name="diagnosis",
        description="The diagnosis evaluated"
    ),
    ResponseSchema(
        name="verdict",
        description="Assessment of whether current lab results confirm, rule out, or are insufficient to evaluate diagnosis"
    ),
    ResponseSchema(
        name="missing_tests",
        description="Additional lab tests recommended to better differentiate or confirm diagnosis"
    ),
    ResponseSchema(
        name="low_value_tests",
        description="Tests performed but providing little useful information for distinguishing diagnosis"
    ),
    ResponseSchema(
        name="rationale",
        description="Brief explanation for the verdict"
    )
]

