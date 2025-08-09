# agent_base.py
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser
import os

class GenericAgent:
    def __init__(
        self,
        prompt_template,
        response_schemas,
        use_rag=True,
        db_dir=None,
        embedding_model="gemini-embedding-001",
        llm_model="gemini-2.0-flash-lite",
        k=5
    ):
        self.prompt_template = prompt_template
        self.parser = StructuredOutputParser.from_response_schemas(response_schemas)
        self.use_rag = use_rag
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.0,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        if use_rag:
            if not db_dir:
                raise ValueError("db_dir is required when use_rag=True")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=os.environ.get("GEMINI_API_KEY")
            )
            self.db = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
            self.retriever = self.db.as_retriever(search_kwargs={"k": k})
        else:
            self.retriever = None

    def invoke(self, inputs: dict):
        if self.use_rag:
            query = inputs.get("query", "")
            docs = self.retriever.get_relevant_documents(query) if query else []
            context = "\n\n".join([d.page_content for d in docs]) if docs else ""
        else:
            context = ""
        format_instructions = self.parser.get_format_instructions()
        full_inputs = {
            **inputs,
            "context": context,
            "format_instructions": format_instructions
        }
        prompt_text = self.prompt_template.format(**full_inputs)
        completion = self.llm.invoke(prompt_text)
        raw_text = getattr(completion, "content", None) or str(completion)
        try:
            parsed = self.parser.parse(raw_text)
        except Exception as e:
            import json
            try:
                json_part = raw_text[raw_text.index("{"): raw_text.rindex("}")+1]
                parsed = json.loads(json_part)
            except Exception:
                parsed = {"raw": raw_text, "parse_error": str(e)}
        return parsed
