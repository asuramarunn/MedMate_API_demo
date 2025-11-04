# agent_base.py (bản nâng cấp)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

class GenericAgent:
    def __init__(
        self,
        functional_prompt,
        use_rag=False,
        db_dir=None,
        embedding_model="gemini-embedding-001",
        llm_model="gemini-2.5-pro",
        system_instruction=None,
        year=None,
        k=5
    ):
        self.year = year
        self.functional_prompt = functional_prompt
        self.use_rag = use_rag

        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.0,
            google_api_key=os.environ["GEMINI_API_KEY"],
            model_kwargs={"system_instruction": system_instruction} if system_instruction else {}
        )

        if use_rag:
            if not db_dir:
                raise ValueError("db_dir is required when use_rag=True")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=os.environ["GEMINI_API_KEY"]
            )
            self.db = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
            self.retriever = self.db.as_retriever(search_kwargs={"k": k})
        else:
            self.retriever = None

    def invoke(self, inputs: dict):
        query = inputs.get("query", "")
        context = ""
        if self.use_rag and query:
            docs = self.retriever.get_relevant_documents(query)
            context = "\n\n".join(d.page_content for d in docs)

        # Thay vì parser, chỉ format prompt từ functional_prompt + context
        full_inputs = {**inputs, "context": context}
        prompt_text = self.functional_prompt.format(**full_inputs)
        completion = self.llm.invoke(prompt_text)
        return getattr(completion, "content", str(completion))
