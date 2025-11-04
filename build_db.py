import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- Đường dẫn file ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "disease_lab_tests.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_2")

if not os.path.exists(persistent_directory):
    print("📦 Khởi tạo vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file {file_path}")

    # Load file txt
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Chia thành chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"📄 Số chunks: {len(docs)}")

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        temperature=0.5,
        google_api_key=os.environ["GEMINI_API_KEY"]
    )
    # Tạo vector DB
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("✅ Vector store đã được tạo và lưu trữ")
else:
    print("✅ Vector store đã tồn tại, không cần tạo lại")
