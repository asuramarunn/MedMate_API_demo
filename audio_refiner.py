import pandas as pd
import time
import google.generativeai as genai
import os
from tqdm import tqdm
from dotenv import load_dotenv

# ==== 1️⃣ LOAD ENV ====
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Không tìm thấy GEMINI_API_KEY trong .env")

genai.configure(api_key=API_KEY)

# ==== 2️⃣ CONFIG ====
CSV_PATH = "results.csv"
OUTPUT_CSV = "audio_refined.csv"
MODELS_TO_REFINE = ["ChunkFormer", "Zipformer"]
GEMINI_MODEL = "gemini-2.0-flash"  # thử model này trước

# ==== 3️⃣ INIT MODEL ====
model = genai.GenerativeModel(GEMINI_MODEL)

# ==== 4️⃣ PROMPT ====
SYSTEM_PROMPT = """
Bạn là bác sĩ Việt Nam chuyên về hiệu đính văn bản y khoa nhận dạng từ giọng nói.
Nhiệm vụ của bạn:
1. Chuẩn hóa chính tả, dấu câu, viết hoa.
2. Hiệu chỉnh các thuật ngữ y khoa sai (ví dụ “sp hai chín lăm” → “SpO2 95%”, “milimet thủy ngân” → “mmHg”).
3. Viết lại văn bản y khoa chuẩn, mạch lạc, đúng ngữ pháp, **giữ nguyên nội dung**.
4. Không thêm diễn giải, chỉ trả về **văn bản đã chỉnh sửa**.
"""

# ==== 5️⃣ LOAD DATA ====
df = pd.read_csv(CSV_PATH)
df["refined_text"] = ""
df["input_tokens"] = 0
df["output_tokens"] = 0

# ==== 6️⃣ PROCESS ====
for i, row in tqdm(df.iterrows(), total=len(df)):
    if row["model"] not in MODELS_TO_REFINE:
        continue

    prompt = f"{SYSTEM_PROMPT}\n\nVăn bản cần hiệu chỉnh:\n{row['transcribe']}\n\nKết quả chỉnh sửa:"

    try:
        response = model.generate_content(prompt)
        refined = ""

        # ✅ Kiểm tra xem Gemini có trả ra text không
        if hasattr(response, "text") and response.text:
            refined = response.text.strip()
        elif response.candidates and response.candidates[0].content.parts:
            refined = response.candidates[0].content.parts[0].text.strip()

        # ✅ Token usage
        usage = getattr(response, "usage_metadata", {})
        input_toks = usage.get("prompt_token_count", 0)
        output_toks = usage.get("candidates_token_count", 0)

        df.loc[i, "refined_text"] = refined
        df.loc[i, "input_tokens"] = input_toks
        df.loc[i, "output_tokens"] = output_toks

        # Ghi dần ra file để không mất dữ liệu
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        # Delay nhẹ để tránh rate limit
        time.sleep(0.8)

    except Exception as e:
        print(f"❌ Error at row {i}: {e}")
        time.sleep(5)
        continue

print(f"✅ Saved refined results to {OUTPUT_CSV}")
