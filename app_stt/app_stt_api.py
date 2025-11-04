import os
import tempfile
import torch
import subprocess
import numpy as np
import soundfile as sf
import time
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import google.generativeai as genai
import sherpa_onnx

# ===============================
# 0️⃣ Load .env & Gemini config
# ===============================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Không tìm thấy GEMINI_API_KEY trong .env")

genai.configure(api_key=API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

SYSTEM_PROMPT = """
Bạn là bác sĩ Việt Nam chuyên về hiệu đính văn bản y khoa nhận dạng từ giọng nói.
Nhiệm vụ của bạn:
1. Chuẩn hóa chính tả, dấu câu, viết hoa.
2. Hiệu chỉnh các thuật ngữ y khoa sai (ví dụ “sp hai chín lăm” → “SpO2 95%”, “milimet thủy ngân” → “mmHg”).
3. Viết lại văn bản y khoa chuẩn, mạch lạc, đúng ngữ pháp, **giữ nguyên nội dung**.
4. Không thêm diễn giải, chỉ trả về **văn bản đã chỉnh sửa**.
5. Xóa các từ thừa, lặp từ, câu vô nghĩa.
"""

# ===============================
# 1️⃣ Load ZIPFORMER (sherpa_onnx)
# ===============================
print("🔹 Loading Zipformer...")
zipformer_model = sherpa_onnx.OfflineRecognizer.from_transducer(
    tokens="resources/config.json",
    encoder="resources/encoder-epoch-20-avg-10.onnx",
    decoder="resources/decoder-epoch-20-avg-10.onnx",
    joiner="resources/joiner-epoch-20-avg-10.onnx",
    num_threads=4,
    decoding_method="greedy_search",
)
print("✅ Zipformer ready.\n")


# ===============================
# 2️⃣ Utility Functions
# ===============================
def convert_to_wav(input_path):
    """Convert any audio/video file to mono WAV 16kHz"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_path = tmp_wav.name
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_path


def transcribe_zipformer(audio_path: str):
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        stream = zipformer_model.create_stream()
        stream.accept_waveform(sr, audio)
        zipformer_model.decode_stream(stream)
        return stream.result.text.strip()
    except Exception as e:
        return f"❌ Lỗi khi xử lý Zipformer: {e}"


def refine_text(raw_text):
    try:
        if not raw_text.strip():
            return "⚠️ Không có văn bản để hiệu đính."
        prompt = f"{SYSTEM_PROMPT}\n\nVăn bản cần hiệu chỉnh:\n{raw_text}\n\nKết quả chỉnh sửa:"
        response = gemini_model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        elif response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        return raw_text
    except Exception as e:
        return f"❌ Lỗi khi hiệu đính: {e}"


# ===============================
# 3️⃣ FastAPI App
# ===============================
app = FastAPI(title="Vietnamese STT Webhook", description="Whisper/Zipformer + Gemini Refiner", version="1.0")


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """Webhook nhận file audio/video, trả về JSON kết quả"""
    try:
        # Lưu file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Chuyển sang wav
        wav_path = convert_to_wav(tmp_path)

        # STT
        start = time.time()
        raw_text = transcribe_zipformer(wav_path)
        elapsed = round(time.time() - start, 2)

        # Refine
        refined = refine_text(raw_text)

        # Cleanup
        os.remove(tmp_path)
        os.remove(wav_path)

        return JSONResponse({
            "elapsed_time": elapsed,
            "raw_text": raw_text,
            "refined_text": refined
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {"message": "Vietnamese STT Webhook API is running 🚀"}


# ===============================
# 4️⃣ Run server
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)

