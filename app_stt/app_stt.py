# import os
# import tempfile
# import gradio as gr
# import torch
# import subprocess
# import numpy as np
# import soundfile as sf
# import time

# from transformers import (
#     pipeline,
#     WhisperTokenizer,
#     WhisperFeatureExtractor,
#     WhisperForConditionalGeneration,
# )
# from huggingface_hub import hf_hub_download

# from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, pipeline
# import torch
# from chunkformer import ChunkFormerModel

# # ===============================
# # 1️⃣ Load WHISPER fine-tuned (OFFLINE)
# # ===============================

# # 🔹 Đường dẫn tới snapshot local của Whisper fine-tuned
# CHECKPOINT_LOCAL = "/home/datnguyen/.cache/huggingface/hub/models--asuramarunnn--medmate-whisper-tiny-vi-v1/snapshots/4707abe66416511895988dbc2f240eb0514b240b"

# # 🔹 Đường dẫn tới base model (nếu cần tokenizer/feature_extractor)
# BASE_MODEL_ID = "/home/datnguyen/.cache/huggingface/hub/models--doof-ferb--whisper-tiny-vi/snapshots/a7f8c3da397f4d4b184b946f18647758682d6a05"

# print("🔹 Loading Whisper fine-tuned (offline)...")
# model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT_LOCAL)
# tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_ID, language="vi", task="transcribe")
# feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_ID)
# device = 0 if torch.cuda.is_available() else -1

# pipe_whisper = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=tokenizer,
#     feature_extractor=feature_extractor,
#     device=device,
# )
# PIPE_KWARGS = {"language": "vi", "task": "transcribe"}
# print("✅ Whisper ready.\n")

# # ===============================
# # 2️⃣ Load CHUNKFORMER (OFFLINE)
# # ===============================

# print("🔹 Loading Chunkformer (offline)...")

# CHUNKFORMER_PATH = "/home/datnguyen/.cache/huggingface/hub/models--khanhld--chunkformer-ctc-large-vie/snapshots/311fc03558a895dc2b32957f2fb4236c7fb1455b"

# chunkformer_model = ChunkFormerModel.from_pretrained(CHUNKFORMER_PATH)
# print("✅ Chunkformer ready.\n")

# # ===============================
# # 3️⃣ Load ZIPFORMER (sherpa_onnx)
# # ===============================
# print("🔹 Loading Zipformer...")
# import sherpa_onnx


# zipformer_model = sherpa_onnx.OfflineRecognizer.from_transducer(
#     tokens="resources/config.json",
#     encoder="resources/encoder-epoch-20-avg-10.onnx",
#     decoder="resources/decoder-epoch-20-avg-10.onnx",
#     joiner="resources/joiner-epoch-20-avg-10.onnx",
#     num_threads=4,
#     decoding_method="greedy_search",
# )

# print("✅ Zipformer ready.\n")


# # ===============================
# # 4️⃣ Tiện ích chuyển đổi định dạng
# # ===============================
# def convert_to_wav(input_data):
#     """
#     Nhận:
#       - tuple (sr, np.array) từ microphone
#       - hoặc đường dẫn file upload (.mp3, .mp4, .m4a, .wav, ...)
#     Trả về:
#       - tuple (sr, np.array)
#     """
#     if isinstance(input_data, tuple):
#         return input_data

#     elif isinstance(input_data, str) and os.path.exists(input_data):
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
#             tmp_wav_path = tmp_wav.name
#         command = [
#             "ffmpeg", "-y", "-i", input_data,
#             "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav_path,
#         ]
#         subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         audio_data, sr = sf.read(tmp_wav_path)
#         os.remove(tmp_wav_path)
#         return sr, audio_data
#     else:
#         raise ValueError("Dữ liệu âm thanh không hợp lệ.")


# # ===============================
# # 5️⃣ Hàm chia audio
# # ===============================
# def chunk_audio(audio_array, sr, max_duration=25):
#     """Chia audio thành các đoạn ~25s"""
#     chunk_size = int(sr * max_duration)
#     num_chunks = int(np.ceil(len(audio_array) / chunk_size))
#     return [audio_array[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


# import sherpa_onnx
# import soundfile as sf
# import numpy as np

# def transcribe_zipformer(audio_path: str):
#     try:
#         # Đọc audio (mono, float32)
#         audio, sr = sf.read(audio_path)
#         if len(audio.shape) > 1:
#             audio = np.mean(audio, axis=1)  # chuyển stereo → mono
#         audio = audio.astype(np.float32)

#         # Tạo stream mới
#         stream = zipformer_model.create_stream()

#         # Thêm waveform vào stream
#         stream.accept_waveform(sr, audio)

#         # Decode
#         zipformer_model.decode_stream(stream)

#         # Lấy kết quả
#         text = stream.result.text
#         return text.strip()

#     except Exception as e:
#         return f"❌ Lỗi khi xử lý Zipformer: {e}"

# # ===============================
# # 6️⃣ Hàm inference cho từng model
# # ===============================
# def transcribe(audio_input, model_name):
#     if audio_input is None:
#         return "⚠️ Vui lòng ghi âm hoặc upload file âm thanh / video."

#     try:
#         sr, audio_array = convert_to_wav(audio_input)

#         # ===============================
#         # Whisper
#         # ===============================
#         if model_name == "Whisper (fine-tuned)":
#             chunks = chunk_audio(np.array(audio_array), sr)
#             full_text = ""
#             t0 = time.time()
#             for i, chunk in enumerate(chunks):
#                 result = pipe_whisper(
#                     {"array": chunk, "sampling_rate": sr},
#                     generate_kwargs=PIPE_KWARGS,
#                     return_timestamps=False,
#                 )
#                 full_text += " " + result.get("text", "").strip()
#             elapsed = time.time() - t0
#             return f"🕒 {elapsed:.2f}s\n\n{full_text.strip()}"

#         # ===============================
#         # Chunkformer
#         # ===============================
#         elif model_name == "Chunkformer":
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
#                 sf.write(tmp_wav.name, audio_array, sr)
#                 t0 = time.time()
#                 result = chunkformer_model.endless_decode(
#                     audio_path=tmp_wav.name,
#                     chunk_size=64,
#                     left_context_size=128,
#                     right_context_size=128,
#                     return_timestamps=False,
#                 )
#                 elapsed = time.time() - t0
#                 text = result["text"] if isinstance(result, dict) else str(result)
#             return f"🕒 {elapsed:.2f}s\n\n{text.strip()}"

#         # ===============================
#         # Zipformer
#         # ===============================
#         elif model_name == "Zipformer":
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
#                 sf.write(tmp_wav.name, audio_array, sr)
#                 t0 = time.time()
#                 result = transcribe_zipformer(tmp_wav.name)
#                 elapsed = time.time() - t0
#             return f"🕒 {elapsed:.2f}s\n\n{result.strip()}"

#         else:
#             return "❌ Model không hợp lệ."

#     except Exception as e:
#         return f"❌ Lỗi khi xử lý âm thanh: {e}"


# # ===============================
# # 7️⃣ Gradio UI
# # ===============================
# demo = gr.Interface(
#     fn=transcribe,
#     inputs=[
#         gr.Audio(
#             sources=["microphone", "upload"],
#             type="filepath",
#             label="🎤 Ghi âm hoặc tải lên file (.mp3, .mp4, .m4a, .wav, ...)",
#         ),
#         gr.Dropdown(
#             ["Whisper (fine-tuned)", "Chunkformer", "Zipformer"],
#             value="Whisper (fine-tuned)",
#             label="Chọn mô hình STT",
#         ),
#     ],
#     outputs=gr.Textbox(label="📝 Kết quả chuyển âm", lines=8),
#     title="🇻🇳 Speech-to-Text Demo (Vietnamese ASR)",
#     description=(
#         "So sánh 3 mô hình STT tiếng Việt:\n"
#         "• Whisper fine-tuned (MedMate)\n"
#         "• Chunkformer (CTC)\n"
#         "• Zipformer (RNNT)\n\n"
#         "Hỗ trợ upload hoặc ghi âm trực tiếp."
#     ),
# )

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)


# ------------------------------------------------------------------------------------

import os
import tempfile
import gradio as gr
import torch
import subprocess
import numpy as np
import soundfile as sf
import time
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai

from transformers import (
    pipeline,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)
# from chunkformer import ChunkFormerModel
import sherpa_onnx
from chunkformer import ChunkFormerModel


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
3. Viết lại văn bản y khoa chuẩn, mạch lạc, đúng ngữ pháp, giữ nguyên nội dung.
4. Không thêm diễn giải, chỉ trả về văn bản đã chỉnh sửa.
5. Xóa các từ thừa, lặp từ, câu vô nghĩa.
6. Giữ nguyên các con số, đơn vị đo lường.
"""

# ===============================
# 2️⃣ Load CHUNKFORMER (OFFLINE)
# ===============================

print("🔹 Loading Chunkformer (offline)...")

CHUNKFORMER_PATH = "/home/datnguyen/.cache/huggingface/hub/models--khanhld--chunkformer-ctc-large-vie/snapshots/311fc03558a895dc2b32957f2fb4236c7fb1455b"

chunkformer_model = ChunkFormerModel.from_pretrained(CHUNKFORMER_PATH)
print("✅ Chunkformer ready.\n")

# ===============================
# 3️⃣ Load ZIPFORMER (sherpa_onnx)
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
# 4️⃣ Convert / Chunk utilities
# ===============================
def convert_to_wav(input_data):
    if isinstance(input_data, tuple):
        return input_data
    elif isinstance(input_data, str) and os.path.exists(input_data):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        command = [
            "ffmpeg", "-y", "-i", input_data,
            "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav_path,
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio_data, sr = sf.read(tmp_wav_path)
        os.remove(tmp_wav_path)
        return sr, audio_data
    else:
        raise ValueError("Dữ liệu âm thanh không hợp lệ.")


def chunk_audio(audio_array, sr, max_duration=25):
    chunk_size = int(sr * max_duration)
    num_chunks = int(np.ceil(len(audio_array) / chunk_size))
    return [audio_array[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


def transcribe_zipformer(audio_path: str):
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        stream = zipformer_model.create_stream()
        stream.accept_waveform(sr, audio)
        zipformer_model.decode_stream(stream)
        text = stream.result.text
        return text.strip()
    except Exception as e:
        return f"❌ Lỗi khi xử lý Zipformer: {e}"


# ===============================
# 5️⃣ Auto Refiner (Gemini)
# ===============================
def refine_text(raw_text):
    try:
        if not raw_text or raw_text.strip() == "":
            return "⚠️ Không có văn bản để hiệu đính."
        prompt = f"{SYSTEM_PROMPT}\n\nVăn bản cần hiệu chỉnh:\n{raw_text}\n\nKết quả chỉnh sửa:"
        response = gemini_model.generate_content(prompt)
        refined = ""
        if hasattr(response, "text") and response.text:
            refined = response.text.strip()
        elif response.candidates and response.candidates[0].content.parts:
            refined = response.candidates[0].content.parts[0].text.strip()
        return refined or raw_text
    except Exception as e:
        return f"❌ Lỗi khi hiệu đính: {e}"


# ===============================
# 6️⃣ Inference
# ===============================
def transcribe(audio_input, model_name):
    if audio_input is None:
        return "⚠️ Vui lòng ghi âm hoặc upload file âm thanh / video."

    try:
        sr, audio_array = convert_to_wav(audio_input)


        if model_name == "Chunkformer":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                sf.write(tmp_wav.name, audio_array, sr)
                t0 = time.time()
                result = chunkformer_model.endless_decode(
                    audio_path=tmp_wav.name,
                    chunk_size=64,
                    left_context_size=128,
                    right_context_size=128,
                    return_timestamps=False,
                )
                elapsed = time.time() - t0
                text = result["text"] if isinstance(result, dict) else str(result)
            refined = refine_text(text.strip())
            return f"🕒 {elapsed:.2f}s\n\n---\n**Raw:** {text.strip()}\n\n---\n**Refined:** {refined}"


        elif model_name == "Zipformer":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                sf.write(tmp_wav.name, audio_array, sr)
                t0 = time.time()
                result = transcribe_zipformer(tmp_wav.name)
                elapsed = time.time() - t0
            refined = refine_text(result.strip())
            return f"🕒 {elapsed:.2f}s\n\n---\n**Raw:** {result.strip()}\n\n---\n**Refined:** {refined}"

        else:
            return "❌ Model không hợp lệ."

    except Exception as e:
        return f"❌ Lỗi khi xử lý âm thanh: {e}"


# ===============================
# 7️⃣ Gradio UI
# ===============================
demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="🎤 Ghi âm hoặc tải lên file (.mp3, .mp4, .m4a, .wav, ...)",
        ),
        gr.Dropdown(
            [ "Chunkformer", "Zipformer"],
            value="Chunkformer",
            label="Chọn mô hình STT",
        ),
    ],
    outputs=gr.Markdown(label="📝 Kết quả chuyển âm"),
    title="🇻🇳 Speech-to-Text Demo + Auto Refiner (Vietnamese ASR)",
    description=(
        "So sánh 3 mô hình STT tiếng Việt (Whisper, Chunkformer, Zipformer)\n\n"
        "Sau khi nhận dạng, Gemini sẽ tự động hiệu đính văn bản y khoa ✨"
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
