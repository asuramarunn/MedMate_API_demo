import os
import time
import tempfile
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
import torch
import jiwer
import subprocess
 

# -------------------------------
# 1️⃣ Folder chứa audio
# -------------------------------

AUDIO_DIR = "VOICE"  # <--- đổi thành folder của bạn


from transformers import pipeline, WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
from chunkformer import ChunkFormerModel


# ===============================
# 1️⃣ Load WHISPER fine-tuned (OFFLINE)
# ===============================

# 🔹 Đường dẫn tới snapshot local của Whisper fine-tuned
CHECKPOINT_LOCAL = "/home/datnguyen/.cache/huggingface/hub/models--asuramarunnn--medmate-whisper-tiny-vi-v1/snapshots/4707abe66416511895988dbc2f240eb0514b240b"

# 🔹 Đường dẫn tới base model (nếu cần tokenizer/feature_extractor)
BASE_MODEL_ID = "/home/datnguyen/.cache/huggingface/hub/models--doof-ferb--whisper-tiny-vi/snapshots/a7f8c3da397f4d4b184b946f18647758682d6a05"

print("🔹 Loading Whisper fine-tuned (offline)...")
model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT_LOCAL)
tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL_ID, language="vi", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_ID)
device = 0 if torch.cuda.is_available() else -1

pipe_whisper = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    device=device,
)
PIPE_KWARGS = {"language": "vi", "task": "transcribe"}
print("✅ Whisper ready.\n")

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
import sherpa_onnx


zipformer_model = sherpa_onnx.OfflineRecognizer.from_transducer(
    tokens="resources/config.json",
    encoder="resources/encoder-epoch-20-avg-10.onnx",
    decoder="resources/decoder-epoch-20-avg-10.onnx",
    joiner="resources/joiner-epoch-20-avg-10.onnx",
    num_threads=4,
    decoding_method="greedy_search",
)

print("✅ Zipformer ready.\n")



# -------------------------------
# 5️⃣ Utility functions
# -------------------------------
def get_audio_path(audio_name):
    path = os.path.join(AUDIO_DIR, f"{audio_name}.m4a")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} không tồn tại")
    return path

def convert_to_wav(input_data):
    """
    Nhận:
      - tuple (sr, np.array) từ microphone
      - hoặc đường dẫn file (.mp3, .mp4, .m4a, .wav, ...)
    Trả về:
      - tuple (sr, np.array)
    """
    if isinstance(input_data, tuple):
        # Dữ liệu từ mic: (sr, np.array)
        return input_data

    elif isinstance(input_data, str) and os.path.exists(input_data):
        # Dữ liệu từ file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        # Chuyển mọi định dạng âm thanh về mono 16kHz WAV
        command = [
            "ffmpeg", "-y",
            "-i", input_data,
            "-ar", "16000",  # sample rate
            "-ac", "1",      # mono
            "-f", "wav", tmp_wav_path,
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Đọc dữ liệu WAV vừa convert
        audio_data, sr = sf.read(tmp_wav_path)
        os.remove(tmp_wav_path)
        return sr, audio_data

    else:
        raise ValueError(f"Đầu vào âm thanh không hợp lệ: {input_data}")

def load_audio(audio_input):
    """
    Hàm đọc âm thanh và trả về:
      - data: numpy array (mono)
      - sr: sample rate
      - duration: độ dài tính bằng giây
    """
    sr, data = convert_to_wav(audio_input)

    # Nếu multi-channel (trong trường hợp lỗi), chuyển về mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    duration = len(data) / sr
    return data, sr, duration


# ---------------- Normalization + WER helpers ----------------
import re, unicodedata, difflib, jiwer

# -------- BẢNG CƠ SỞ -------- #
_DIGIT_WORD = {"0":"không","1":"một","2":"hai","3":"ba","4":"bốn","5":"năm","6":"sáu","7":"bảy","8":"tám","9":"chín"}
_NUM_WORDS_SIMPLE = {
    0:"không",1:"một",2:"hai",3:"ba",4:"bốn",5:"năm",6:"sáu",7:"bảy",8:"tám",9:"chín",
    10:"mười",11:"mười một",12:"mười hai",13:"mười ba",14:"mười bốn",15:"mười lăm",
    20:"hai mươi",30:"ba mươi",40:"bốn mươi",50:"năm mươi",60:"sáu mươi",
    70:"bảy mươi",80:"tám mươi",90:"chín mươi"
}

def int_to_vn(n:int)->str:
    if n<0: return "âm "+int_to_vn(-n)
    if n<=20: return _NUM_WORDS_SIMPLE.get(n,str(n))
    if n<100:
        tens=(n//10)*10; unit=n%10
        return _NUM_WORDS_SIMPLE[tens]+("" if unit==0 else " "+_NUM_WORDS_SIMPLE.get(unit,str(unit)))
    if n<1000:
        h=n//100; rest=n%100
        return _NUM_WORDS_SIMPLE[h]+" trăm"+("" if rest==0 else " "+int_to_vn(rest))
    if n<10000:
        th=n//1000; rest=n%1000
        return _NUM_WORDS_SIMPLE[th]+" nghìn"+("" if rest==0 else " "+int_to_vn(rest))
    return str(n)

def frac_digits_to_vn(frac:str)->str:
    return " ".join(_DIGIT_WORD.get(ch,ch) for ch in frac)

def number_to_vn_words(token:str)->str:
    s=token.replace(",",".")
    if "." in s:
        ip,fp=s.split(".",1)
        try: ipw=int_to_vn(int(ip))
        except: ipw=" ".join(_DIGIT_WORD.get(ch,ch) for ch in ip)
        fpw=frac_digits_to_vn(fp)
        return f"{ipw} chấm {fpw}"
    try: return int_to_vn(int(s))
    except: return " ".join(_DIGIT_WORD.get(ch,ch) for ch in s)

# -------- ĐƠN VỊ & THUẬT NGỮ Y HỌC -------- #
_UNIT_PATTERNS = [
    (r"\bmmhg\b","milimet thủy ngân"),
    (r"\bspo2\b","sp ô hai"),
    (r"\bbmi\b","bê em ai"),
    (r"°c","độ c"),
    (r"kg\/m2","kg trên mét vuông"),
    (r"kg\/m²","kg trên mét vuông"),
    (r"%","phần trăm"),
    (r"/phút","trên phút"),
    (r"\/","trên"),
]

_SPECIAL_TERMS = {
    "mạch":"mach",
    "huyết áp":"huyet ap",
    "nhiệt độ":"nhiet do",
    "nhịp thở":"nhip tho",
    "tỉnh":"tinh",
    "toàn trạng":"toan trang"
}

# -------- NORMALIZATION -------- #
_PUNCT_RE = re.compile(r"[^0-9a-zA-Zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
                       r"ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữự"
                       r"ỳýỷỹỵđ\s°²]", re.U)

def normalize_text(text:str)->str:
    if text is None: return ""
    t=str(text).strip().lower()
    t=unicodedata.normalize("NFC",t)
    
    # thay thế thuật ngữ y học trước
    for k,v in _SPECIAL_TERMS.items():
        t=re.sub(rf"\b{k}\b",v,t)
    
    # xử lý từ "phẩy", "chấm"
    t=re.sub(r"\b(phẩy|chấm)\b","chấm",t)
    
    # thay đơn vị & ký hiệu
    for pat,rep in _UNIT_PATTERNS:
        t=re.sub(pat, f" {rep} ", t, flags=re.I)
    
    # tách 100/60 để parse chính xác
    t=re.sub(r"(\d+)/(\d+)", r"\1 / \2", t)
    
    # thay số thành chữ
    t=re.sub(r"\d+[0-9.,]*", lambda m:number_to_vn_words(m.group(0)), t)
    
    # chuẩn hoá số đọc “năm/lăm”
    t=re.sub(r"\b(lăm|năm)\b","năm",t)
    
    # xoá dấu
    t=_PUNCT_RE.sub(" ",t)
    t=re.sub(r"\s+"," ",t).strip()
    return t

# -------- WER & SAI KHÁC -------- #
def compute_normalized_wer(ref:str, hyp:str)->float:
    r=normalize_text(ref)
    h=normalize_text(hyp)
    return float(jiwer.wer(r,h))

def get_wrong_pairs(ref:str,hyp:str,max_pairs=10000)->str:
    r_words=normalize_text(ref).split()
    h_words=normalize_text(hyp).split()
    sm=difflib.SequenceMatcher(a=r_words,b=h_words)
    pairs=[]
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=="equal": continue
        if tag=="replace":
            for a,b in zip(r_words[i1:i2],h_words[j1:j2]):
                # nếu khác dạng số mà tương đương (vd: 5.2 vs 5,2)
                if re.sub(r"[.,]","",a)==re.sub(r"[.,]","",b): continue
                pairs.append(f"{a}->{b}")
        elif tag=="delete":
            for a in r_words[i1:i2]: pairs.append(f"{a}->")
        elif tag=="insert":
            for b in h_words[j1:j2]: pairs.append(f"->{b}")
        if len(pairs)>=max_pairs: break
    return ", ".join(pairs)


# -------------------------------
# 6️⃣ Transcribe functions
# -------------------------------
import os
import time
import tempfile
import numpy as np
import soundfile as sf
import jiwer

# ===============================
# 1️⃣ Whisper
# ===============================
def transcribe_whisper(audio_array, sr, max_chunk_duration=25):
    import numpy as np
    start = time.time()

    chunk_size = int(sr * max_chunk_duration)
    num_chunks = int(np.ceil(len(audio_array) / chunk_size))

    full_text = ""
    for i in range(num_chunks):
        chunk = audio_array[i * chunk_size:(i + 1) * chunk_size]
        result = pipe_whisper(
            {"array": chunk, "sampling_rate": sr},
            generate_kwargs=PIPE_KWARGS,
            return_timestamps=False,
        )
        full_text += " " + result.get("text", "").strip()

    elapsed = time.time() - start
    return full_text.strip(), elapsed


# ===============================
# 2️⃣ Chunkformer
# ===============================
def transcribe_chunkformer(audio_array, sr):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, audio_array, sr)
        tmp_path = tmp_wav.name

    start = time.time()
    result = chunkformer_model.endless_decode(
        audio_path=tmp_path,
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        return_timestamps=False,
    )
    elapsed = time.time() - start

    text = result["text"] if isinstance(result, dict) else str(result)
    os.remove(tmp_path)
    return text.strip(), elapsed


# ===============================
# 3️⃣ Zipformer
# ===============================
def transcribe_zipformer(audio_array, sr):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, audio_array, sr)
        tmp_path = tmp_wav.name

    try:
        audio, sr = sf.read(tmp_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        start = time.time()
        stream = zipformer_model.create_stream()
        stream.accept_waveform(sr, audio)
        zipformer_model.decode_stream(stream)
        text = stream.result.text
        elapsed = time.time() - start
        os.remove(tmp_path)

        return text.strip(), elapsed
    except Exception as e:
        os.remove(tmp_path)
        return f"❌ Lỗi khi xử lý Zipformer: {e}", 0.0


# ===============================
# 4️⃣ Xử lý từng hàng CSV
# ===============================
def process_row(row):
    audio_name = row["Audio"]
    ref_text = row["Transcribe"]
    results = []

    try:
        # Lấy đường dẫn audio
        audio_file = get_audio_path(audio_name)
        sr, audio_array = convert_to_wav(audio_file)

        # Tính độ dài
        duration = len(audio_array) / sr

        # -------------------------------
        # Whisper
        # -------------------------------
        text, t = transcribe_whisper(audio_array, sr)
        results.append({
            "audio": audio_name,
            "audio_length": duration,
            "transcribe": text,
            "wer": compute_normalized_wer(ref_text, text),
            "speed": t,
            "wrong": get_wrong_pairs(ref_text, text),
            "model": "Whisper"
        })

        # -------------------------------
        # ChunkFormer
        # -------------------------------
        text, t = transcribe_chunkformer(audio_array, sr)
        results.append({
            "audio": audio_name,
            "audio_length": duration,
            "transcribe": text,
            "wer": compute_normalized_wer(ref_text, text),
            "speed": t,
            "wrong": get_wrong_pairs(ref_text, text),
            "model": "ChunkFormer"
        })

        # -------------------------------
        # Zipformer
        # -------------------------------
        text, t = transcribe_zipformer(audio_array, sr)
        results.append({
            "audio": audio_name,
            "audio_length": duration,
            "transcribe": text,
            "wer": compute_normalized_wer(ref_text, text),
            "speed": t,
            "wrong": get_wrong_pairs(ref_text, text),
            "model": "Zipformer"
        })

    except Exception as e:
        print(f"❌ Error processing {audio_name}: {e}")

    return results

# -------------------------------
# 8️⃣ Main
# -------------------------------
if __name__ == "__main__":
    df = pd.read_csv("audio.csv")
    print(f"✅ Load CSV với {len(df)} hàng.")

    all_results = []
    for row in df.to_dict(orient="records"):
        all_results.extend(process_row(row))

    df_result = pd.DataFrame(all_results)
    df_result.to_csv("results.csv", index=False)
    print("✅ Đã ghi kết quả ra results.csv")
