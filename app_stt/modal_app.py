import modal

app = modal.App("vietnamese-stt-webhook")

# 🧱 Base image
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi",
        "uvicorn",
        "soundfile",
        "numpy",
        "torch",
        "google-generativeai",
        "sherpa-onnx",
        "python-dotenv",
        "packaging",   # 👈 thêm dòng này
        "python-multipart",  # 👈 thêm dòng này

    )
    .add_local_dir("resources", "/root/resources")  # ✅ Mount resources
    .add_local_file("app_stt_api.py", "/root/app_stt_api.py")  # 🔹 Thêm dòng này

)

# ✅ Thứ tự đúng: app.function ở trên, modal.asgi_app ở dưới
@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv(".env")],
    min_containers=1,  # Thay cho keep_warm
)
@modal.asgi_app()
def fastapi_app():
    from app_stt_api import app
    return app
