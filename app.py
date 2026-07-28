import os
import sys
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import io

# --- Windows GPU DLL Düzeltmesi ---
if os.name == 'nt':
    try:
        import nvidia.cuda_runtime.lib
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
        os.add_dll_directory(os.path.dirname(nvidia.cuda_runtime.lib.__file__))
        os.add_dll_directory(os.path.dirname(nvidia.cublas.lib.__file__))
        os.add_dll_directory(os.path.dirname(nvidia.cudnn.lib.__file__))
    except ImportError:
        pass

from faster_whisper import WhisperModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# --- Modeller ve Global Değişkenler ---
whisper_model = None
llm_model = None
guide_content = ""

# --- JSON Schema (Llama.cpp için) ---
json_schema = {
    "type": "object",
    "properties": {
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target_npc_id": {
                        "type": ["integer", "string"]
                    },
                    "action_type": {
                        "type": "string",
                        "enum": ["move", "gather", "attack", "defend", "idle", "talk"]
                    },
                    "target_object": {
                        "type": ["string", "null"]
                    },
                    "target_location": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "is_specified": {"type": "boolean"}
                        },
                        "required": ["x", "y", "z", "is_specified"]
                    },
                    "npc_reply": {
                        "type": "string"
                    }
                },
                "required": ["target_npc_id", "action_type", "target_object", "target_location", "npc_reply"]
            }
        }
    },
    "required": ["commands"]
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, llm_model, guide_content
    
    print("Modeller yükleniyor, lütfen bekleyin...")
    
    # 1. Kılavuz dosyasını oku
    guide_path = os.path.join(os.path.dirname(__file__), "guide.md")
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            guide_content = f.read()
    except Exception as e:
        print(f"Uyarı: guide.md okunamadı: {e}")
        guide_content = "Kılavuz bulunamadı."
        
    # 2. Faster-Whisper'ı yükle (Önce CUDA deniyoruz, olmazsa CPU)
    try:
        print("Faster-Whisper 'cuda' (float16) ile başlatılmaya çalışılıyor...")
        whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
        device = "cuda"
    except Exception as e:
        print(f"GPU (CUDA) ile Whisper yükleme hatası: {e}")
        print("Faster-Whisper 'cpu' (int8) moduna düşürülüyor...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        device = "cpu"
        
    # 3. Llama.cpp (Llama-3.2-1B) yükle
    print("Llama-3.2-1B-Instruct-GGUF modeli kontrol ediliyor/indiriliyor...")
    try:
        model_path = hf_hub_download(
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        )
        print(f"LLM modeli bulundu: {model_path}")
        # n_ctx bağlam penceresi, n_gpu_layers GPU'ya yüklenecek katman sayısı (0=sadece CPU, -1=tamamı GPU)
        n_gpu_layers = -1 if device == "cuda" else 0
        llm_model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
    except Exception as e:
        print(f"LLM yükleme hatası: {e}")

    print("Sistem hazır!")
    yield
    print("Sistem kapanıyor. Kaynaklar temizleniyor...")

app = FastAPI(lifespan=lifespan, title="Oyun Ses Kontrol API")

@app.post("/api/v1/process-voice")
async def process_voice(
    audio: UploadFile = File(...),
    game_context: str = Form(...)
):
    if whisper_model is None or llm_model is None:
        return JSONResponse(status_code=503, content={
            "status": "error",
            "error_message": "Modeller henüz yüklenmedi veya yüklenirken bir hata oluştu."
        })
        
    if not audio.filename.endswith(".wav"):
        return JSONResponse(status_code=400, content={
            "status": "error",
            "error_message": "Sadece .wav formatında ses dosyaları kabul edilmektedir."
        })
        
    try:
        # Sesi doğrudan RAM'den (in-memory) işle
        content = await audio.read()
        audio_stream = io.BytesIO(content)
            
        # 1. STT: Ses -> Metin (beam_size=1 ile hızlandırıldı)
        segments, info = whisper_model.transcribe(audio_stream, language="tr", beam_size=1)
        transcription = " ".join([segment.text for segment in segments]).strip()
        
        if not transcription:
            raise ValueError("Ses dosyasından metin anlaşılamadı veya boş.")
            
        # 2. SLM: Metin -> JSON Komut
        system_prompt = f"""Sen bir JSON üreten yapay zeka asistanısın. Görevin oyuncunun verdiği tek cümleyi analiz edip kısa bir JSON çıktısı vermektir. Başka HİÇBİR ŞEY YAZMA.

KURALLAR:
1. 'visible_npcs' listesini kullanarak isimleri ID'lere dönüştür. İsim yoksa 'pointed_npc_id' değerini kullan.
2. Koordinat verilmemişse x,y,z HER ZAMAN 0 olmalı ve is_specified false olmalıdır.
3. SADECE oyuncunun istediği eylemleri listeye ekle. Asla fazladan eylem uydurma.
4. "npc_reply" için 1-2 kelimelik kısa bir onay mesajı yaz (Örn: "Emredersin!").

--- OYUN KILAVUZU ---
{guide_content}
"""

        user_prompt = f"""Oyun Bağlamı:
{game_context}

Oyuncunun komutu: '{transcription}'

SADECE gecerli bir JSON dondur."""
        
        # LLM'den yanıt al
        response = llm_model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_object",
                "schema": json_schema
            },
            temperature=0.1,
            max_tokens=1024
        )
        
        llm_output = response["choices"][0]["message"]["content"]
        command_data = json.loads(llm_output)
        
        return {
            "transcription": transcription,
            "commands": command_data.get("commands", []),
            "status": "success",
            "error_message": None
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "transcription": locals().get("transcription", ""),
            "command": None,
            "status": "error",
            "error_message": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
