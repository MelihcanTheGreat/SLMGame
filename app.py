import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile


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
        
    # 2. Faster-Whisper'ı yükle (CUDA dll'leri eksik olduğu için CPU moduna zorluyoruz)
    try:
        print("Faster-Whisper 'cpu' (int8) ile başlatılıyor...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        device = "cpu"
    except Exception as e:
        print(f"Whisper yükleme hatası: {e}")
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
        
    temp_file_path = ""
    try:
        # Sesi geçici dosyaya kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        # 1. STT: Ses -> Metin
        segments, info = whisper_model.transcribe(temp_file_path, language="tr", beam_size=5)
        transcription = " ".join([segment.text for segment in segments]).strip()
        
        if not transcription:
            raise ValueError("Ses dosyasından metin anlaşılamadı veya boş.")
            
        # 2. SLM: Metin -> JSON Komut
        system_prompt = f"""Sen bir oyun içi yapay zeka asistanısın. Görevin oyuncunun verdiği sesli komutu analiz edip JSON formatında çıktı üretmektir.

Oyunun şu anki durumu (game_context) sana 'user' mesajında verilecektir. Oradaki 'visible_npcs' listesini kullanarak isimleri ID'lere dönüştür.

KURALLAR:
1. Eğer komutta bir isim geçiyorsa, visible_npcs içinden o ismin ID'sini bul ve "target_npc_id" yap.
2. İsim geçmiyorsa ve "pointed_npc_id" null değilse, komutu "pointed_npc_id" değerine ata. Hedef herkes ise "target_npc_id" değerini "all" yap.
3. Birden fazla hedefe farklı komutlar verilmişse "commands" listesine birden fazla obje ekle.
4. Hedef nesne veya kişi ismi YOKSA "target_object" kesinlikle "null" olmalıdır.
5. Bir X, Y veya Z koordinatı belirtilmişse "is_specified" true olmalı, bahsedilmeyenler 0 olmalıdır.
6. Koordinat veya sayı yoksa x,y,z 0 kalmalı, "is_specified" false olmalıdır.
7. "action_type" kılavuza uygun olmalıdır.
8. Her karakter için "npc_reply" alanına karakterin ağzından kısa ve eğlenceli bir cevap yaz (Örn: "Emredersin patron!", "Hemen odun kesmeye gidiyorum"). Cevap Türkçe olmalıdır.

--- OYUN KILAVUZU ---
{guide_content}
"""

        user_prompt = f"Oyun Bağlamı (game_context):\n{game_context}\n\nOyuncunun komutu: '{transcription}'"
        
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
            max_tokens=512
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
        
    finally:
        # Geçici dosyayı sil
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
