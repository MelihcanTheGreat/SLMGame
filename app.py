import os
import sys
import json
import time
import asyncio
from typing import Annotated
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import io

# --- Windows GPU DLL Düzeltmesi ---
if os.name == 'nt':
    try:
        import nvidia.cuda_runtime.lib
        import nvidia.cublas.lib  # type: ignore
        import nvidia.cudnn.lib  # type: ignore
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
        def read_guide():
            with open(guide_path, "r", encoding="utf-8") as f:
                return f.read()
        guide_content = await asyncio.to_thread(read_guide)
    except Exception as e:
        print(f"Uyarı: guide.md okunamadı: {e}")
        guide_content = "Kılavuz bulunamadı."
        
    # 2. Faster-Whisper'ı yükle (Önce CUDA deniyoruz, olmazsa CPU)
    # "base" model, "small"a göre ~3x daha hızlı, kısa komutlar için yeterli doğruluk
    try:
        print("Faster-Whisper 'base' modeli 'cuda' (float16) ile başlatılıyor...")
        whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    except Exception as e:
        print(f"GPU (CUDA) ile Whisper yükleme hatası: {e}")
        print("Faster-Whisper 'base' modeli 'cpu' (int8) moduna düşürülüyor...")
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        
    # 3. Llama.cpp (Llama-3.2-1B) yükle
    print("Llama-3.2-1B-Instruct-GGUF modeli kontrol ediliyor/indiriliyor...")
    try:
        model_path = hf_hub_download(
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        )
        print(f"LLM modeli bulundu: {model_path}")
        # n_gpu_layers=-1 ile modelin tamamı (mümkünse) GPU'ya yüklenir. 
        # CUDA veya Vulkan desteği yoksa kütüphane otomatik olarak işlemcide (CPU) çalıştırır.
        llm_model = Llama(
            model_path=model_path,
            n_ctx=1024,  # Kısa komutlar için 1024 yeterli (2048'den düşürüldü)
            n_gpu_layers=-1,
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
    audio: Annotated[UploadFile, File(...)],
    game_context: Annotated[str, Form(...)]
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
            
        # 1. STT: Ses -> Metin (beam_size=1 + vad_filter ile hızlandırıldı)
        t0 = time.perf_counter()
        segments, info = whisper_model.transcribe(
            audio_stream,
            language="tr",
            beam_size=1,
            vad_filter=True,          # Sessiz bölümleri atla
            vad_parameters={
                "min_silence_duration_ms": 300  # 300ms sessizlik = konuşma sonu
            }
        )
        transcription = " ".join([segment.text for segment in segments]).strip()
        t1 = time.perf_counter()
        stt_time = t1 - t0
        
        if not transcription:
            raise ValueError("Ses dosyasından metin anlaşılamadı veya boş.")
            
        # Ornek JSON ekleyerek 1B modelin dogru format uretmesi saglaniyor
        system_prompt = """Return ONLY valid JSON. Format:
{
  "commands": [
    {
      "target_npc_id": 1,
      "action_type": "move",
      "target_object": null,
      "target_location": {"x": 0, "y": 0, "z": 0, "is_specified": false},
      "npc_reply": "OK"
    }
  ]
}
Actions: move,gather,attack,defend,idle,talk. Match NPC names to IDs from visible_npcs. If no name, use pointed_npc_id. If no coordinates, set x,y,z=0 and is_specified=false."""

        user_prompt = f"Context: {game_context}\nCommand: '{transcription}'\nJSON:"
        
        # LLM'den yanıt al
        t2 = time.perf_counter()
        
        response = llm_model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=192,
            response_format={"type": "json_object"}
        )
        llm_output = response["choices"][0]["message"]["content"]
        
        # JSON'i ciktidan ayikla (model bazen ekstra metin ekleyebilir)
        # raw_decode sadece ilk JSON objesini parse eder, gerisini yok sayar
        start_idx = llm_output.find("{")
        if start_idx != -1:
            try:
                decoder = json.JSONDecoder()
                command_data, _ = decoder.raw_decode(llm_output, start_idx)
            except json.JSONDecodeError as jde:
                print(f"JSON Parse Hatası. LLM Çıktısı:\n{llm_output}")
                raise ValueError(f"Geçersiz JSON formatı: {jde}\nLLM Çıktısı: {llm_output}")
        else:
            raise ValueError(f"JSON bulunamadi: {llm_output[:200]}")
        
        t3 = time.perf_counter()
        llm_time = t3 - t2
        
        print(f"[TIMING] STT: {stt_time:.2f}s | LLM: {llm_time:.2f}s | Toplam: {stt_time+llm_time:.2f}s")
        
        return {
            "transcription": transcription,
            "commands": command_data.get("commands", []),
            "status": "success",
            "error_message": None,
            "timing": {"stt_seconds": round(stt_time, 2), "llm_seconds": round(llm_time, 2)}
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "transcription": locals().get("transcription", ""),
            "command": None,
            "status": "error",
            "error_message": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
