import sounddevice as sd
from scipy.io.wavfile import write
import requests
import time
import json
import numpy as np
import io

# --- Ayarlar ---
API_URL = "http://127.0.0.1:8000/api/v1/process-voice"
# --- Oyun Context'i (Sanki Unity'den geliyormuş gibi) ---
GAME_CONTEXT = json.dumps({
    "pointed_npc_id": 5,
    "visible_npcs": [
        {"id": 1, "name": "Veli"},
        {"id": 2, "name": "Hasan"},
        {"id": 5, "name": "Okçu Ali"}
    ]
})

SAMPLE_RATE = 16000 # Whisper için ideal oran 16000'dir
SILENCE_THRESHOLD = 0.03 # Ses seviyesi (mikrofona göre ayarlanabilir)
SILENCE_DURATION = 0.7   # Konuşma sonrası beklenecek sessizlik süresi (saniye)

def record_until_silence():
    print("\n🎤 Lütfen konuşun... (Konuşmanız bitince otomatik algılanacak)")
    print(f"Örnek: '5 numaralı karakter buraya gel'")
    print("-" * 40)
    
    audio_buffer = []
    silent_chunks = 0
    chunk_size = int(SAMPLE_RATE * 0.1) # 100ms
    max_silence_chunks = int(SILENCE_DURATION / 0.1)
    has_spoken = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            audio_chunk, overflowed = stream.read(chunk_size)
            volume = np.max(np.abs(audio_chunk))
            
            if volume > SILENCE_THRESHOLD:
                has_spoken = True
                silent_chunks = 0
            else:
                if has_spoken:
                    silent_chunks += 1
            
            # Başta çok az sessizlik kaydedelim, konuşma başlayınca asıl kayda alalım
            if has_spoken:
                audio_buffer.append(audio_chunk)
                
            if has_spoken and silent_chunks >= max_silence_chunks:
                print("✅ Konuşma bittiği algılandı, API'ye gönderiliyor...")
                break
                
    return np.concatenate(audio_buffer, axis=0)

audio_data = record_until_silence()

# Sesi hafızada WAV formatına çevir
wav_io = io.BytesIO()
# numpy float32'yi int16'ya çeviriyoruz (WAV için daha güvenli)
audio_int16 = (audio_data * 32767).astype(np.int16)
write(wav_io, SAMPLE_RATE, audio_int16)
wav_io.seek(0)

# API'ye POST isteği gönder
try:
    files = {"audio": ("record.wav", wav_io, "audio/wav")}
    data = {"game_context": GAME_CONTEXT}
    
    start_time = time.time()
    response = requests.post(API_URL, files=files, data=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print("\n" + "="*50)
        print("🗣️  SES TANIMA (STT) SONUCU:")
        print(f"   ({result['transcription']})")
        print("-" * 50)
        print("🤖 YAPAY ZEKA (LLM) ANLADIĞI KOMUT (JSON):")
        print(json.dumps(result.get('commands', []), indent=2, ensure_ascii=False))
        print("-" * 50)
        print(f"⏱️  Toplam İşlem Süresi (API): {end_time - start_time:.2f} saniye")
        print("="*50)
    else:
        print(f"\n❌ Hata! Sunucu {response.status_code} döndürdü:")
        print(response.text)
except Exception as e:
    print(f"\n❌ Bağlantı hatası: Sunucu çalışmıyor olabilir. (Hata: {e})")
