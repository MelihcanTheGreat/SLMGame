import sounddevice as sd
from scipy.io.wavfile import write
import requests
import time
import json

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
DURATION = 5  # Saniye cinsinden kayıt süresi
SAMPLE_RATE = 44100
FILENAME = "test_record.wav"

print(f"\n🎤 Lütfen konuşun... (Kayıt {DURATION} saniye sürecek)")
print(f"Örnek: '5 numaralı karakter buraya gel'")
print("-" * 40)

# Sesi kaydet
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
# Kullanıcıya kaydın başladığını hissettirmek için saniye sayalım
for i in range(DURATION, 0, -1):
    print(f"Kayıt bitmesine: {i} saniye...", end="\r")
    time.sleep(1)

sd.wait()  # İşlemin tam bitmesini garantiye al
print("\n✅ Kayıt bitti, API'ye gönderiliyor (STT ve LLM çalışıyor). Lütfen bekleyin...")

# Sesi geçici olarak dosyaya yaz
write(FILENAME, SAMPLE_RATE, audio_data)

# API'ye POST isteği gönder
try:
    with open(FILENAME, "rb") as f:
        files = {"audio": (FILENAME, f, "audio/wav")}
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
            print(f"⏱️  Toplam İşlem Süresi: {end_time - start_time:.2f} saniye")
            print("="*50)
        else:
            print(f"\n❌ Hata! Sunucu {response.status_code} döndürdü:")
            print(response.text)
except Exception as e:
    print(f"\n❌ Bağlantı hatası: Sunucu çalışmıyor olabilir. (Hata: {e})")
