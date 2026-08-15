import requests, json, time

API_URL = "http://127.0.0.1:8000/api/v1/process-voice"
GAME_CONTEXT = json.dumps({
    "pointed_npc_id": 5,
    "visible_npcs": [
        {"id": 1, "name": "Veli"},
        {"id": 2, "name": "Hasan"},
        {"id": 5, "name": "Okcu Ali"}
    ]
})

print("=== Performans Testi (3 deneme) ===\n")
for i in range(3):
    with open("test_record.wav", "rb") as f:
        files = {"audio": ("test_record.wav", f, "audio/wav")}
        data = {"game_context": GAME_CONTEXT}
        start = time.time()
        r = requests.post(API_URL, files=files, data=data)
        elapsed = time.time() - start

    result = r.json()
    timing = result.get("timing", {})
    stt = timing.get("stt_seconds", "?")
    llm = timing.get("llm_seconds", "?")
    status = result.get("status", "?")
    err = result.get("error_message", "")
    print(f"Deneme {i+1}: STT={stt}s | LLM={llm}s | Toplam={elapsed:.2f}s | Status={r.status_code} | Metin: {result.get('transcription', '?')}")
    if err:
        print(f"  HATA: {err}")
