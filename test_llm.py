import os
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import json
import time

model_path = hf_hub_download(
    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
)

llm_model = Llama(
    model_path=model_path,
    n_ctx=1024,
    n_gpu_layers=-1,
    verbose=False
)

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

game_context = json.dumps({
    "pointed_npc_id": 5,
    "visible_npcs": [
        {"id": 1, "name": "Veli"},
        {"id": 3, "name": "Mehmet"}
    ]
})

user_prompt = f"Context: {game_context}\nCommand: '3 numara odun toplamaya git.'\nJSON:"

t0 = time.time()
response = llm_model.create_chat_completion(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.1,
    max_tokens=192,
    response_format={"type": "json_object"}
)
t1 = time.time()

llm_output = response["choices"][0]["message"]["content"]
print("\n--- LLM CIKTISI ---")
print(llm_output)
print(f"Sure: {t1-t0:.2f}s")
