# import requests
# import json

# OLLAMA_HOST = "http://192.168.3.127:11434"  # replace with your Windows host IP

# payload = {
#     "model": "phi3:latest",
#     "prompt": "Explain quantum computing in simple terms."
# }

# with requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=True) as response:
#     for line in response.iter_lines():
#         if line:
#             data = json.loads(line.decode('utf-8'))
#             # Print only the text output as it arrives
#             if "response" in data:
#                 print(data["response"], end="", flush=True)

# print()  # newline at end
import os
from dotenv import load_dotenv
load_dotenv()
import ollama

print("Host from env:", os.getenv("OLLAMA_HOST"))
print(ollama.chat(
    model=os.getenv("OLLAMA_MODEL", "phi3:latest"),
    messages=[{"role": "user", "content": "explain bodmas"}],
))