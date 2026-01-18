from huggingface_hub import InferenceClient
import os

QWEN = InferenceClient("Qwen/Qwen2-7B-Instruct", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
MISTRAL = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

def llm(prompt: str) -> str:
    try:
        r = QWEN.chat_completion(
            messages=[{"role":"user","content":prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        return r.choices[0].message["content"]
    except:
        r = MISTRAL.chat_completion(
            messages=[{"role":"user","content":prompt}],
            max_tokens=512,
            temperature=0.2,
        )
        return r.choices[0].message["content"]
