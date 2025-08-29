from llama_cpp import Llama

# Модель Mistral-7B-Instruct-v0.2 GGUF
# wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf
model_path = "./mistral-7b-instruct-v0.2.Q8_0.gguf"

print("Загрузка GGUF модели (Mistral-7B-Instruct)...")
llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=0, verbose=False)
print("Модель загружена.")

# У Mistral свой формат промпта
prompt = "<s>[INST] Расскажи короткую шутку про программистов. [/INST]"

print("\nГенерация ответа...")
output = llm(
    prompt,
    max_tokens=150,
    stop=["</s>"],
    echo=False
)

response_text = output['choices'][0]['text']

print("\nОтвет модели:")
print(response_text.strip())
