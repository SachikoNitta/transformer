import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

load_dotenv()
model_id = os.getenv("MODEL_ID")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "User: Hi!\nAssistant:"
response = chat(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
print(response[0]["generated_text"])