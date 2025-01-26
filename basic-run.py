from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
start_time = time.time()
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for better performance on GPU
    device_map="auto"  # Let accelerate handle device placement
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model loaded time
model_load_time = time.time() 
print("Model loaded in",model_load_time  - start_time, "seconds")
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Move inputs to GPU
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

# Generate text
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
]

# Decode the generated text
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text)
print("Total prompt response taken:", time.time() - model_load_time, "seconds")
