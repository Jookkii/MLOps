from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "./moj_model_lokalnie"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

model.save_pretrained("./moj_model_lokalnie")
tokenizer.save_pretrained("./moj_model_lokalnie")

messages = [{"role": "user", "content": "Explain gravity in one sentence."}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=200)
print("---------------------------")
print(tokenizer.decode(outputs[0]))