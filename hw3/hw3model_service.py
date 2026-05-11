import bentoml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Definiujemy nazwę i konfigurację serwisu
@bentoml.service(
    name="smollm_service",
    traffic={"timeout": 60},
    resources={"cpu": "2"} # Możesz też dodać gpu: 1, jeśli masz kartę
)
class SmolLMService:
    def __init__(self):
        # Ścieżka do modelu na HuggingFace
        self.model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
        
        # Ładowanie tokenizera i modelu
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        
        # Przeniesienie na GPU jeśli dostępne
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @bentoml.api
    def generate(self, prompt: str) -> str:
        # Formatowanie zgodnie z template'em SmolLM
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Generowanie odpowiedzi
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=150, 
                temperature=0.3,
                do_sample=True
            )
        
        # Dekodowanie (pomijając prompt wejściowy)
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Wycinamy część z instrukcją użytkownika, żeby zwrócić samą odpowiedź asystenta
        return full_text.split("assistant")[-1].strip()