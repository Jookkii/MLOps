import requests


prompt = ""
print("type in your prompt")

current_ip = ""
if prompt == "":
    prompt = input()

print("wysyłanie requesta do vm/generate...")


try:
    # BentoML przyjmuje JSON-a, więc zamieniamy tablicę numpy na listę Pythona
    url = "http://"+current_ip+":3000/generate"
    response = requests.post(
        url,
        json={"prompt": prompt}, # Nazwa klucza musi pasować do argumentu w service.py
        timeout=10
    )

    # 4. Odczytanie odpowiedzi
    if response.status_code == 200:
        answer = response.text
        last_dot_index = answer.rfind('.')

        if last_dot_index != -1:
            # Wycinamy tekst od początku do ostatniej kropki (włącznie z nią)
            clean_answer = answer[:last_dot_index + 1]
        else:
            clean_answer = answer
        # Wynik z modelu to zazwyczaj logits lub softmax (10 wartości)
        # Wybieramy indeks z najwyższą wartością
        print(f"prompt{prompt}")
        print("answer")
        print(clean_answer)
        
            
    else:
        print(f"Błąd serwera! Kod statusu: {response.status_code}")
        print(f"Treść błędu: {response.text}")

except requests.exceptions.ConnectionError:
    print("Błąd: Nie można połączyć się z serwerem. Upewnij się, że 'bentoml serve' działa!")